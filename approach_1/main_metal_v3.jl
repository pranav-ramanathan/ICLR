#!/usr/bin/env julia
#=
N3L Pure Gradient Flow — Metal GPU Accelerated (Line-Based RK4, v3)
===================================================================
This version uses line representation and adds collinearity normalization
to stabilize alpha across larger board sizes.

Key improvements vs `main_metal_v2.jl`:
  - Per-point collinearity normalization (`mean-incidence`) in-kernel
  - New coefficient mode `normalized` (default)
  - Better portability of alpha as n grows
=#

using Metal
using KernelAbstractions
using Random
using Printf
using Dates
using ArgParse

const MAX_STATE_DIM = 1024  # kernel private scratch limit (supports n <= 32)

# ============================================================================
# Deterministic RNG Helpers
# ============================================================================

@inline function splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    z = x
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

# ============================================================================
# Top-k Mask Helper (CPU)
# ============================================================================

function topk_mask(x::AbstractVector{<:Real}, k::Int)
    idx = partialsortperm(x, 1:k; rev=true)
    m = falses(length(x))
    @inbounds for i in idx
        m[i] = true
    end
    return BitVector(m)
end

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef struct Config
    n::Int
    R::Int = 5000
    T::Float32 = 10.0f0
    dt::Float32 = 0.005f0
    α::Float32 = 10.0f0
    β::Float32 = 1.0f0
    γ::Float32 = 4.5f0
end

@inline function legacy_coefficients(n::Int)
    α = n <= 10 ? Float32(10.0 * (n / 6)) : 40.0f0
    γ = n <= 10 ? 5.0f0 : 15.0f0
    return α, γ
end

function density_coefficients(n::Int, line_offsets::Vector{Int32})
    if n <= 10
        α, γ = legacy_coefficients(n)
        return α, γ, NaN, NaN
    end

    N = n * n
    L = length(line_offsets) - 1

    triple_incidence_sum = 0.0
    @inbounds for l in 1:L
        len = Int(line_offsets[l + 1] - line_offsets[l])
        # Sum_i C(len-1,2) over points on this line.
        triple_incidence_sum += len * ((len - 1) * (len - 2) / 2)
    end

    avg_triples_per_var = triple_incidence_sum / max(1, N)
    target_density = 2.0 / n
    col_scale = avg_triples_per_var * target_density * target_density

    # Empirical scaling from metal runs n=10..16:
    # keep alpha moderate; very high alpha tends to freeze bad local minima.
    α = clamp(105.0 / max(col_scale, 1e-6), 10.0, 28.0)

    # Strong gamma (>~10) was consistently harmful on metal for n>=14.
    γ = if n <= 16
        4.5
    else
        4.0
    end

    return Float32(α), Float32(γ), avg_triples_per_var, col_scale
end

@inline function normalized_coefficients(n::Int, normalization_mode::String)
    if n <= 10
        return legacy_coefficients(n)
    end

    norm_mode_l = lowercase(normalization_mode)

    # With mean-incidence normalization, alpha can stay in a narrow range.
    α = if norm_mode_l == "mean-incidence"
        10.0f0
    else
        # If normalization is disabled, fall back to a conservative alpha.
        n <= 16 ? 10.0f0 : 8.0f0
    end

    γ = n <= 16 ? 4.5f0 : 4.0f0
    return α, γ
end

function choose_coefficients(
    n::Int,
    line_offsets::Vector{Int32};
    mode::String="normalized",
    normalization_mode::String="mean-incidence",
    alpha_override=nothing,
    gamma_override=nothing,
)
    mode_l = lowercase(mode)

    base_α::Float32 = 0.0f0
    base_γ::Float32 = 0.0f0
    avg_triples_per_var = NaN
    col_scale = NaN

    if mode_l == "legacy"
        base_α, base_γ = legacy_coefficients(n)
    elseif mode_l == "density"
        base_α, base_γ, avg_triples_per_var, col_scale = density_coefficients(n, line_offsets)
    elseif mode_l == "normalized"
        base_α, base_γ = normalized_coefficients(n, normalization_mode)
    else
        error("Invalid --coeff-mode '$mode'. Expected one of: legacy, density, normalized")
    end

    α = isnothing(alpha_override) ? base_α : Float32(alpha_override)
    γ = isnothing(gamma_override) ? base_γ : Float32(gamma_override)

    return α, γ, base_α, base_γ, mode_l, avg_triples_per_var, col_scale
end

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_cli_args(args)
    s = ArgParseSettings(
        description = "N3L Gradient Flow — Metal GPU (Line-Based Custom Kernel v3)",
        version = "4.0.0",
        add_version = true
    )

    @add_arg_table! s begin
        "n"
            help = "Board size (n x n grid)"
            arg_type = Int
            required = true
        "--R"
            help = "Number of trajectories to try"
            arg_type = Int
            default = 5000
        "--T"
            help = "Integration horizon"
            arg_type = Float64
            default = 10.0
        "--dt"
            help = "RK4 timestep"
            arg_type = Float64
            default = 0.005
        "--alpha"
            help = "Override alpha penalty coefficient"
            arg_type = Float64
        "--gamma"
            help = "Override gamma binary regularization"
            arg_type = Float64
        "--coeff-mode"
            help = "Coefficient strategy when not overridden: normalized | legacy | density"
            arg_type = String
            default = "normalized"
        "--col-normalization"
            help = "Collinearity normalization: mean-incidence | none"
            arg_type = String
            default = "mean-incidence"
        "--seed"
            help = "Random seed"
            arg_type = UInt64
        "--outdir"
            help = "Output directory"
            arg_type = String
            default = "solutions"
        "--quiet", "-q"
            help = "Suppress output"
            action = :store_true
        "--batch-size"
            help = "Trajectories per GPU batch"
            arg_type = Int
            default = 1024
        "--max-batches"
            help = "Maximum number of batches (0 = unlimited until R)"
            arg_type = Int
            default = 0
    end

    return parse_args(args, s)
end

# ============================================================================
# Precompute Collinear Lines (Exact Triple-Equivalent Representation)
# ============================================================================

@inline function normalize_direction(dx::Int, dy::Int)
    g = gcd(abs(dx), abs(dy))
    dx = div(dx, g)
    dy = div(dy, g)

    # Canonical sign so each geometric line has one direction key.
    if dy < 0 || (dy == 0 && dx < 0)
        dx = -dx
        dy = -dy
    end
    return dx, dy
end

function compute_lines(n::Int)
    # Key: (dx, dy, c) where c = dy*x - dx*y for points on the line.
    line_sets = Dict{NTuple{3,Int32}, BitSet}()

    for x1 in 1:n, y1 in 1:n
        p1 = (x1 - 1) * n + y1
        for x2 in 1:n, y2 in 1:n
            (x2, y2) <= (x1, y1) && continue
            dx, dy = normalize_direction(x2 - x1, y2 - y1)
            c = dy * x1 - dx * y1

            key = (Int32(dx), Int32(dy), Int32(c))
            pts = get!(line_sets, key, BitSet())
            push!(pts, p1)
            push!(pts, (x2 - 1) * n + y2)
        end
    end

    keys_sorted = sort!(collect(keys(line_sets)))

    line_offsets = Int32[1]  # CSR offsets, 1-based
    line_points = Int32[]

    for key in keys_sorted
        pts = sort!(collect(line_sets[key]))
        length(pts) < 3 && continue
        append!(line_points, Int32.(pts))
        push!(line_offsets, Int32(length(line_points) + 1))
    end

    return line_offsets, line_points
end

function line_stats(line_offsets::Vector{Int32})
    L = length(line_offsets) - 1
    sum_len = 0
    max_len = 0
    triples_equiv = 0

    @inbounds for l in 1:L
        len = Int(line_offsets[l + 1] - line_offsets[l])
        sum_len += len
        max_len = max(max_len, len)
        triples_equiv += (len * (len - 1) * (len - 2)) ÷ 6
    end

    return L, sum_len, max_len, triples_equiv
end

function compute_point_collinearity_scale(
    n::Int,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32};
    mode::String="mean-incidence",
)
    mode_l = lowercase(mode)
    N = n * n

    if mode_l == "none"
        return ones(Float32, N), NaN, NaN, NaN, mode_l
    elseif mode_l != "mean-incidence"
        error("Invalid --col-normalization '$mode'. Expected one of: mean-incidence, none")
    end

    L = length(line_offsets) - 1
    incidence = zeros(Float64, N)

    @inbounds for l in 1:L
        start_idx = Int(line_offsets[l])
        stop_idx = Int(line_offsets[l + 1] - 1)
        len = stop_idx - start_idx + 1
        len < 3 && continue

        # Number of line-local pairs that combine with this point into triples.
        local_inc = (len - 1) * (len - 2) / 2
        for idx in start_idx:stop_idx
            p = Int(line_points[idx])
            incidence[p] += local_inc
        end
    end

    mean_inc = sum(incidence) / max(1, N)
    min_inc = minimum(incidence)
    max_inc = maximum(incidence)

    if mean_inc <= 0
        return ones(Float32, N), mean_inc, min_inc, max_inc, mode_l
    end

    scale = Vector{Float32}(undef, N)
    @inbounds for i in 1:N
        denom = max(1.0, incidence[i])
        scale[i] = Float32(mean_inc / denom)
    end

    return scale, mean_inc, min_inc, max_inc, mode_l
end

# ============================================================================
# GPU Kernel: RK4 Gradient Flow Integrator (Line-Based)
# ============================================================================
# Each thread integrates one trajectory from t=0 to t=T using fixed-step RK4.
# The line-based gradient is exact for the triple energy:
#   For point i on a line with point values {x_j}, contribution is
#   sum_{j<k, j!=i, k!=i} x_j*x_k
#   = ((S1-xi)^2 - (S2-xi^2)) / 2, where S1=sum(x), S2=sum(x^2) on that line.

@kernel function rk4_gradient_flow_lines_kernel!(
    state,               # N × R matrix (Float32) — state, modified in-place
    @Const(line_offsets),# (L+1) vector (Int32) CSR offsets
    @Const(line_points), # packed point indices for all lines (Int32)
    @Const(point_col_scale), # N vector (Float32) pointwise collinearity scale
    α::Float32,
    β::Float32,
    γ::Float32,
    dt::Float32,
    nsteps::Int32,
    N::Int32,            # state dimension (n^2)
    L::Int32             # number of lines
)
    traj = @index(Global, Linear)

    # Temporary arrays in private memory.
    k1 = @private Float32 (MAX_STATE_DIM,)
    k2 = @private Float32 (MAX_STATE_DIM,)
    k3 = @private Float32 (MAX_STATE_DIM,)
    k4 = @private Float32 (MAX_STATE_DIM,)
    xtmp = @private Float32 (MAX_STATE_DIM,)

    for step in Int32(1):nsteps
        # ---- k1 = f(x) ----
        for i in Int32(1):N
            xi = state[i, traj]
            @inbounds k1[i] = β - γ * xi * (2.0f0 - 6.0f0 * xi + 4.0f0 * xi * xi)
        end

        for l in Int32(1):L
            start_idx = @inbounds line_offsets[l]
            stop_idx = @inbounds line_offsets[l + Int32(1)] - Int32(1)

            s1 = 0.0f0
            s2 = 0.0f0
            for idx in start_idx:stop_idx
                p = @inbounds line_points[idx]
                x = state[p, traj]
                s1 += x
                s2 += x * x
            end

            for idx in start_idx:stop_idx
                p = @inbounds line_points[idx]
                xi = state[p, traj]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5f0 * (s1o * s1o - s2o)
                scale = @inbounds point_col_scale[p]
                @inbounds k1[p] -= α * scale * pair_sum
            end
        end

        # ---- k2 = f(x + dt/2 * k1) ----
        half_dt = dt * 0.5f0
        for i in Int32(1):N
            @inbounds begin
                xi_new = state[i, traj] + half_dt * k1[i]
                xtmp[i] = min(1.0f0, max(0.0f0, xi_new))
            end
        end

        for i in Int32(1):N
            xi = @inbounds xtmp[i]
            @inbounds k2[i] = β - γ * xi * (2.0f0 - 6.0f0 * xi + 4.0f0 * xi * xi)
        end

        for l in Int32(1):L
            start_idx = @inbounds line_offsets[l]
            stop_idx = @inbounds line_offsets[l + Int32(1)] - Int32(1)

            s1 = 0.0f0
            s2 = 0.0f0
            for idx in start_idx:stop_idx
                p = @inbounds line_points[idx]
                x = @inbounds xtmp[p]
                s1 += x
                s2 += x * x
            end

            for idx in start_idx:stop_idx
                p = @inbounds line_points[idx]
                xi = @inbounds xtmp[p]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5f0 * (s1o * s1o - s2o)
                scale = @inbounds point_col_scale[p]
                @inbounds k2[p] -= α * scale * pair_sum
            end
        end

        # ---- k3 = f(x + dt/2 * k2) ----
        for i in Int32(1):N
            @inbounds begin
                xi_new = state[i, traj] + half_dt * k2[i]
                xtmp[i] = min(1.0f0, max(0.0f0, xi_new))
            end
        end

        for i in Int32(1):N
            xi = @inbounds xtmp[i]
            @inbounds k3[i] = β - γ * xi * (2.0f0 - 6.0f0 * xi + 4.0f0 * xi * xi)
        end

        for l in Int32(1):L
            start_idx = @inbounds line_offsets[l]
            stop_idx = @inbounds line_offsets[l + Int32(1)] - Int32(1)

            s1 = 0.0f0
            s2 = 0.0f0
            for idx in start_idx:stop_idx
                p = @inbounds line_points[idx]
                x = @inbounds xtmp[p]
                s1 += x
                s2 += x * x
            end

            for idx in start_idx:stop_idx
                p = @inbounds line_points[idx]
                xi = @inbounds xtmp[p]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5f0 * (s1o * s1o - s2o)
                scale = @inbounds point_col_scale[p]
                @inbounds k3[p] -= α * scale * pair_sum
            end
        end

        # ---- k4 = f(x + dt * k3) ----
        for i in Int32(1):N
            @inbounds begin
                xi_new = state[i, traj] + dt * k3[i]
                xtmp[i] = min(1.0f0, max(0.0f0, xi_new))
            end
        end

        for i in Int32(1):N
            xi = @inbounds xtmp[i]
            @inbounds k4[i] = β - γ * xi * (2.0f0 - 6.0f0 * xi + 4.0f0 * xi * xi)
        end

        for l in Int32(1):L
            start_idx = @inbounds line_offsets[l]
            stop_idx = @inbounds line_offsets[l + Int32(1)] - Int32(1)

            s1 = 0.0f0
            s2 = 0.0f0
            for idx in start_idx:stop_idx
                p = @inbounds line_points[idx]
                x = @inbounds xtmp[p]
                s1 += x
                s2 += x * x
            end

            for idx in start_idx:stop_idx
                p = @inbounds line_points[idx]
                xi = @inbounds xtmp[p]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5f0 * (s1o * s1o - s2o)
                scale = @inbounds point_col_scale[p]
                @inbounds k4[p] -= α * scale * pair_sum
            end
        end

        # ---- RK4 update + box constraint [0, 1] ----
        sixth_dt = dt / 6.0f0
        for i in Int32(1):N
            @inbounds begin
                xnew = state[i, traj] + sixth_dt * (k1[i] + 2.0f0 * k2[i] + 2.0f0 * k3[i] + k4[i])
                state[i, traj] = min(1.0f0, max(0.0f0, xnew))
            end
        end
    end
end

# ============================================================================
# Biased Initial Conditions (CPU, then transfer to GPU)
# ============================================================================

function generate_initial_conditions(n::Int, R::Int, seed::UInt64)
    N = n * n
    target_density = Float32(2 * n) / Float32(N)
    a = max(0.5f0, 2.0f0 * target_density)
    b = max(0.5f0, 2.0f0 * (1.0f0 - target_density))

    state = Matrix{Float32}(undef, N, R)
    for traj in 1:R
        traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(traj) << 1))
        rng = Xoshiro(traj_seed)
        for i in 1:N
            u = Float32(rand(rng))^(1.0f0 / a)
            v = Float32(rand(rng))^(1.0f0 / b)
            state[i, traj] = u / (u + v)
        end
    end
    return state
end

# ============================================================================
# Validation (CPU, line-based)
# ============================================================================

function count_violations_lines(x_bin::BitVector, line_offsets::Vector{Int32}, line_points::Vector{Int32})
    count = 0
    L = length(line_offsets) - 1

    @inbounds for l in 1:L
        c = 0
        start_idx = line_offsets[l]
        stop_idx = line_offsets[l + 1] - 1
        for idx in start_idx:stop_idx
            c += x_bin[line_points[idx]]
        end
        if c >= 3
            count += (c * (c - 1) * (c - 2)) ÷ 6
        end
    end

    return count
end

# ============================================================================
# GPU Solver
# ============================================================================

function solve_metal(
    cfg::Config,
    line_offsets_cpu::Vector{Int32},
    line_points_cpu::Vector{Int32},
    point_col_scale_cpu::Vector{Float32},
    seed::UInt64,
    outdir::String;
    verbose::Bool=true,
    batch_size::Int=1024,
    max_batches::Int=0,
    col_normalization::String="mean-incidence",
)
    n = cfg.n
    N = Int32(n * n)
    L = Int32(length(line_offsets_cpu) - 1)
    target = 2 * n
    nsteps = Int32(max(1, floor(Int, cfg.T / cfg.dt)))
    effective_T = Float32(nsteps) * cfg.dt

    if Int(N) > MAX_STATE_DIM
        println("ERROR: n^2=$(Int(N)) exceeds MAX_STATE_DIM=$MAX_STATE_DIM in kernel scratch memory.")
        println("       Increase MAX_STATE_DIM (with performance tradeoff) or reduce n.")
        return false, nothing, 0.0, Dict(:success => 0, :tried => 0), seed
    end

    line_count, packed_points, max_line_len, triples_equiv = line_stats(line_offsets_cpu)

    verbose && println("="^60)
    verbose && println("N3L Gradient Flow — Metal GPU (Line-Based RK4 Kernel v3)")
    verbose && println("="^60)
    verbose && @printf("n=%d, target=%d | α=%.3f, β=%.1f, γ=%.3f\n", n, target, cfg.α, cfg.β, cfg.γ)
    verbose && @printf("Max: %d traj, T=%.3fs (effective %.3fs), dt=%.4f, steps=%d\n",
                       cfg.R, cfg.T, effective_T, cfg.dt, nsteps)
    verbose && @printf("Batch size: %d | Seed: %d\n", batch_size, seed)
    verbose && println("-"^60)
    verbose && @printf("Lines: %d | Packed points: %d | max line len: %d | triple-equivalent terms: %d\n",
                       line_count, packed_points, max_line_len, triples_equiv)
    verbose && @printf("Collinearity normalization: %s\n", col_normalization)
    verbose && @printf("State dim: %d\n", N)
    verbose && println("-"^60)

    d_line_offsets = MtlArray(line_offsets_cpu)
    d_line_points = MtlArray(line_points_cpu)
    d_point_col_scale = MtlArray(point_col_scale_cpu)

    backend = Metal.MetalBackend()
    kern = rk4_gradient_flow_lines_kernel!(backend, 64)

    verbose && print("Warming up Metal GPU (compiling kernel)...")
    try
        warmup_state = MtlArray(rand(Float32, N, 2))
        kern(warmup_state, d_line_offsets, d_line_points, d_point_col_scale,
             cfg.α, cfg.β, cfg.γ, cfg.dt, Int32(1), N, L;
             ndrange=2)
        KernelAbstractions.synchronize(backend)
        verbose && println("\n  ✓ Metal warmup OK")
    catch e
        verbose && println("\n  ⚠ Metal warmup failed: $(typeof(e)): $(sprint(showerror, e))")
        verbose && println("  This is a fatal error — kernel cannot compile.")
        return false, nothing, 0.0, Dict(:success => 0, :tried => 0), seed
    end
    verbose && println("-"^60)

    total_tried = 0
    total_batches = cld(cfg.R, batch_size)
    if max_batches > 0
        total_batches = min(total_batches, max_batches)
    end
    total_traj = min(cfg.R, total_batches * batch_size)

    solution_found = false
    solution_grid = nothing
    solution_traj_id = 0
    best_viols = typemax(Int)
    violation_histogram = Dict{Int, Int}()

    start_time = time()

    for batch_idx in 1:total_batches
        this_batch = min(batch_size, total_traj - total_tried)
        this_batch <= 0 && break

        verbose && @printf("[Batch %d/%d] Launching %d trajectories on GPU...\n",
                           batch_idx, total_batches, this_batch)

        batch_seed = splitmix64(seed ⊻ UInt64(batch_idx) ⊻ (UInt64(0xBA7C4) << 16))
        ic_cpu = generate_initial_conditions(n, this_batch, batch_seed)
        d_state = MtlArray(ic_cpu)

        kern(d_state, d_line_offsets, d_line_points, d_point_col_scale,
             cfg.α, cfg.β, cfg.γ, cfg.dt, nsteps, N, L;
             ndrange=this_batch)
        KernelAbstractions.synchronize(backend)

        result_cpu = Array(d_state)

        batch_best = typemax(Int)
        for traj in 1:this_batch
            x_final = @view result_cpu[:, traj]
            x_bin = topk_mask(x_final, target)
            viols = count_violations_lines(x_bin, line_offsets_cpu, line_points_cpu)

            violation_histogram[viols] = get(violation_histogram, viols, 0) + 1

            if viols < batch_best
                batch_best = viols
            end

            if viols == 0 && !solution_found
                solution_found = true
                solution_grid = reshape(x_bin, (n, n))
                solution_traj_id = total_tried + traj
                elapsed = time() - start_time
                verbose && @printf("  SOLUTION! batch=%d, traj=%d, time=%.2fs\n", batch_idx, traj, elapsed)
            end
        end

        total_tried += this_batch

        if batch_best < best_viols
            best_viols = batch_best
        end

        elapsed = time() - start_time
        rate = total_tried / elapsed
        verbose && @printf("  Best violations: %d | %d tried | %.1f traj/s\n", best_viols, total_tried, rate)

        if solution_found
            break
        end
    end

    elapsed = time() - start_time
    verbose && println("-"^60)

    if solution_found
        verbose && println("SUCCESS")
        verbose && @printf("Time: %.2fs | Tried: %d/%d (%.1f%%) | Rate: %.1f/s\n",
                           elapsed, total_tried, total_traj,
                           100 * total_tried / max(1, total_traj), total_tried / elapsed)

        if !isempty(violation_histogram)
            verbose && println("\nViolation distribution:")
            for v in sort!(collect(keys(violation_histogram)))
                count = violation_histogram[v]
                pct = 100 * count / total_tried
                bar = repeat("#", min(40, round(Int, pct / 2)))
                verbose && @printf("  %2d violations: %5d (%5.2f%%) %s\n", v, count, pct, bar)
            end
        end

        verbose && println("\nSolution:")
        print_grid(solution_grid)
        save_solution(
            n, solution_grid, solution_traj_id, cfg.R, cfg.T, seed, cfg.α, cfg.γ, outdir;
            col_normalization=col_normalization
        )

        return true, solution_grid, elapsed, Dict(:success => 1, :tried => total_tried), seed
    else
        verbose && println("NO SOLUTION FOUND")
        verbose && @printf("Time: %.2fs | Tried: %d | Best: %d viols | Rate: %.1f/s\n",
                           elapsed, total_tried, best_viols, total_tried / elapsed)

        if !isempty(violation_histogram)
            verbose && println("\nViolation distribution:")
            for v in sort!(collect(keys(violation_histogram)))
                count = violation_histogram[v]
                pct = 100 * count / total_tried
                bar = repeat("#", min(40, round(Int, pct / 2)))
                verbose && @printf("  %2d violations: %5d (%5.2f%%) %s\n", v, count, pct, bar)
            end
        end

        return false, nothing, elapsed, Dict(:success => 0, :tried => total_tried), seed
    end
end

# ============================================================================
# Utility
# ============================================================================

function print_grid(grid)
    n = size(grid, 1)
    for i in 1:n
        print("  ")
        for j in 1:n
            print(grid[i, j] ? "● " : "· ")
        end
        println()
    end
end

function save_solution(n, grid, traj_id, R, T, seed, α, γ, outdir; col_normalization::String="mean-incidence")
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "$(outdir)/$(n)"
    mkpath(dir)
    filename = "$(dir)/sol_metal_v3_$(timestamp)_traj$(traj_id).txt"

    open(filename, "w") do io
        println(io, "# n=$(n)")
        println(io, "# target=$(2n)")
        println(io, "# trajectory_id=$(traj_id)")
        println(io, "# R=$(R)")
        println(io, "# T=$(T)")
        println(io, "# seed=$(seed)")
        println(io, "# α=$(α), β=1.0, γ=$(γ)")
        println(io, "# method=Metal GPU Line-Based Custom RK4 Kernel (v3)")
        println(io, "# col_normalization=$(col_normalization)")
        println(io, "# timestamp=$(Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"))")
        println(io, "#")
        println(io, "# Grid (0/1):")
        for i in 1:n
            println(io, join(Int.(grid[i, :]), " "))
        end
        println(io, "#")
        println(io, "# Coordinates (row, col):")
        for i in 1:n, j in 1:n
            grid[i, j] && println(io, "($i, $j)")
        end
    end

    println("  Saved: $filename")
end

# ============================================================================
# Main
# ============================================================================

function main()
    args = parse_cli_args(ARGS)

    n = args["n"]
    R = args["R"]
    T = Float32(args["T"])
    dt = Float32(args["dt"])
    outdir = args["outdir"]
    quiet = args["quiet"]
    verbose = !quiet
    batch_size = args["batch-size"]
    max_batches = args["max-batches"]
    coeff_mode = args["coeff-mode"]
    col_norm_mode = args["col-normalization"]

    if n < 2
        println("ERROR: n must be >= 2")
        return 1
    end
    if R <= 0
        println("ERROR: R must be > 0")
        return 1
    end
    if T <= 0
        println("ERROR: T must be > 0")
        return 1
    end
    if dt <= 0
        println("ERROR: dt must be > 0")
        return 1
    end
    if batch_size <= 0
        println("ERROR: batch-size must be > 0")
        return 1
    end
    if max_batches < 0
        println("ERROR: max-batches must be >= 0")
        return 1
    end
    if !(lowercase(col_norm_mode) in ("mean-incidence", "none"))
        println("ERROR: col-normalization must be one of: mean-incidence, none")
        return 1
    end

    seed = if isnothing(args["seed"])
        rand(RandomDevice(), UInt64)
    else
        args["seed"]
    end

    verbose && println("Checking Metal GPU...")
    try
        dev = Metal.current_device()
        verbose && println("  Device: $(dev)")
    catch e
        println("ERROR: No Metal GPU available: $e")
        return 1
    end

    verbose && println("Precomputing lines...")
    t0_lines = time()
    line_offsets_cpu, line_points_cpu = compute_lines(n)
    lines_elapsed = time() - t0_lines

    line_count, packed_points, max_line_len, triples_equiv = line_stats(line_offsets_cpu)
    verbose && @printf("  Lines: %d | Packed points: %d | Max len: %d | Triple-equiv: %d | %.3fs\n",
                       line_count, packed_points, max_line_len, triples_equiv, lines_elapsed)

    point_col_scale_cpu, mean_inc, min_inc, max_inc, col_norm_mode_l = compute_point_collinearity_scale(
        n, line_offsets_cpu, line_points_cpu; mode=col_norm_mode
    )

    if verbose
        if col_norm_mode_l == "mean-incidence"
            @printf("  Collinearity normalization: %s | incidence mean=%.3f, min=%.3f, max=%.3f\n",
                    col_norm_mode_l, mean_inc, min_inc, max_inc)
        else
            @printf("  Collinearity normalization: %s\n", col_norm_mode_l)
        end
    end

    alpha_override = args["alpha"]
    gamma_override = args["gamma"]

    α, γ, base_α, base_γ, mode_l, avg_triples_per_var, col_scale = choose_coefficients(
        n,
        line_offsets_cpu;
        mode=coeff_mode,
        normalization_mode=col_norm_mode_l,
        alpha_override=alpha_override,
        gamma_override=gamma_override,
    )

    if verbose
        println("Coefficient selection:")
        @printf("  mode=%s | base α=%.3f, γ=%.3f\n", mode_l, base_α, base_γ)
        if mode_l == "density"
            if isnan(avg_triples_per_var)
                println("  density stats: small-n fallback to legacy scaling")
            else
                @printf("  density stats: avg triples/var=%.3f, col-scale=%.4f\n",
                        avg_triples_per_var, col_scale)
            end
        elseif mode_l == "normalized"
            @printf("  normalized mode active with col-normalization=%s\n", col_norm_mode_l)
        end
        if !isnothing(alpha_override) || !isnothing(gamma_override)
            @printf("  overrides applied -> α=%.3f, γ=%.3f\n", α, γ)
        end
    end

    cfg = Config(n=n, R=R, T=T, dt=dt, α=α, γ=γ)

    success, grid, elapsed, stats, used_seed = solve_metal(
        cfg,
        line_offsets_cpu,
        line_points_cpu,
        point_col_scale_cpu,
        seed,
        outdir;
        verbose=verbose,
        batch_size=batch_size,
        max_batches=max_batches,
        col_normalization=col_norm_mode_l,
    )

    if success
        println()
        println("Reproduce exact solution:")
        println("julia --project=. $(PROGRAM_FILE) $(n) --R $(R) --T $(T) --dt $(dt) --coeff-mode $(mode_l) --col-normalization $(col_norm_mode_l) --alpha $(cfg.α) --gamma $(cfg.γ) --seed $(used_seed) --batch-size $(batch_size) --max-batches $(max_batches) --outdir $(outdir)")
    end

    return success ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
