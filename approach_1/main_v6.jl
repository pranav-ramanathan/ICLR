#!/usr/bin/env julia
#=
N3L Pure Gradient Flow — CPU Tsit5, Line-Based Representation (v6)
===================================================================
Combines v5's adaptive Tsit5 ODE integration and lock-free parallel
search with the line-based representation, collinearity normalization,
and coefficient strategies from main_metal_v3.jl / main_rocm.jl.

Key changes vs main_v5.jl:
  - Triple precompute → line CSR precompute (exact triple-equivalent)
  - Per-point collinearity normalization (mean-incidence, default)
  - Coefficient modes: normalized (default), legacy, density
  - Line-based violation counting
  - Float64 throughout (CPU precision advantage over GPU Float32)
=#

using OrdinaryDiffEq
using DiffEqCallbacks
using Random
using Printf
using Dates
using ArgParse
include("logging_utils.jl")
using Base.Threads: Atomic, atomic_add!, atomic_cas!

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
# Top-k Mask Helper
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
# Line Representation (from metal_v3 / rocm)
# ============================================================================

@inline function normalize_direction(dx::Int, dy::Int)
    g = gcd(abs(dx), abs(dy))
    dx = div(dx, g)
    dy = div(dy, g)
    if dy < 0 || (dy == 0 && dx < 0)
        dx = -dx
        dy = -dy
    end
    return dx, dy
end

function compute_lines(n::Int)
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

    line_offsets = Int32[1]
    line_points  = Int32[]

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

# ============================================================================
# Point Collinearity Scale (normalization)
# ============================================================================

function compute_point_collinearity_scale(
    n::Int,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32};
    mode::String="mean-incidence",
)
    mode_l = lowercase(mode)
    N = n * n

    if mode_l == "none"
        return ones(Float64, N), NaN, NaN, NaN, mode_l
    elseif mode_l != "mean-incidence"
        error("Invalid --col-normalization '$mode'. Expected one of: mean-incidence, none")
    end

    L = length(line_offsets) - 1
    incidence = zeros(Float64, N)

    @inbounds for l in 1:L
        start_idx = Int(line_offsets[l])
        stop_idx  = Int(line_offsets[l + 1] - 1)
        len = stop_idx - start_idx + 1
        len < 3 && continue

        local_inc = (len - 1) * (len - 2) / 2
        for idx in start_idx:stop_idx
            p = Int(line_points[idx])
            incidence[p] += local_inc
        end
    end

    mean_inc = sum(incidence) / max(1, N)
    min_inc  = minimum(incidence)
    max_inc  = maximum(incidence)

    if mean_inc <= 0
        return ones(Float64, N), mean_inc, min_inc, max_inc, mode_l
    end

    scale = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        denom = max(1.0, incidence[i])
        scale[i] = mean_inc / denom
    end

    return scale, mean_inc, min_inc, max_inc, mode_l
end

# ============================================================================
# Coefficient Selection (normalized / legacy / density)
# ============================================================================

@inline function legacy_coefficients(n::Int)
    α = n <= 10 ? 10.0 * (n / 6) : 40.0
    γ = n <= 10 ? 5.0 : 15.0
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
        triple_incidence_sum += len * ((len - 1) * (len - 2) / 2)
    end

    avg_triples_per_var = triple_incidence_sum / max(1, N)
    target_density = 2.0 / n
    col_scale = avg_triples_per_var * target_density * target_density

    α = clamp(105.0 / max(col_scale, 1e-6), 10.0, 28.0)
    γ = n <= 16 ? 4.5 : 4.0

    return α, γ, avg_triples_per_var, col_scale
end

@inline function normalized_coefficients(n::Int, normalization_mode::String)
    if n <= 10
        return legacy_coefficients(n)
    end

    norm_mode_l = lowercase(normalization_mode)

    α = if norm_mode_l == "mean-incidence"
        10.0
    else
        n <= 16 ? 10.0 : 8.0
    end

    γ = n <= 16 ? 4.5 : 4.0
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

    base_α = 0.0
    base_γ = 0.0
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

    α = isnothing(alpha_override) ? base_α : Float64(alpha_override)
    γ = isnothing(gamma_override) ? base_γ : Float64(gamma_override)

    return α, γ, base_α, base_γ, mode_l, avg_triples_per_var, col_scale
end

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef struct Config
    n::Int
    R::Int = 200
    T::Float64 = 15.0
    check_interval::Float64 = 0.1
    α::Float64 = 10.0
    β::Float64 = 1.0
    γ::Float64 = 4.5
end

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_cli_args(args)
    s = ArgParseSettings(
        description = "N3L Pure Gradient Flow — CPU Tsit5, Line-Based (v6)",
        version = "6.0.0",
        add_version = true
    )

    @add_arg_table! s begin
        "n"
            help = "Board size (n x n grid)"
            arg_type = Int
            required = true
        "--R"
            help = "Maximum number of trajectories to try"
            arg_type = Int
            default = 200
        "--T"
            help = "Max integration time per trajectory"
            arg_type = Float64
            default = 15.0
        "--alpha"
            help = "Override alpha penalty coefficient"
            arg_type = Float64
        "--gamma"
            help = "Override gamma binary regularization"
            arg_type = Float64
        "--coeff-mode"
            help = "Coefficient strategy: normalized | legacy | density"
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
        "--progress-interval"
            help = "Print progress every N trajectories"
            arg_type = Int
            default = 50
    end

    return parse_args(args, s)
end

# ============================================================================
# Validation (line-based)
# ============================================================================

function count_violations_lines(x_bin::BitVector, line_offsets::Vector{Int32}, line_points::Vector{Int32})
    count = 0
    L = length(line_offsets) - 1

    @inbounds for l in 1:L
        c = 0
        start_idx = line_offsets[l]
        stop_idx  = line_offsets[l + 1] - 1
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
# Initial Condition
# ============================================================================

function biased_init(rng, N, target_density)
    a = max(0.5, 2.0 * target_density)
    b = max(0.5, 2.0 * (1.0 - target_density))

    x0 = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        u = rand(rng)^(1/a)
        v = rand(rng)^(1/b)
        x0[i] = u / (u + v)
    end
    return x0
end

# ============================================================================
# ODE System — LINE-BASED GRADIENT FLOW
# ============================================================================
# The collinearity energy is: E_col = α * Σ_lines Σ_{i<j<k on line} x_i x_j x_k
# The gradient ∂E_col/∂x_p for point p on a line with point values {x_j} is:
#   α * Σ_{j<k, j≠p, k≠p} x_j x_k = α * ((S1 - xp)^2 - (S2 - xp^2)) / 2
# where S1 = Σ x_j over line, S2 = Σ x_j^2 over line.
# This is accumulated over all lines containing point p.
# The point_col_scale[p] factor normalises by collinearity incidence.

function make_rhs(
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float64},
    cfg::Config,
)
    N = cfg.n^2
    L = length(line_offsets) - 1
    g = zeros(N)

    function rhs!(dx, x, p, t)
        # Base gradient: count term + binary regularization
        @inbounds for i in 1:N
            xi = x[i]
            g[i] = -cfg.β + cfg.γ * xi * (2.0 - 6.0*xi + 4.0*xi*xi)
        end

        # Line-based collinearity gradient (accumulate pair_sum per point per line)
        @inbounds for l in 1:L
            start_idx = Int(line_offsets[l])
            stop_idx  = Int(line_offsets[l + 1] - 1)

            # Compute line sums
            s1 = 0.0
            s2 = 0.0
            for idx in start_idx:stop_idx
                xp = x[line_points[idx]]
                s1 += xp
                s2 += xp * xp
            end

            # Gradient contribution for each point on this line
            for idx in start_idx:stop_idx
                pi = Int(line_points[idx])
                xi = x[pi]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5 * (s1o * s1o - s2o)
                g[pi] += cfg.α * point_col_scale[pi] * pair_sum
            end
        end

        # Negative gradient flow with box constraints
        @inbounds for i in 1:N
            dx[i] = -g[i]
            if x[i] <= 0.0 && dx[i] < 0.0
                dx[i] = 0.0
            elseif x[i] >= 1.0 && dx[i] > 0.0
                dx[i] = 0.0
            end
        end
    end

    return rhs!
end

# ============================================================================
# Single Trajectory
# ============================================================================

@enum Status RUNNING SUCCESS TIMEOUT

function run_trajectory(cfg::Config, line_offsets, line_points, point_col_scale, rng)
    N = cfg.n^2
    target = 2 * cfg.n

    x0 = biased_init(rng, N, target / N)
    status = Ref(RUNNING)

    rhs! = make_rhs(line_offsets, line_points, point_col_scale, cfg)

    function check!(integrator)
        x = integrator.u
        x_bin = topk_mask(x, target)
        viols = count_violations_lines(x_bin, line_offsets, line_points)

        if viols == 0
            status[] = SUCCESS
            terminate!(integrator)
        end
    end

    cb = PeriodicCallback(check!, cfg.check_interval; save_positions=(false, false))

    prob = ODEProblem(rhs!, x0, (0.0, cfg.T))
    sol = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-4, callback=cb,
                save_everystep=false, save_start=false, maxiters=1_000_000)

    x_final = sol.u[end]
    x_bin = topk_mask(x_final, target)
    viols = count_violations_lines(x_bin, line_offsets, line_points)

    if status[] == RUNNING
        status[] = (viols == 0) ? SUCCESS : TIMEOUT
    end

    return status[], x_bin, viols
end

# ============================================================================
# Parallel Search — LOCK-FREE OPTIMIZED
# ============================================================================

function solve_n3l(n::Int, R::Int, T::Float64, seed::UInt64, outdir::String;
                   α::Float64=10.0,
                   β::Float64=1.0,
                   γ::Float64=4.5,
                   coeff_mode::String="normalized",
                   col_normalization::String="mean-incidence",
                   verbose::Bool=true,
                   progress_interval::Int=50)

    cfg = Config(n=n, R=R, T=T, α=α, β=β, γ=γ)
    target = 2n

    # Precompute lines
    verbose && println("Precomputing lines...")
    t0_lines = time()
    line_offsets, line_points = compute_lines(n)
    lines_elapsed = time() - t0_lines

    line_count, packed_points, max_line_len, triples_equiv = line_stats(line_offsets)
    verbose && @printf("  Lines: %d | Packed points: %d | Max len: %d | Triple-equiv: %d | %.3fs\n",
                       line_count, packed_points, max_line_len, triples_equiv, lines_elapsed)

    # Collinearity normalization
    point_col_scale, mean_inc, min_inc, max_inc, col_norm_mode_l = compute_point_collinearity_scale(
        n, line_offsets, line_points; mode=col_normalization
    )

    if verbose
        if col_norm_mode_l == "mean-incidence"
            @printf("  Collinearity normalization: %s | incidence mean=%.3f, min=%.3f, max=%.3f\n",
                    col_norm_mode_l, mean_inc, min_inc, max_inc)
        else
            @printf("  Collinearity normalization: %s\n", col_norm_mode_l)
        end
    end

    verbose && println("="^60)
    verbose && println("N3L Gradient Flow — CPU Tsit5, Line-Based (v6)")
    verbose && println("="^60)
    verbose && @printf("n=%d, target=%d | α=%.3f, β=%.1f, γ=%.3f\n", n, target, cfg.α, cfg.β, cfg.γ)
    verbose && @printf("Max: %d traj, T=%.1fs, %d threads\n", R, T, Threads.nthreads())
    verbose && @printf("Seed: %d | Coeff mode: %s | Col norm: %s\n", seed, coeff_mode, col_norm_mode_l)
    verbose && @printf("Lines: %d (triple-equiv: %d) | Progress: every %d\n",
                       line_count, triples_equiv, progress_interval)
    verbose && println("="^60)

    # Atomics for lock-free coordination
    solution_found = Atomic{Bool}(false)
    trajectories_tried = Atomic{Int}(0)
    best_viols = Atomic{Int}(typemax(Int))
    last_progress = Atomic{Int}(0)

    solution_lock = ReentrantLock()
    solution_grid = nothing
    solution_traj_id = 0

    thread_histograms = [Dict{Int,Int}() for _ in 1:Threads.nthreads()]

    start_time = time()

    Threads.@threads for id in 1:R
        if solution_found[]
            break
        end

        traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(id) << 1))
        rng = Xoshiro(traj_seed)
        status, x_bin, viols = run_trajectory(cfg, line_offsets, line_points, point_col_scale, rng)

        tried = atomic_add!(trajectories_tried, 1)

        tid = Threads.threadid()
        thread_histograms[tid][viols] = get(thread_histograms[tid], viols, 0) + 1

        if viols == 0
            if !solution_found[]
                lock(solution_lock) do
                    if !solution_found[]
                        solution_found[] = true
                        solution_grid = reshape(x_bin, (n, n))
                        solution_traj_id = id
                        elapsed = time() - start_time
                        verbose && @printf("\n🎉 SOLUTION! traj=%d, time=%.1fs, tried=%d\n",
                                          id, elapsed, tried)
                    end
                end
            end
        elseif viols < best_viols[]
            old_best = best_viols[]
            if viols < old_best && atomic_cas!(best_viols, old_best, viols) == old_best
                verbose && @printf("[%6d] ★ NEW BEST: %d viols (traj %d)\n", tried, viols, id)
            end
        end

        if tried - last_progress[] >= progress_interval
            old = last_progress[]
            if atomic_cas!(last_progress, old, tried) == old
                elapsed = time() - start_time
                rate = tried / elapsed
                eta = (R - tried) / rate
                verbose && @printf("[%6d] best=%d | %.1f/s | eta=%s\n",
                                  tried, best_viols[], rate, format_time(eta))
            end
        end

        if solution_found[]
            break
        end
    end

    elapsed = time() - start_time

    violation_histogram = Dict{Int,Int}()
    for hist in thread_histograms
        for (v, count) in hist
            violation_histogram[v] = get(violation_histogram, v, 0) + count
        end
    end

    verbose && println("-"^60)

    if solution_found[]
        verbose && println("✓✓✓ SUCCESS ✓✓✓")
        verbose && @printf("Time: %.2fs | Tried: %d/%d (%.1f%%) | Rate: %.1f/s\n",
                          elapsed, trajectories_tried[], R,
                          100*trajectories_tried[]/R, trajectories_tried[]/elapsed)

        if !isempty(violation_histogram)
            verbose && println("\nViolation distribution:")
            for v in sort(collect(keys(violation_histogram)))
                count = violation_histogram[v]
                pct = 100 * count / trajectories_tried[]
                bar = repeat("█", min(40, round(Int, pct/2)))
                verbose && @printf("  %2d violations: %5d (%5.2f%%) %s\n", v, count, pct, bar)
            end
        end

        verbose && println("\nSolution:")
        print_grid(solution_grid)
        save_solution(n, solution_grid, solution_traj_id, R, T, seed, cfg.α, cfg.γ, outdir;
                      coeff_mode=coeff_mode, col_normalization=col_norm_mode_l)

        return true, solution_grid, elapsed, Dict(:success=>1, :tried=>trajectories_tried[]), seed
    else
        verbose && println("✗✗✗ NO SOLUTION FOUND ✗✗✗")
        verbose && @printf("Time: %.2fs | Tried: %d | Best: %d viols | Rate: %.1f/s\n",
                          elapsed, trajectories_tried[], best_viols[], trajectories_tried[]/elapsed)

        if !isempty(violation_histogram)
            verbose && println("\nViolation distribution:")
            for v in sort(collect(keys(violation_histogram)))
                count = violation_histogram[v]
                pct = 100 * count / trajectories_tried[]
                bar = repeat("█", min(40, round(Int, pct/2)))
                verbose && @printf("  %2d violations: %5d (%5.2f%%) %s\n", v, count, pct, bar)
            end
        end

        return false, nothing, elapsed, Dict(:success=>0, :tried=>trajectories_tried[]), seed
    end
end

function format_time(seconds::Float64)
    if seconds < 60
        return @sprintf("%.0fs", seconds)
    elseif seconds < 3600
        mins = floor(Int, seconds / 60)
        secs = round(Int, seconds % 60)
        return @sprintf("%dm %ds", mins, secs)
    else
        hours = floor(Int, seconds / 3600)
        mins  = floor(Int, (seconds % 3600) / 60)
        return @sprintf("%dh %dm", hours, mins)
    end
end

function print_grid(grid)
    n = size(grid, 1)
    for i in 1:n
        print("  ")
        for j in 1:n
            print(grid[i,j] ? "● " : "· ")
        end
        println()
    end
end

# ============================================================================
# Save Solution (compatible format)
# ============================================================================

function save_solution(n, grid, traj_id, R, T, seed, α, γ, outdir;
                       coeff_mode::String="normalized",
                       col_normalization::String="mean-incidence")
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "$(outdir)/$(n)"
    mkpath(dir)
    filename = "$(dir)/sol_v6_$(timestamp)_traj$(traj_id).txt"

    open(filename, "w") do io
        println(io, "# n=$(n)")
        println(io, "# target=$(2n)")
        println(io, "# trajectory_id=$(traj_id)")
        println(io, "# R=$(R)")
        println(io, "# T=$(T)")
        println(io, "# seed=$(seed)")
        println(io, "# α=$(α), β=1.0, γ=$(γ)")
        println(io, "# method=CPU Tsit5 Line-Based Gradient Flow (v6)")
        println(io, "# coeff_mode=$(coeff_mode)")
        println(io, "# col_normalization=$(col_normalization)")
        println(io, "# timestamp=$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SSZ"))")
        println(io, "#")
        println(io, "# Grid (0/1):")
        for i in 1:n
            println(io, join(Int.(grid[i, :]), " "))
        end
        println(io, "#")
        println(io, "# Coordinates (row, col):")
        for i in 1:n, j in 1:n
            grid[i,j] && println(io, "($i, $j)")
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
    T = args["T"]
    outdir = args["outdir"]
    quiet = args["quiet"]
    verbose = !quiet
    progress_interval = args["progress-interval"]
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
    if !(lowercase(coeff_mode) in ("normalized", "legacy", "density"))
        println("ERROR: coeff-mode must be one of: normalized, legacy, density")
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

    # Precompute lines early to feed into coefficient chooser
    verbose && println("Precomputing lines for coefficient selection...")
    line_offsets_tmp, _ = compute_lines(n)

    alpha_override = args["alpha"]
    gamma_override = args["gamma"]

    α, γ, base_α, base_γ, mode_l, avg_triples_per_var, col_scale = choose_coefficients(
        n,
        line_offsets_tmp;
        mode=coeff_mode,
        normalization_mode=col_norm_mode,
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
            @printf("  normalized mode active with col-normalization=%s\n", col_norm_mode)
        end
        if !isnothing(alpha_override) || !isnothing(gamma_override)
            @printf("  overrides applied -> α=%.3f, γ=%.3f\n", α, γ)
        end
    end

    success, grid, elapsed, stats, used_seed = solve_n3l(n, R, T, seed, outdir;
                                                          α=α, β=1.0, γ=γ,
                                                          coeff_mode=mode_l,
                                                          col_normalization=col_norm_mode,
                                                          verbose=verbose,
                                                          progress_interval=progress_interval)

    if success
        println()
        alpha_str = isnothing(alpha_override) ? "" : " --alpha $(alpha_override)"
        gamma_str = isnothing(gamma_override) ? "" : " --gamma $(gamma_override)"
        println("Reproduce exact solution:")
        println("julia --project=. --threads=auto $(PROGRAM_FILE) $(n) --R $(R) --T $(T) --coeff-mode $(mode_l) --col-normalization $(col_norm_mode)$(alpha_str)$(gamma_str) --seed $(used_seed)")
    end

    return success ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_with_terminal_log("cpu_v6", ARGS) do
        main()
    end)
end
