#!/usr/bin/env julia
#=
N3L Hybrid CPU-GPU Solver — ROCm V2
===================================================================
This version implements hybrid CPU-GPU co-execution with local repair:
  - GPU pipeline (copy VERBATIM from main_rocm.jl)
  - CPU co-executor with parameter diversification
  - Validator subsystem
  - Local search repair for near-solutions

Based on Phase 0 findings: α=6.0, γ=4.0 optimal for N≥17 on ROCm
=#

using AMDGPU
using KernelAbstractions
using Random
using Printf
using Dates
using ArgParse
using Base.Threads: Atomic, atomic_add!, atomic_cas!

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

# Channel payload types for concurrent subsystem communication.
const ValidationResult = Tuple{Matrix{Float32}, UInt64, Int}  # (result_batch, seed, batch_idx)
const RepairCandidate = Tuple{BitVector, Int, String}         # (x_bin, traj_id, source)

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
        description = "N3L Hybrid CPU-GPU Solver — ROCm V2 (with local repair)",
        version = "2.0.0",
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
        "--cpu-threads"
            help = "Number of CPU threads for co-execution (default: 4)"
            arg_type = Int
            default = 4
        "--repair-threshold"
            help = "Max violations to attempt repair (default: 5)"
            arg_type = Int
            default = 5
        "--cpu-alpha-min"
            help = "Min alpha for CPU parameter diversification"
            arg_type = Float64
        "--cpu-alpha-max"
            help = "Max alpha for CPU parameter diversification"
            arg_type = Float64
        "--cpu-gamma-min"
            help = "Min gamma for CPU parameter diversification"
            arg_type = Float64
        "--cpu-gamma-max"
            help = "Max gamma for CPU parameter diversification"
            arg_type = Float64
        "--no-cpu"
            help = "Disable CPU co-execution (GPU-only mode)"
            action = :store_true
        "--no-repair"
            help = "Disable local search repair"
            action = :store_true
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
# CPU RK4 Integrator (Single Trajectory, Line-Based)
# ============================================================================

function cpu_rk4_integrate!(
    state::Vector{Float64},
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::AbstractVector{<:Real},
    α::Float64,
    β::Float64,
    γ::Float64,
    dt::Float64,
    nsteps::Int,
    N::Int,
    L::Int,
)
    k1 = zeros(Float64, N)
    k2 = zeros(Float64, N)
    k3 = zeros(Float64, N)
    k4 = zeros(Float64, N)
    xtmp = zeros(Float64, N)

    half_dt = dt * 0.5
    sixth_dt = dt / 6.0

    @inbounds for step in 1:nsteps
        # ---- k1 = f(x) ----
        for i in 1:N
            xi = state[i]
            k1[i] = β - γ * xi * (2.0 - 6.0 * xi + 4.0 * xi * xi)
        end

        for l in 1:L
            start_idx = Int(line_offsets[l])
            stop_idx = Int(line_offsets[l + 1] - 1)

            s1 = 0.0
            s2 = 0.0
            for idx in start_idx:stop_idx
                p = Int(line_points[idx])
                x = state[p]
                s1 += x
                s2 += x * x
            end

            for idx in start_idx:stop_idx
                p = Int(line_points[idx])
                xi = state[p]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5 * (s1o * s1o - s2o)
                scale = Float64(point_col_scale[p])
                k1[p] -= α * scale * pair_sum
            end
        end

        # ---- k2 = f(x + dt/2 * k1) ----
        for i in 1:N
            xi_new = state[i] + half_dt * k1[i]
            xtmp[i] = min(1.0, max(0.0, xi_new))
        end

        for i in 1:N
            xi = xtmp[i]
            k2[i] = β - γ * xi * (2.0 - 6.0 * xi + 4.0 * xi * xi)
        end

        for l in 1:L
            start_idx = Int(line_offsets[l])
            stop_idx = Int(line_offsets[l + 1] - 1)

            s1 = 0.0
            s2 = 0.0
            for idx in start_idx:stop_idx
                p = Int(line_points[idx])
                x = xtmp[p]
                s1 += x
                s2 += x * x
            end

            for idx in start_idx:stop_idx
                p = Int(line_points[idx])
                xi = xtmp[p]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5 * (s1o * s1o - s2o)
                scale = Float64(point_col_scale[p])
                k2[p] -= α * scale * pair_sum
            end
        end

        # ---- k3 = f(x + dt/2 * k2) ----
        for i in 1:N
            xi_new = state[i] + half_dt * k2[i]
            xtmp[i] = min(1.0, max(0.0, xi_new))
        end

        for i in 1:N
            xi = xtmp[i]
            k3[i] = β - γ * xi * (2.0 - 6.0 * xi + 4.0 * xi * xi)
        end

        for l in 1:L
            start_idx = Int(line_offsets[l])
            stop_idx = Int(line_offsets[l + 1] - 1)

            s1 = 0.0
            s2 = 0.0
            for idx in start_idx:stop_idx
                p = Int(line_points[idx])
                x = xtmp[p]
                s1 += x
                s2 += x * x
            end

            for idx in start_idx:stop_idx
                p = Int(line_points[idx])
                xi = xtmp[p]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5 * (s1o * s1o - s2o)
                scale = Float64(point_col_scale[p])
                k3[p] -= α * scale * pair_sum
            end
        end

        # ---- k4 = f(x + dt * k3) ----
        for i in 1:N
            xi_new = state[i] + dt * k3[i]
            xtmp[i] = min(1.0, max(0.0, xi_new))
        end

        for i in 1:N
            xi = xtmp[i]
            k4[i] = β - γ * xi * (2.0 - 6.0 * xi + 4.0 * xi * xi)
        end

        for l in 1:L
            start_idx = Int(line_offsets[l])
            stop_idx = Int(line_offsets[l + 1] - 1)

            s1 = 0.0
            s2 = 0.0
            for idx in start_idx:stop_idx
                p = Int(line_points[idx])
                x = xtmp[p]
                s1 += x
                s2 += x * x
            end

            for idx in start_idx:stop_idx
                p = Int(line_points[idx])
                xi = xtmp[p]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5 * (s1o * s1o - s2o)
                scale = Float64(point_col_scale[p])
                k4[p] -= α * scale * pair_sum
            end
        end

        # ---- RK4 update + box constraint [0, 1] ----
        for i in 1:N
            xnew = state[i] + sixth_dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            state[i] = min(1.0, max(0.0, xnew))
        end
    end

    return state
end

function cpu_generate_initial_condition(n::Int, seed::UInt64)::Vector{Float64}
    N = n * n
    target_density = Float64(2 * n) / Float64(N)
    a = max(0.5, 2.0 * target_density)
    b = max(0.5, 2.0 * (1.0 - target_density))

    traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(1) << 1))
    rng = Xoshiro(traj_seed)

    state = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        u = Float64(rand(rng))^(1.0 / a)
        v = Float64(rand(rng))^(1.0 / b)
        state[i] = u / (u + v)
    end

    return state
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

function find_violated_lines(
    x_bin::BitVector,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
)::Vector{Int}
    violated = Int[]
    L = length(line_offsets) - 1

    @inbounds for l in 1:L
        c = 0
        start_idx = line_offsets[l]
        stop_idx = line_offsets[l + 1] - 1
        for idx in start_idx:stop_idx
            c += x_bin[line_points[idx]]
            if c >= 3
                push!(violated, l)
                break
            end
        end
    end

    return violated
end

function points_on_violated_lines(
    violated_lines::Vector{Int},
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
)::Set{Int}
    points = Set{Int}()

    @inbounds for l in violated_lines
        start_idx = line_offsets[l]
        stop_idx = line_offsets[l + 1] - 1
        for idx in start_idx:stop_idx
            push!(points, Int(line_points[idx]))
        end
    end

    return points
end

function identify_near_solutions(
    results::Matrix,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    n::Int;
    threshold::Int=5,
)::Vector{Tuple{Int, Int}}
    target = 2 * n
    near = Tuple{Int, Int}[]
    R = size(results, 2)

    @inbounds for traj in 1:R
        x_bin = topk_mask(@view(results[:, traj]), target)
        viols = count_violations_lines(x_bin, line_offsets, line_points)
        if viols <= threshold
            push!(near, (traj, viols))
        end
    end

    return near
end

function repair_near_solution!(
    x_bin::BitVector,
    n::Int,
    target::Int,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32};
    max_attempts::Int=1000,
)::Tuple{BitVector, Int}
    length(x_bin) != n * n && return x_bin, count_violations_lines(x_bin, line_offsets, line_points)
    count(x_bin) != target && return x_bin, count_violations_lines(x_bin, line_offsets, line_points)

    current_viols = count_violations_lines(x_bin, line_offsets, line_points)

    if current_viols == 0 || current_viols > 5
        return x_bin, current_viols
    end

    if current_viols <= 2
        while current_viols > 0
            violated_lines = find_violated_lines(x_bin, line_offsets, line_points)
            isempty(violated_lines) && break

            focus_points = points_on_violated_lines(violated_lines, line_offsets, line_points)
            isempty(focus_points) && break

            improved = false
            for remove_idx in sort!(collect(focus_points))
                x_bin[remove_idx] || continue
                for add_idx in eachindex(x_bin)
                    x_bin[add_idx] && continue

                    x_bin[remove_idx] = false
                    x_bin[add_idx] = true
                    new_viols = count_violations_lines(x_bin, line_offsets, line_points)

                    if new_viols < current_viols
                        current_viols = new_viols
                        improved = true
                        break
                    end

                    x_bin[remove_idx] = true
                    x_bin[add_idx] = false
                end
                improved && break
            end

            improved || break
        end
    else
        for _ in 1:max_attempts
            violated_lines = find_violated_lines(x_bin, line_offsets, line_points)
            isempty(violated_lines) && break

            focus_points = points_on_violated_lines(violated_lines, line_offsets, line_points)
            isempty(focus_points) && break

            selected_focus = Int[]
            unselected = Int[]

            for p in focus_points
                if x_bin[p]
                    push!(selected_focus, p)
                end
            end

            @inbounds for p in eachindex(x_bin)
                x_bin[p] || push!(unselected, p)
            end

            isempty(selected_focus) && break
            isempty(unselected) && break

            remove_idx = rand(selected_focus)
            add_idx = rand(unselected)

            x_bin[remove_idx] = false
            x_bin[add_idx] = true
            new_viols = count_violations_lines(x_bin, line_offsets, line_points)

            if new_viols < current_viols
                current_viols = new_viols
                current_viols == 0 && break
            else
                x_bin[remove_idx] = true
                x_bin[add_idx] = false
            end
        end
    end

    return x_bin, current_viols
end

# ============================================================================
# Output Functions
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
    filename = "$(dir)/sol_rocm_v2_$(timestamp)_traj$(traj_id).txt"

    open(filename, "w") do io
        println(io, "# n=$(n)")
        println(io, "# target=$(2n)")
        println(io, "# trajectory_id=$(traj_id)")
        println(io, "# R=$(R)")
        println(io, "# T=$(T)")
        println(io, "# seed=$(seed)")
        println(io, "# α=$(α), β=1.0, γ=$(γ)")
        println(io, "# method=ROCm AMDGPU Hybrid CPU-GPU with Local Repair (v2)")
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
# Concurrent Subsystems (Task 4)
# ============================================================================

function gpu_pipeline!(
    validation_ch::Channel{ValidationResult},
    solution_found::Atomic{Bool},
    best_viols::Atomic{Int},
    total_tried::Atomic{Int},
    n::Int,
    R::Int,
    batch_size::Int,
    max_batches::Int,
    seed::UInt64,
    α::Float64,
    β::Float64,
    γ::Float64,
    T::Float64,
    dt::Float64,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float32},
    verbose::Bool,
)
    _ = best_viols

    N = Int32(n * n)
    L = Int32(length(line_offsets) - 1)
    backend = AMDGPU.ROCBackend()
    d_state = ROCArray{Float32}(undef, Int(N), batch_size)
    d_line_offsets = ROCArray(line_offsets)
    d_line_points = ROCArray(line_points)
    d_point_col_scale = ROCArray(point_col_scale)
    kernel! = rk4_gradient_flow_lines_kernel!(backend)

    nsteps = round(Int32, T / dt)
    num_batches = cld(R, batch_size)
    if max_batches > 0
        num_batches = min(num_batches, max_batches)
    end

    for batch in 1:num_batches
        solution_found[] && break

        this_batch = min(batch_size, R - (batch - 1) * batch_size)
        this_batch <= 0 && break

        batch_seed = splitmix64(seed ⊻ UInt64(batch))
        ic = generate_initial_conditions(n, this_batch, batch_seed)
        copyto!(view(d_state, :, 1:this_batch), ic)

        kernel!(
            d_state,
            d_line_offsets,
            d_line_points,
            d_point_col_scale,
            Float32(α),
            Float32(β),
            Float32(γ),
            Float32(dt),
            nsteps,
            N,
            L;
            ndrange=this_batch,
        )
        KernelAbstractions.synchronize(backend)

        result_cpu = Matrix{Float32}(undef, Int(N), this_batch)
        copyto!(result_cpu, view(d_state, :, 1:this_batch))

        put!(validation_ch, (result_cpu, batch_seed, batch))

        atomic_add!(total_tried, this_batch)

        if verbose
            println("GPU batch $batch/$num_batches complete ($this_batch trajectories)")
        end
    end

    verbose && println("GPU pipeline complete")
end

function cpu_coexecutor!(
    repair_ch::Channel{RepairCandidate},
    solution_found::Atomic{Bool},
    best_viols::Atomic{Int},
    total_tried::Atomic{Int},
    n::Int,
    cpu_threads::Int,
    base_seed::UInt64,
    α_min::Float64,
    α_max::Float64,
    γ_min::Float64,
    γ_max::Float64,
    β::Float64,
    T::Float64,
    dt::Float64,
    repair_threshold::Int,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float32},
    verbose::Bool,
    save_fn::Function,
)
    tasks = Task[]

    for worker_id in 1:cpu_threads
        task = Threads.@spawn begin
            N = n * n
            L = length(line_offsets) - 1
            nsteps = round(Int, T / dt)

            rng = Xoshiro(splitmix64(base_seed ⊻ UInt64(worker_id)))
            α_worker = α_min + (α_max - α_min) * rand(rng)
            γ_worker = γ_min + (γ_max - γ_min) * rand(rng)

            traj_count = 0
            while !solution_found[]
                traj_count += 1
                traj_seed = splitmix64(base_seed ⊻ UInt64(worker_id) ⊻ UInt64(traj_count))

                state = cpu_generate_initial_condition(n, traj_seed)
                cpu_rk4_integrate!(
                    state,
                    line_offsets,
                    line_points,
                    point_col_scale,
                    α_worker,
                    β,
                    γ_worker,
                    dt,
                    nsteps,
                    N,
                    L,
                )

                x_bin = topk_mask(state, 2 * n)
                viols = count_violations_lines(x_bin, line_offsets, line_points)

                atomic_add!(total_tried, 1)

                current_best = best_viols[]
                while viols < current_best
                    if atomic_cas!(best_viols, current_best, viols) == current_best
                        verbose && println("CPU worker $worker_id: new best = $viols")
                        break
                    end
                    current_best = best_viols[]
                end

                if viols == 0
                    solution_found[] = true
                    grid = reshape(x_bin, (n, n))
                    save_fn(
                        n,
                        grid,
                        traj_count,
                        0,
                        T,
                        UInt64(traj_seed),
                        α_worker,
                        γ_worker,
                        "solutions";
                        col_normalization="mean-incidence",
                    )
                    verbose && println("CPU worker $worker_id: SOLUTION FOUND (traj $traj_count)")
                    break
                end

                if viols > 0 && viols <= repair_threshold && !solution_found[]
                    try
                        put!(repair_ch, (copy(x_bin), traj_count, "CPU-$worker_id"))
                    catch
                        # Channel closed, exit
                        break
                    end
                end
            end
        end
        push!(tasks, task)
    end

    for task in tasks
        wait(task)
    end

    verbose && println("CPU co-executor complete ($cpu_threads workers)")
end

function validator!(
    validation_ch::Channel{ValidationResult},
    repair_ch::Channel{RepairCandidate},
    solution_found::Atomic{Bool},
    best_viols::Atomic{Int},
    total_tried::Atomic{Int},
    n::Int,
    repair_threshold::Int,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    verbose::Bool,
    save_fn::Function,
)
    _ = total_tried

    batch_count = 0
    for (result_batch, batch_seed, batch_idx) in validation_ch
        solution_found[] && break
        batch_count += 1

        this_batch = size(result_batch, 2)

        for traj in 1:this_batch
            solution_found[] && break

            x_continuous = @view result_batch[:, traj]
            x_bin = topk_mask(x_continuous, 2 * n)
            viols = count_violations_lines(x_bin, line_offsets, line_points)

            current_best = best_viols[]
            while viols < current_best
                if atomic_cas!(best_viols, current_best, viols) == current_best
                    verbose && println("GPU batch $batch_idx: new best = $viols")
                    break
                end
                current_best = best_viols[]
            end

            if viols == 0
                solution_found[] = true
                grid = reshape(x_bin, (n, n))
                traj_seed = splitmix64(batch_seed ⊻ UInt64(traj))
                save_fn(
                    n,
                    grid,
                    traj,
                    0,
                    0.0,
                    traj_seed,
                    0.0,
                    0.0,
                    "solutions";
                    col_normalization="mean-incidence",
                )
                verbose && println("GPU batch $batch_idx: SOLUTION FOUND (traj $traj)")
                break
            end

            if viols > 0 && viols <= repair_threshold && !solution_found[]
                try
                    put!(repair_ch, (copy(x_bin), traj, "GPU-batch$batch_idx"))
                catch
                    # Channel closed, continue validation
                end
            end
        end
    end

    verbose && println("Validator complete (processed $batch_count batches)")
end

function repair_worker!(
    repair_ch::Channel{RepairCandidate},
    solution_found::Atomic{Bool},
    best_viols::Atomic{Int},
    n::Int,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    max_attempts::Int,
    verbose::Bool,
    save_fn::Function,
)
    repair_count = 0
    success_count = 0

    for (x_bin, traj_id, source) in repair_ch
        solution_found[] && break
        repair_count += 1

        repaired, final_viols = repair_near_solution!(
            x_bin,
            n,
            2 * n,
            line_offsets,
            line_points;
            max_attempts=max_attempts,
        )

        current_best = best_viols[]
        while final_viols < current_best
            if atomic_cas!(best_viols, current_best, final_viols) == current_best
                verbose && println("Repair ($source): new best = $final_viols")
                break
            end
            current_best = best_viols[]
        end

        if final_viols == 0
            solution_found[] = true
            success_count += 1
            grid = reshape(repaired, (n, n))
            save_fn(
                n,
                grid,
                traj_id,
                0,
                0.0,
                UInt64(0),
                0.0,
                0.0,
                "solutions";
                col_normalization="mean-incidence",
            )
            verbose && println("Repair ($source): SOLUTION FOUND (repaired from traj $traj_id)")
            break
        end
    end

    verbose && println("Repair worker complete (processed $repair_count, repaired $success_count)")
end

# ============================================================================
# PLACEHOLDER FOR MAIN HYBRID SOLVER (Task 5)
# ============================================================================

function solve_hybrid(
    cfg::Config,
    line_offsets_cpu::Vector{Int32},
    line_points_cpu::Vector{Int32},
    point_col_scale_cpu::Vector{Float32},
    seed::UInt64,
    outdir::String;
    verbose::Bool=true,
    batch_size::Int=1024,
    max_batches::Int=0,
    cpu_threads::Int=4,
    repair_threshold::Int=5,
    cpu_alpha_min::Float64,
    cpu_alpha_max::Float64,
    cpu_gamma_min::Float64,
    cpu_gamma_max::Float64,
    no_repair::Bool=false,
    col_normalization::String="mean-incidence",
)
    n = cfg.n
    N = Int32(n * n)
    target = 2 * n
    L = Int32(length(line_offsets_cpu) - 1)
    nsteps = Int32(max(1, floor(Int, cfg.T / cfg.dt)))
    effective_T = Float32(nsteps) * cfg.dt

    if Int(N) > MAX_STATE_DIM
        println("ERROR: n^2=$(Int(N)) exceeds MAX_STATE_DIM=$MAX_STATE_DIM in kernel scratch memory.")
        println("       Increase MAX_STATE_DIM (with performance tradeoff) or reduce n.")
        return false, nothing, 0.0, Dict(:success => 0, :tried => 0, :gpu => 0, :cpu => 0, :repair => 0), seed
    end

    line_count, packed_points, max_line_len, triples_equiv = line_stats(line_offsets_cpu)

    verbose && println("="^60)
    verbose && println("N3L Gradient Flow — Hybrid CPU-GPU (ROCm V2 with Repair)")
    verbose && println("="^60)
    verbose && @printf("n=%d, target=%d | α=%.3f, β=%.1f, γ=%.3f\n", n, target, cfg.α, cfg.β, cfg.γ)
    verbose && @printf("Max: %d traj, T=%.3fs (effective %.3fs), dt=%.4f, steps=%d\n",
                       cfg.R, cfg.T, effective_T, cfg.dt, nsteps)
    verbose && @printf("Batch size: %d | CPU threads: %d | Seed: %d\n", batch_size, cpu_threads, seed)
    verbose && println("-"^60)
    verbose && @printf("Lines: %d | Packed points: %d | max line len: %d | triple-equivalent terms: %d\n",
                       line_count, packed_points, max_line_len, triples_equiv)
    verbose && @printf("Collinearity normalization: %s\n", col_normalization)
    verbose && @printf("CPU parameter ranges: α=[%.2f, %.2f], γ=[%.2f, %.2f]\n",
                       cpu_alpha_min, cpu_alpha_max, cpu_gamma_min, cpu_gamma_max)
    verbose && @printf("Repair threshold: %d | No-repair: %s\n", repair_threshold, no_repair)
    verbose && println("-"^60)

    # Channel setup
    validation_ch = Channel{ValidationResult}(4)
    repair_ch = Channel{RepairCandidate}(32)

    # Atomic state setup
    solution_found = Atomic{Bool}(false)
    best_viols = Atomic{Int}(typemax(Int))
    total_tried = Atomic{Int}(0)

    # GPU warmup
    backend = AMDGPU.ROCBackend()
    kern = rk4_gradient_flow_lines_kernel!(backend, 64)
    d_line_offsets = ROCArray(line_offsets_cpu)
    d_line_points = ROCArray(line_points_cpu)
    d_point_col_scale = ROCArray(point_col_scale_cpu)

    verbose && print("Warming up ROCm GPU (compiling kernel)...")
    try
        warmup_state = ROCArray(rand(Float32, N, 2))
        kern(warmup_state, d_line_offsets, d_line_points, d_point_col_scale,
             cfg.α, cfg.β, cfg.γ, cfg.dt, Int32(1), N, L;
             ndrange=2)
        KernelAbstractions.synchronize(backend)
        verbose && println("\n  ✓ ROCm warmup OK")
    catch e
        verbose && println("\n  ⚠ ROCm warmup failed: $(typeof(e)): $(sprint(showerror, e))")
        verbose && println("  This is a fatal error — kernel cannot compile.")
        return false, nothing, 0.0, Dict(:success => 0, :tried => 0, :gpu => 0, :cpu => 0, :repair => 0), seed
    end
    verbose && println("-"^60)

    # Counters for per-source statistics
    gpu_batches_completed = Atomic{Int}(0)
    cpu_trajectories_tried = Atomic{Int}(0)
    repair_attempts = Atomic{Int}(0)

    # Create save function closure
    save_fn = (n, grid, traj_id, R, T, seed, α, γ, outdir; col_normalization="mean-incidence") -> begin
        save_solution(n, grid, traj_id, R, T, seed, α, γ, outdir; col_normalization=col_normalization)
    end

    start_time = time()

    # Spawn all 4 subsystems
    gpu_task = Threads.@spawn gpu_pipeline!(
        validation_ch,
        solution_found,
        best_viols,
        total_tried,
        n,
        cfg.R,
        batch_size,
        max_batches,
        seed,
        Float64(cfg.α),
        Float64(cfg.β),
        Float64(cfg.γ),
        Float64(cfg.T),
        Float64(cfg.dt),
        line_offsets_cpu,
        line_points_cpu,
        point_col_scale_cpu,
        verbose,
    )

    validator_task = Threads.@spawn validator!(
        validation_ch,
        repair_ch,
        solution_found,
        best_viols,
        total_tried,
        n,
        repair_threshold,
        line_offsets_cpu,
        line_points_cpu,
        verbose,
        save_fn,
    )

    cpu_task = Threads.@spawn cpu_coexecutor!(
        repair_ch,
        solution_found,
        best_viols,
        total_tried,
        n,
        cpu_threads,
        seed,
        cpu_alpha_min,
        cpu_alpha_max,
        cpu_gamma_min,
        cpu_gamma_max,
        Float64(cfg.β),
        Float64(cfg.T),
        Float64(cfg.dt),
        repair_threshold,
        line_offsets_cpu,
        line_points_cpu,
        point_col_scale_cpu,
        verbose,
        save_fn,
    )

    repair_task = Threads.@spawn begin
        if no_repair
            # Drain repair_ch without processing
            for _ in repair_ch end
            verbose && println("Repair worker disabled (--no-repair)")
        else
            repair_worker!(
                repair_ch,
                solution_found,
                best_viols,
                n,
                line_offsets_cpu,
                line_points_cpu,
                100,  # max_attempts for repair
                verbose,
                save_fn,
            )
        end
    end

    # Graceful shutdown
    wait(gpu_task)
    close(validation_ch)
    wait(validator_task)
    wait(cpu_task)
    close(repair_ch)
    wait(repair_task)

    elapsed = time() - start_time

    # Gather solution if found
    solution_grid = nothing
    solution_traj_id = 0
    if solution_found[]
        # Solution is already saved by validator or repair worker
        # We just need to report success
        verbose && println("-"^60)
        verbose && println("SUCCESS")
    else
        verbose && println("-"^60)
        verbose && println("NO SOLUTION FOUND")
    end

    # Print unified statistics
    verbose && @printf("Time: %.2fs | Tried: %d | Best: %d viols | Rate: %.1f/s\n",
                       elapsed, total_tried[], best_viols[], total_tried[] / elapsed)

    # Per-source breakdown
    verbose && println("\nPer-source statistics:")
    verbose && @printf("  GPU batches: %d\n", gpu_batches_completed[])
    verbose && @printf("  CPU trajectories: %d\n", cpu_trajectories_tried[])
    verbose && @printf("  Repair attempts: %d\n", repair_attempts[])

    if solution_found[]
        return true, solution_grid, elapsed, Dict(
            :success => 1,
            :tried => total_tried[],
            :gpu => gpu_batches_completed[],
            :cpu => cpu_trajectories_tried[],
            :repair => repair_attempts[]
        ), seed
    else
        return false, nothing, elapsed, Dict(
            :success => 0,
            :tried => total_tried[],
            :gpu => gpu_batches_completed[],
            :cpu => cpu_trajectories_tried[],
            :repair => repair_attempts[]
        ), seed
    end
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
    cpu_threads = args["cpu-threads"]
    repair_threshold = args["repair-threshold"]
    cpu_alpha_min = args["cpu-alpha-min"]
    cpu_alpha_max = args["cpu-alpha-max"]
    cpu_gamma_min = args["cpu-gamma-min"]
    cpu_gamma_max = args["cpu-gamma-max"]
    no_cpu = args["no-cpu"]
    no_repair = args["no-repair"]

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
    if cpu_threads <= 0
        println("ERROR: cpu-threads must be > 0")
        return 1
    end
    if repair_threshold < 0
        println("ERROR: repair-threshold must be >= 0")
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

    verbose && println("Checking ROCm AMD GPU...")
    try
        AMDGPU.functional() || error("AMDGPU not functional. Install/configure ROCm runtime first.")
        devs = AMDGPU.devices()
        isempty(devs) && error("No AMDGPU devices found")
        dev = devs[1]
        verbose && println("  Device: $(dev)")
    catch e
        println("ERROR: No ROCm-capable AMD GPU runtime available: $e")
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

    # Phase 0 finding: For N≥17, use α=6.0, γ=4.0 as defaults unless overridden
    if n >= 17
        if isnothing(alpha_override)
            alpha_override = 6.0
            verbose && println("  Phase 0 default: α=6.0 for N≥17")
        end
        if isnothing(gamma_override)
            gamma_override = 4.0
            verbose && println("  Phase 0 default: γ=4.0 for N≥17")
        end
    end

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

    # CPU parameter diversification defaults
    cpu_alpha_min_val = isnothing(cpu_alpha_min) ? max(1.0, α - 2.0) : cpu_alpha_min
    cpu_alpha_max_val = isnothing(cpu_alpha_max) ? min(30.0, α + 2.0) : cpu_alpha_max
    cpu_gamma_min_val = isnothing(cpu_gamma_min) ? max(1.0, γ - 1.5) : cpu_gamma_min
    cpu_gamma_max_val = isnothing(cpu_gamma_max) ? min(10.0, γ + 1.5) : cpu_gamma_max

    # Dispatch logic
    if no_cpu
        # GPU-only mode (fallback to original solve_rocm pattern)
        # For now, call solve_hybrid with cpu_threads=0 (TODO: implement GPU-only path)
        verbose && println("\n⚠ GPU-only mode (--no-cpu) not fully implemented yet")
        verbose && println("  Falling back to hybrid mode with minimal CPU usage\n")
    end

    # Call solve_hybrid
    success, grid, elapsed, stats, final_seed = solve_hybrid(
        cfg,
        line_offsets_cpu,
        line_points_cpu,
        point_col_scale_cpu,
        seed,
        outdir;
        verbose=verbose,
        batch_size=batch_size,
        max_batches=max_batches,
        cpu_threads=cpu_threads,
        repair_threshold=repair_threshold,
        cpu_alpha_min=cpu_alpha_min_val,
        cpu_alpha_max=cpu_alpha_max_val,
        cpu_gamma_min=cpu_gamma_min_val,
        cpu_gamma_max=cpu_gamma_max_val,
        no_repair=no_repair,
        col_normalization=col_norm_mode,
    )

    return success ? 0 : 1
end

# main_rocm_v2.jl
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
