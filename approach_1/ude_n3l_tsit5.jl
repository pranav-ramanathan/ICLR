#!/usr/bin/env julia
#=
N3L UDE Gradient Flow — CPU Tsit5, Line-Based Representation
=============================================================
Single-file script with two modes:
  1) `--mode train`: learn time schedules α(t), γ(t), η(t) using Lux + Optimization.jl
  2) `--mode solve`: run multi-trajectory search using learned schedules (or fixed fallback)

Core constraints implemented:
  - Line CSR precompute and line-based exact violation checker.
  - Gradient-flow ODE with box constraints, no hard top-k in dynamics.
  - Option A mass penalty term: (sum(x)-k)^2 with gradient 2*(sum(x)-k).
  - Schedules bounded via sigmoid squash.
=#

using OrdinaryDiffEq
using DiffEqCallbacks
using SciMLSensitivity
using Lux
using Random
using Printf
using Dates
using ArgParse
using Optimization
using OptimizationOptimisers
using ComponentArrays
using JLD2
using Statistics
using Zygote
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
# Line Representation (kept consistent with main_v6)
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
# Coefficient Selection (fixed fallback in solve mode)
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
    η::Float64 = 1.0
    abstol::Float64 = 1e-6
    reltol::Float64 = 1e-4
end

Base.@kwdef struct ScheduleBounds
    alpha_min::Float64 = 2.0
    alpha_max::Float64 = 24.0
    gamma_min::Float64 = 1.0
    gamma_max::Float64 = 12.0
    eta_min::Float64 = 0.05
    eta_max::Float64 = 5.0
end

Base.@kwdef struct LossWeights
    λ_col::Float64 = 1.0
    λ_bin::Float64 = 0.5
    λ_mass::Float64 = 1.0
end

@inline function target_count(n::Int)
    return 2 * n
end

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_cli_args(args)
    s = ArgParseSettings(
        description = "N3L UDE Gradient Flow — CPU Tsit5 (Lux schedules α(t), γ(t), η(t))",
        version = "1.0.0",
        add_version = true
    )

    @add_arg_table! s begin
        "n"
            help = "Board size (n x n grid)"
            arg_type = Int
            required = true
        "--mode"
            help = "Mode: train | solve"
            arg_type = String
            default = "solve"
        "--model-path"
            help = "Path to learned schedule model (.jld2) for solve mode"
            arg_type = String
            default = ""
        "--R"
            help = "Maximum number of trajectories to try (solve mode)"
            arg_type = Int
            default = 200
        "--T"
            help = "Integration horizon"
            arg_type = Float64
            default = 15.0
        "--check-interval"
            help = "Periodic callback interval for solve mode"
            arg_type = Float64
            default = 0.1
        "--alpha"
            help = "Override fixed alpha (fallback solve mode only)"
            arg_type = Float64
        "--gamma"
            help = "Override fixed gamma (fallback solve mode only)"
            arg_type = Float64
        "--eta"
            help = "Fixed eta (fallback solve mode only)"
            arg_type = Float64
            default = 1.0
        "--coeff-mode"
            help = "Fixed-coefficient strategy when no model: normalized | legacy | density"
            arg_type = String
            default = "normalized"
        "--col-normalization"
            help = "Collinearity normalization: mean-incidence | none"
            arg_type = String
            default = "mean-incidence"
        "--alpha-min"
            help = "Lower bound for α(t)"
            arg_type = Float64
            default = 2.0
        "--alpha-max"
            help = "Upper bound for α(t)"
            arg_type = Float64
            default = 24.0
        "--gamma-min"
            help = "Lower bound for γ(t)"
            arg_type = Float64
            default = 1.0
        "--gamma-max"
            help = "Upper bound for γ(t)"
            arg_type = Float64
            default = 12.0
        "--eta-min"
            help = "Lower bound for η(t)"
            arg_type = Float64
            default = 0.05
        "--eta-max"
            help = "Upper bound for η(t)"
            arg_type = Float64
            default = 5.0
        "--hidden"
            help = "Hidden width for schedule MLP"
            arg_type = Int
            default = 32
        "--train-iters"
            help = "Number of training iterations"
            arg_type = Int
            default = 2000
        "--train-batch"
            help = "Training batch size (trajectories per optimization step)"
            arg_type = Int
            default = 8
        "--train-lr"
            help = "Training learning rate"
            arg_type = Float64
            default = 1e-3
        "--loss-lambda-col"
            help = "Loss weight for continuous collinearity energy"
            arg_type = Float64
            default = 1.0
        "--loss-lambda-bin"
            help = "Loss weight for smooth binarization term"
            arg_type = Float64
            default = 0.5
        "--loss-lambda-mass"
            help = "Loss weight for smooth mass penalty"
            arg_type = Float64
            default = 1.0
        "--abstol"
            help = "ODE absolute tolerance"
            arg_type = Float64
            default = 1e-6
        "--reltol"
            help = "ODE relative tolerance"
            arg_type = Float64
            default = 1e-4
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
            help = "Print progress every N iterations (train) or trajectories (solve)"
            arg_type = Int
            default = 50
    end

    return parse_args(args, s)
end

function validate_args(args)
    n = args["n"]
    n < 2 && error("n must be >= 2")
    args["R"] <= 0 && error("R must be > 0")
    args["T"] <= 0 && error("T must be > 0")
    args["check-interval"] <= 0 && error("check-interval must be > 0")
    args["train-iters"] <= 0 && error("train-iters must be > 0")
    args["train-batch"] <= 0 && error("train-batch must be > 0")
    args["train-lr"] <= 0 && error("train-lr must be > 0")
    args["abstol"] <= 0 && error("abstol must be > 0")
    args["reltol"] <= 0 && error("reltol must be > 0")
    args["hidden"] <= 0 && error("hidden must be > 0")
    args["progress-interval"] <= 0 && error("progress-interval must be > 0")

    mode_l = lowercase(args["mode"])
    !(mode_l in ("train", "solve")) && error("mode must be one of: train, solve")
    !(lowercase(args["coeff-mode"]) in ("normalized", "legacy", "density")) &&
        error("coeff-mode must be one of: normalized, legacy, density")
    !(lowercase(args["col-normalization"]) in ("mean-incidence", "none")) &&
        error("col-normalization must be one of: mean-incidence, none")

    bounds = ScheduleBounds(
        alpha_min=args["alpha-min"],
        alpha_max=args["alpha-max"],
        gamma_min=args["gamma-min"],
        gamma_max=args["gamma-max"],
        eta_min=args["eta-min"],
        eta_max=args["eta-max"],
    )
    bounds.alpha_min <= 0 && error("alpha-min must be > 0")
    bounds.gamma_min <= 0 && error("gamma-min must be > 0")
    bounds.eta_min <= 0 && error("eta-min must be > 0")
    bounds.alpha_max <= bounds.alpha_min && error("alpha-max must be > alpha-min")
    bounds.gamma_max <= bounds.gamma_min && error("gamma-max must be > gamma-min")
    bounds.eta_max <= bounds.eta_min && error("eta-max must be > eta-min")

    return mode_l, bounds
end

# ============================================================================
# Validation (line-based exact checker)
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

@inline function trajectory_seed(seed::UInt64, n::Int, traj_id::Int)
    return splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(traj_id) << 1))
end

@inline function iter_batch_seed(seed::UInt64, n::Int, iter::Int, j::Int, salt::UInt64)
    return splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(iter) << 24) ⊻ (UInt64(j) << 1) ⊻ salt)
end

function init_from_seed(n::Int, seed::UInt64)
    N = n * n
    ρ = target_count(n) / N
    rng = Xoshiro(seed)
    return biased_init(rng, N, ρ)
end

Zygote.@nograd biased_init
Zygote.@nograd init_from_seed

# ============================================================================
# Continuous Energies for Training Loss
# ============================================================================

function collinearity_energy_lines(x::AbstractVector{<:Real}, line_offsets::Vector{Int32}, line_points::Vector{Int32})
    L = length(line_offsets) - 1
    E = 0.0

    @inbounds for l in 1:L
        start_idx = Int(line_offsets[l])
        stop_idx  = Int(line_offsets[l + 1] - 1)

        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for idx in start_idx:stop_idx
            xi = Float64(x[Int(line_points[idx])])
            x2 = xi * xi
            s1 += xi
            s2 += x2
            s3 += x2 * xi
        end

        E += (s1 * s1 * s1 - 3.0 * s1 * s2 + 2.0 * s3) / 6.0
    end

    return E
end

@inline function binarization_energy(x::AbstractVector{<:Real})
    s = 0.0
    @inbounds for i in eachindex(x)
        xi = Float64(x[i])
        s += xi * (1.0 - xi)
    end
    return s
end

# ============================================================================
# Schedule Model
# ============================================================================

@inline σ(x) = 1.0 / (1.0 + exp(-x))

build_schedule_model(hidden::Int) = Lux.Chain(
    Lux.Dense(1, hidden, tanh),
    Lux.Dense(hidden, hidden, tanh),
    Lux.Dense(hidden, 3),
)

Base.@kwdef struct NeuralScheduleRuntime{M,S}
    model::M
    st::S
    bounds::ScheduleBounds
    T::Float64
end

Base.@kwdef struct FixedScheduleRuntime
    α::Float64
    γ::Float64
    η::Float64
end

function schedule_triplet(runtime::NeuralScheduleRuntime, t::Real, p)
    t_norm = runtime.T > 0 ? clamp(Float64(t) / runtime.T, 0.0, 1.0) : 0.0
    raw, _ = Lux.apply(runtime.model, [t_norm], p, runtime.st)

    α = runtime.bounds.alpha_min + (runtime.bounds.alpha_max - runtime.bounds.alpha_min) * σ(Float64(raw[1]))
    γ = runtime.bounds.gamma_min + (runtime.bounds.gamma_max - runtime.bounds.gamma_min) * σ(Float64(raw[2]))
    η = runtime.bounds.eta_min + (runtime.bounds.eta_max - runtime.bounds.eta_min) * σ(Float64(raw[3]))
    return α, γ, η
end

@inline function schedule_triplet(runtime::FixedScheduleRuntime, t::Real, p)
    return runtime.α, runtime.γ, runtime.η
end

function schedule_samples(runtime, p, T::Float64)
    α0, γ0, η0 = schedule_triplet(runtime, 0.0, p)
    αm, γm, ηm = schedule_triplet(runtime, 0.5 * T, p)
    αT, γT, ηT = schedule_triplet(runtime, T, p)
    return (α0, αm, αT), (γ0, γm, γT), (η0, ηm, ηT)
end

# ============================================================================
# ODE System — Line-Based Gradient Flow with Mass Penalty
# ============================================================================

function make_rhs(
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float64},
    cfg::Config,
    schedule_runtime,
)
    N = cfg.n^2
    L = length(line_offsets) - 1
    k_target = target_count(cfg.n)

    function rhs!(dx, x, p, t)
        αt, γt, ηt = schedule_triplet(schedule_runtime, t, p)

        mass = sum(x)
        mass_grad = 2.0 * ηt * (mass - k_target)

        # Base gradient: count term + binary regularization + mass penalty
        @inbounds for i in 1:N
            xi = x[i]
            dx[i] = -cfg.β + γt * xi * (2.0 - 6.0 * xi + 4.0 * xi * xi) + mass_grad
        end

        # Line-based collinearity gradient (pair-sum form)
        @inbounds for l in 1:L
            start_idx = Int(line_offsets[l])
            stop_idx  = Int(line_offsets[l + 1] - 1)

            s1 = 0.0
            s2 = 0.0
            for idx in start_idx:stop_idx
                xp = x[Int(line_points[idx])]
                s1 += xp
                s2 += xp * xp
            end

            for idx in start_idx:stop_idx
                pi = Int(line_points[idx])
                xi = x[pi]
                s1o = s1 - xi
                s2o = s2 - xi * xi
                pair_sum = 0.5 * (s1o * s1o - s2o)
                dx[pi] += αt * point_col_scale[pi] * pair_sum
            end
        end

        # Negative gradient flow with box constraints
        @inbounds for i in 1:N
            dxi = -dx[i]
            if x[i] <= 0.0 && dxi < 0.0
                dx[i] = 0.0
            elseif x[i] >= 1.0 && dxi > 0.0
                dx[i] = 0.0
            else
                dx[i] = dxi
            end
        end
    end

    return rhs!
end

# ============================================================================
# Trajectory Solves
# ============================================================================

@enum Status RUNNING SUCCESS TIMEOUT DIVERGED

function solve_final_state(
    rhs!,
    x0::Vector{Float64},
    T::Float64,
    p;
    abstol::Float64=1e-6,
    reltol::Float64=1e-4,
    sensealg=nothing,
)
    prob = ODEProblem(rhs!, x0, (0.0, T), p)

    if isnothing(sensealg)
        sol = solve(
            prob,
            Tsit5();
            abstol=abstol,
            reltol=reltol,
            save_everystep=false,
            save_start=false,
            maxiters=1_000_000,
        )
    else
        sol = solve(
            prob,
            Tsit5();
            abstol=abstol,
            reltol=reltol,
            sensealg=sensealg,
            save_everystep=false,
            save_start=false,
            maxiters=1_000_000,
        )
    end

    return sol.u[end], sol.retcode
end

function run_trajectory(cfg::Config, line_offsets, line_points, point_col_scale, seed::UInt64, schedule_runtime, p_schedule)
    N = cfg.n^2
    target = target_count(cfg.n)

    x0 = init_from_seed(cfg.n, seed)
    status = Ref(RUNNING)

    rhs! = make_rhs(line_offsets, line_points, point_col_scale, cfg, schedule_runtime)

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

    prob = ODEProblem(rhs!, x0, (0.0, cfg.T), p_schedule)
    sol = solve(
        prob,
        Tsit5();
        abstol=cfg.abstol,
        reltol=cfg.reltol,
        callback=cb,
        save_everystep=false,
        save_start=false,
        maxiters=1_000_000,
    )

    x_final = sol.u[end]
    if !all(isfinite, x_final)
        return DIVERGED, falses(N), typemax(Int)
    end

    x_bin = topk_mask(x_final, target)
    viols = count_violations_lines(x_bin, line_offsets, line_points)

    if status[] == RUNNING
        status[] = (viols == 0) ? SUCCESS : TIMEOUT
    end

    return status[], x_bin, viols
end

# ============================================================================
# Training
# ============================================================================

function training_loss_factory(
    cfg::Config,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float64},
    schedule_runtime::NeuralScheduleRuntime,
    weights::LossWeights,
    train_batch::Int,
    base_seed::UInt64,
    iter_ref::Base.RefValue{Int},
)
    rhs! = make_rhs(line_offsets, line_points, point_col_scale, cfg, schedule_runtime)
    target = target_count(cfg.n)
    sensealg = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    salt = UInt64(0xA11CE5EED1234F0)

    function loss(θ, _)
        iter = iter_ref[]
        total = 0.0

        @inbounds for j in 1:train_batch
            s = iter_batch_seed(base_seed, cfg.n, iter, j, salt)
            x0 = init_from_seed(cfg.n, s)

            xT = try
                xf, _ = solve_final_state(
                    rhs!,
                    x0,
                    cfg.T,
                    θ;
                    abstol=cfg.abstol,
                    reltol=cfg.reltol,
                    sensealg=sensealg,
                )
                xf
            catch
                return Inf
            end

            if !all(isfinite, xT)
                return Inf
            end

            xTc = clamp.(xT, 0.0, 1.0)

            e_col = collinearity_energy_lines(xTc, line_offsets, line_points)
            e_bin = binarization_energy(xTc)
            mass_err = sum(xTc) - target
            e_mass = mass_err * mass_err

            total += weights.λ_col * e_col + weights.λ_bin * e_bin + weights.λ_mass * e_mass
        end

        return total / train_batch
    end

    return loss
end

function evaluate_schedule_metrics(
    cfg::Config,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float64},
    schedule_runtime,
    p_schedule,
    eval_seeds::Vector{UInt64},
)
    rhs! = make_rhs(line_offsets, line_points, point_col_scale, cfg, schedule_runtime)
    target = target_count(cfg.n)

    best_viols = typemax(Int)
    mass_err_sum = 0.0
    bin_measure_sum = 0.0

    for s in eval_seeds
        x0 = init_from_seed(cfg.n, s)
        xT = try
            xf, _ = solve_final_state(
                rhs!,
                x0,
                cfg.T,
                p_schedule;
                abstol=cfg.abstol,
                reltol=cfg.reltol,
            )
            xf
        catch
            return typemax(Int), Inf, Inf
        end

        if !all(isfinite, xT)
            return typemax(Int), Inf, Inf
        end

        xTc = clamp.(xT, 0.0, 1.0)
        x_bin = topk_mask(xTc, target)
        viols = count_violations_lines(x_bin, line_offsets, line_points)
        best_viols = min(best_viols, viols)

        mass_err_sum += abs(sum(xTc) - target)

        local_bin = 0.0
        @inbounds for i in eachindex(xTc)
            xi = xTc[i]
            local_bin += xi * (1.0 - xi)
        end
        bin_measure_sum += local_bin / length(xTc)
    end

    m = length(eval_seeds)
    return best_viols, mass_err_sum / m, bin_measure_sum / m
end

function save_model(path::String, hidden::Int, ps, st, bounds::ScheduleBounds; meta=Dict{String,Any}())
    mkpath(dirname(path))
    jldsave(
        path;
        impl="n3l_ude_tsit5_v1",
        hidden=hidden,
        ps=ps,
        st=st,
        alpha_min=bounds.alpha_min,
        alpha_max=bounds.alpha_max,
        gamma_min=bounds.gamma_min,
        gamma_max=bounds.gamma_max,
        eta_min=bounds.eta_min,
        eta_max=bounds.eta_max,
        meta=meta,
    )
    return path
end

function load_model(path::AbstractString)
    d = load(String(path))

    hidden = Int(get(d, "hidden", 32))
    model = build_schedule_model(hidden)
    ps = d["ps"]
    st = d["st"]
    bounds = ScheduleBounds(
        alpha_min=Float64(d["alpha_min"]),
        alpha_max=Float64(d["alpha_max"]),
        gamma_min=Float64(d["gamma_min"]),
        gamma_max=Float64(d["gamma_max"]),
        eta_min=Float64(d["eta_min"]),
        eta_max=Float64(d["eta_max"]),
    )
    meta = get(d, "meta", Dict{String,Any}())

    return model, ps, st, bounds, hidden, meta
end

function run_train_mode(
    n::Int,
    args,
    seed::UInt64,
    outdir::String,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float64},
    bounds::ScheduleBounds;
    verbose::Bool=true,
)
    cfg = Config(
        n=n,
        T=args["T"],
        β=1.0,
        abstol=args["abstol"],
        reltol=args["reltol"],
    )
    train_iters = args["train-iters"]
    train_batch = args["train-batch"]
    train_lr = args["train-lr"]
    hidden = args["hidden"]
    progress_interval = args["progress-interval"]

    weights = LossWeights(
        λ_col=args["loss-lambda-col"],
        λ_bin=args["loss-lambda-bin"],
        λ_mass=args["loss-lambda-mass"],
    )

    verbose && println("="^68)
    verbose && println("N3L UDE Training — CPU Tsit5 + Lux Schedules")
    verbose && println("="^68)
    verbose && @printf("n=%d, target=%d | T=%.3f | batch=%d | iters=%d | lr=%.3e\n",
                       n, target_count(n), cfg.T, train_batch, train_iters, train_lr)
    verbose && @printf("bounds α:[%.3f, %.3f] γ:[%.3f, %.3f] η:[%.3f, %.3f]\n",
                       bounds.alpha_min, bounds.alpha_max,
                       bounds.gamma_min, bounds.gamma_max,
                       bounds.eta_min, bounds.eta_max)
    verbose && @printf("loss λ: col=%.3f, bin=%.3f, mass=%.3f | abstol=%.1e reltol=%.1e\n",
                       weights.λ_col, weights.λ_bin, weights.λ_mass, cfg.abstol, cfg.reltol)
    verbose && @printf("seed=%d | hidden=%d\n", seed, hidden)
    verbose && println("-"^68)

    rng = Xoshiro(splitmix64(seed ⊻ UInt64(0x5A17E)))
    model = build_schedule_model(hidden)
    ps0, st = Lux.setup(rng, model)
    ps0 = ComponentArray(Lux.f64(ps0))
    st = Lux.f64(st)

    schedule_runtime = NeuralScheduleRuntime(model=model, st=st, bounds=bounds, T=cfg.T)
    iter_ref = Ref(1)
    loss_fn = training_loss_factory(
        cfg,
        line_offsets,
        line_points,
        point_col_scale,
        schedule_runtime,
        weights,
        train_batch,
        seed,
        iter_ref,
    )

    optf = OptimizationFunction(loss_fn, Optimization.AutoZygote())
    optprob = OptimizationProblem(optf, ps0)
    opt = OptimizationOptimisers.Adam(train_lr)

    eval_batch = min(max(4, train_batch), 16)
    eval_salt = UInt64(0xE1EA7E1234567890)
    eval_seeds = [iter_batch_seed(seed, n, 1, j, eval_salt) for j in 1:eval_batch]

    best_loss = Ref(Inf)
    best_ps = Ref(copy(ps0))
    last_reported_iter = Ref(0)

    cb = function (state, l)
        iter = state.iter
        iter_ref[] = iter + 1

        if isfinite(l) && l < best_loss[]
            best_loss[] = l
            best_ps[] = copy(state.u)
        end

        if !isfinite(l)
            println("[train] NaN/Inf loss detected at iter=$iter. Stopping early.")
            return true
        end

        should_report = iter != last_reported_iter[] &&
                        (iter == 1 || iter % progress_interval == 0 || iter == train_iters)
        if verbose && should_report
            last_reported_iter[] = iter
            bviols, avg_mass_err, avg_bin = evaluate_schedule_metrics(
                cfg,
                line_offsets,
                line_points,
                point_col_scale,
                schedule_runtime,
                state.u,
                eval_seeds,
            )
            αs, γs, ηs = schedule_samples(schedule_runtime, state.u, cfg.T)
            bviols_str = (bviols == typemax(Int)) ? "diverged" : string(bviols)
            @printf("[train %6d] loss=%.6e | eval best_viols=%s | avg|sum-k|=%.5f | avg x(1-x)=%.5f\n",
                    iter, Float64(l), bviols_str, avg_mass_err, avg_bin)
            @printf("              α(0,0.5T,T)=[%.3f, %.3f, %.3f] γ=[%.3f, %.3f, %.3f] η=[%.3f, %.3f, %.3f]\n",
                    αs[1], αs[2], αs[3], γs[1], γs[2], γs[3], ηs[1], ηs[2], ηs[3])
        end

        return false
    end

    t0 = time()
    sol = Optimization.solve(optprob, opt; maxiters=train_iters, callback=cb)
    elapsed = time() - t0

    learned_ps = isfinite(best_loss[]) ? best_ps[] : sol.u
    if !isfinite(best_loss[])
        println("Training failed: objective diverged (NaN/Inf).")
        return false, ""
    end

    final_bviols, final_mass_err, final_bin = evaluate_schedule_metrics(
        cfg,
        line_offsets,
        line_points,
        point_col_scale,
        schedule_runtime,
        learned_ps,
        eval_seeds,
    )
    αs, γs, ηs = schedule_samples(schedule_runtime, learned_ps, cfg.T)

    model_dir = joinpath(outdir, string(n))
    mkpath(model_dir)
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    model_path = joinpath(model_dir, "model_ude_$(ts).jld2")
    meta = Dict{String,Any}(
        "trained_at_utc" => Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"),
        "n" => n,
        "T" => cfg.T,
        "train_iters" => train_iters,
        "train_batch" => train_batch,
        "train_lr" => train_lr,
        "best_loss" => best_loss[],
        "eval_best_viols" => final_bviols,
        "eval_avg_mass_err" => final_mass_err,
        "eval_avg_bin_measure" => final_bin,
    )
    save_model(model_path, hidden, learned_ps, st, bounds; meta=meta)

    verbose && println("-"^68)
    verbose && @printf("Training finished in %.2fs | best loss=%.6e\n", elapsed, best_loss[])
    verbose && @printf("Final eval: best_viols=%s | avg|sum-k|=%.5f | avg x(1-x)=%.5f\n",
                       (final_bviols == typemax(Int) ? "diverged" : string(final_bviols)),
                       final_mass_err, final_bin)
    verbose && @printf("Final schedules α(0,0.5T,T)=[%.3f, %.3f, %.3f], γ=[%.3f, %.3f, %.3f], η=[%.3f, %.3f, %.3f]\n",
                       αs[1], αs[2], αs[3], γs[1], γs[2], γs[3], ηs[1], ηs[2], ηs[3])
    verbose && println("Saved model: $model_path")

    return true, model_path
end

# ============================================================================
# Parallel Search (solve mode)
# ============================================================================

function solve_n3l(
    cfg::Config,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float64},
    seed::UInt64,
    outdir::String,
    schedule_runtime,
    p_schedule;
    coeff_mode::String="normalized",
    col_normalization::String="mean-incidence",
    model_path::Union{Nothing,String}=nothing,
    verbose::Bool=true,
    progress_interval::Int=50,
)
    n = cfg.n
    target = target_count(n)

    verbose && println("="^60)
    verbose && println("N3L UDE Solve — CPU Tsit5, Line-Based")
    verbose && println("="^60)
    verbose && @printf("n=%d, target=%d | R=%d | T=%.3f | check=%.3f | threads=%d\n",
                       n, target, cfg.R, cfg.T, cfg.check_interval, Threads.nthreads())
    verbose && @printf("β=%.1f | tolerances: abstol=%.1e reltol=%.1e\n", cfg.β, cfg.abstol, cfg.reltol)
    if schedule_runtime isa NeuralScheduleRuntime
        αs, γs, ηs = schedule_samples(schedule_runtime, p_schedule, cfg.T)
        verbose && println("Schedule mode: learned neural schedule")
        verbose && @printf("  α(0,0.5T,T)=[%.3f, %.3f, %.3f]\n", αs[1], αs[2], αs[3])
        verbose && @printf("  γ(0,0.5T,T)=[%.3f, %.3f, %.3f]\n", γs[1], γs[2], γs[3])
        verbose && @printf("  η(0,0.5T,T)=[%.3f, %.3f, %.3f]\n", ηs[1], ηs[2], ηs[3])
        !isnothing(model_path) && println("  model_path=$(model_path)")
    else
        verbose && println("Schedule mode: fixed coefficients")
        verbose && @printf("  α=%.3f, γ=%.3f, η=%.3f\n",
                           schedule_runtime.α, schedule_runtime.γ, schedule_runtime.η)
    end
    verbose && @printf("Seed: %d | coeff-mode=%s | col-normalization=%s\n",
                       seed, coeff_mode, col_normalization)
    verbose && println("="^60)

    solution_found = Atomic{Bool}(false)
    trajectories_tried = Atomic{Int}(0)
    best_viols = Atomic{Int}(typemax(Int))
    last_progress = Atomic{Int}(0)

    solution_lock = ReentrantLock()
    solution_grid = nothing
    solution_traj_id = 0

    thread_histograms = [Dict{Int,Int}() for _ in 1:Threads.nthreads()]
    start_time = time()

    Threads.@threads for id in 1:cfg.R
        if solution_found[]
            break
        end

        s = trajectory_seed(seed, n, id)
        status, x_bin, viols = run_trajectory(
            cfg,
            line_offsets,
            line_points,
            point_col_scale,
            s,
            schedule_runtime,
            p_schedule,
        )

        tried = atomic_add!(trajectories_tried, 1)
        tid = Threads.threadid()
        if viols != typemax(Int)
            thread_histograms[tid][viols] = get(thread_histograms[tid], viols, 0) + 1
        end

        if status == SUCCESS
            if !solution_found[]
                lock(solution_lock) do
                    if !solution_found[]
                        solution_found[] = true
                        solution_grid = reshape(x_bin, (n, n))
                        solution_traj_id = id
                        elapsed = time() - start_time
                        verbose && @printf("\nSOLUTION! traj=%d, time=%.2fs, tried=%d\n", id, elapsed, tried)
                    end
                end
            end
        elseif viols < best_viols[]
            old_best = best_viols[]
            if viols < old_best && atomic_cas!(best_viols, old_best, viols) == old_best
                verbose && @printf("[%6d] NEW BEST: %d viols (traj %d)\n", tried, viols, id)
            end
        end

        if tried - last_progress[] >= progress_interval
            old = last_progress[]
            if atomic_cas!(last_progress, old, tried) == old
                elapsed = time() - start_time
                rate = tried / max(elapsed, 1e-9)
                eta = (cfg.R - tried) / max(rate, 1e-9)
                best_str = best_viols[] == typemax(Int) ? "∞" : string(best_viols[])
                verbose && @printf("[%6d] best=%s | %.1f/s | eta=%s\n", tried, best_str, rate, format_time(eta))
            end
        end

        if solution_found[]
            break
        end
    end

    elapsed = time() - start_time

    violation_histogram = Dict{Int,Int}()
    for hist in thread_histograms
        for (v, c) in hist
            violation_histogram[v] = get(violation_histogram, v, 0) + c
        end
    end

    verbose && println("-"^60)

    if solution_found[]
        verbose && println("SUCCESS")
        verbose && @printf("Time: %.2fs | Tried: %d/%d (%.1f%%) | Rate: %.1f/s\n",
                           elapsed, trajectories_tried[], cfg.R,
                           100 * trajectories_tried[] / cfg.R,
                           trajectories_tried[] / max(elapsed, 1e-9))

        if !isempty(violation_histogram)
            verbose && println("\nViolation distribution:")
            for v in sort(collect(keys(violation_histogram)))
                count = violation_histogram[v]
                pct = 100 * count / max(1, trajectories_tried[])
                bar = repeat("█", min(40, round(Int, pct / 2)))
                verbose && @printf("  %2d violations: %5d (%5.2f%%) %s\n", v, count, pct, bar)
            end
        end

        verbose && println("\nSolution:")
        print_grid(solution_grid)

        α0, γ0, η0 = schedule_triplet(schedule_runtime, 0.0, p_schedule)
        save_solution(
            n,
            solution_grid,
            solution_traj_id,
            cfg.R,
            cfg.T,
            seed,
            α0,
            γ0,
            η0,
            outdir;
            coeff_mode=coeff_mode,
            col_normalization=col_normalization,
            model_path=model_path,
            schedule_kind=(schedule_runtime isa NeuralScheduleRuntime ? "neural" : "fixed"),
        )

        return true, solution_grid, elapsed, Dict(:success => 1, :tried => trajectories_tried[]), seed
    else
        best_str = best_viols[] == typemax(Int) ? "∞" : string(best_viols[])
        verbose && println("NO SOLUTION FOUND")
        verbose && @printf("Time: %.2fs | Tried: %d | Best: %s viols | Rate: %.1f/s\n",
                           elapsed, trajectories_tried[], best_str,
                           trajectories_tried[] / max(elapsed, 1e-9))

        if !isempty(violation_histogram)
            verbose && println("\nViolation distribution:")
            for v in sort(collect(keys(violation_histogram)))
                count = violation_histogram[v]
                pct = 100 * count / max(1, trajectories_tried[])
                bar = repeat("█", min(40, round(Int, pct / 2)))
                verbose && @printf("  %2d violations: %5d (%5.2f%%) %s\n", v, count, pct, bar)
            end
        end

        return false, nothing, elapsed, Dict(:success => 0, :tried => trajectories_tried[]), seed
    end
end

function run_solve_mode(
    n::Int,
    args,
    seed::UInt64,
    outdir::String,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float64},
    col_norm_mode_l::String,
    bounds_cli::ScheduleBounds;
    verbose::Bool=true,
)
    cfg = Config(
        n=n,
        R=args["R"],
        T=args["T"],
        check_interval=args["check-interval"],
        β=1.0,
        abstol=args["abstol"],
        reltol=args["reltol"],
    )

    model_path_raw = strip(args["model-path"])
    has_model = !isempty(model_path_raw)
    model_path = has_model ? String(model_path_raw) : nothing

    coeff_mode = args["coeff-mode"]
    α_fixed = 0.0
    γ_fixed = 0.0
    η_fixed = clamp(args["eta"], bounds_cli.eta_min, bounds_cli.eta_max)
    mode_l = lowercase(coeff_mode)

    schedule_runtime = nothing
    p_schedule = nothing

    if has_model
        if !isfile(model_path_raw)
            println("ERROR: model path not found: $model_path_raw")
            return false, nothing, 0.0, Dict(:success => 0, :tried => 0), seed
        end
        model, ps, st, bounds_loaded, hidden, _ = load_model(model_path_raw)
        ps = ps isa ComponentArray ? ps : ComponentArray(ps)
        ps = Lux.f64(ps)
        st = Lux.f64(st)
        schedule_runtime = NeuralScheduleRuntime(model=model, st=st, bounds=bounds_loaded, T=cfg.T)
        p_schedule = ps

        verbose && println("Loaded neural schedule model:")
        verbose && @printf("  path=%s\n", model_path_raw)
        verbose && @printf("  hidden=%d\n", hidden)
        verbose && @printf("  bounds α:[%.3f, %.3f] γ:[%.3f, %.3f] η:[%.3f, %.3f]\n",
                           bounds_loaded.alpha_min, bounds_loaded.alpha_max,
                           bounds_loaded.gamma_min, bounds_loaded.gamma_max,
                           bounds_loaded.eta_min, bounds_loaded.eta_max)
    else
        α_fixed, γ_fixed, base_α, base_γ, mode_l, avg_triples_per_var, col_scale = choose_coefficients(
            n,
            line_offsets;
            mode=coeff_mode,
            normalization_mode=col_norm_mode_l,
            alpha_override=args["alpha"],
            gamma_override=args["gamma"],
        )
        schedule_runtime = FixedScheduleRuntime(α=α_fixed, γ=γ_fixed, η=η_fixed)
        p_schedule = nothing

        if verbose
            println("Fixed schedule fallback (no model provided):")
            @printf("  mode=%s | base α=%.3f, γ=%.3f\n", mode_l, base_α, base_γ)
            if mode_l == "density" && !isnan(avg_triples_per_var)
                @printf("  density stats: avg triples/var=%.3f, col-scale=%.4f\n", avg_triples_per_var, col_scale)
            end
            if !isnothing(args["alpha"]) || !isnothing(args["gamma"])
                @printf("  overrides -> α=%.3f, γ=%.3f\n", α_fixed, γ_fixed)
            end
            @printf("  η fixed=%.3f\n", η_fixed)
        end
    end

    return solve_n3l(
        cfg,
        line_offsets,
        line_points,
        point_col_scale,
        seed,
        outdir,
        schedule_runtime,
        p_schedule;
        coeff_mode=mode_l,
        col_normalization=col_norm_mode_l,
        model_path=model_path,
        verbose=verbose,
        progress_interval=args["progress-interval"],
    )
end

# ============================================================================
# Utility
# ============================================================================

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
# Save Solution
# ============================================================================

function save_solution(
    n,
    grid,
    traj_id,
    R,
    T,
    seed,
    α,
    γ,
    η,
    outdir;
    coeff_mode::String="normalized",
    col_normalization::String="mean-incidence",
    model_path::Union{Nothing,String}=nothing,
    schedule_kind::String="fixed",
)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "$(outdir)/$(n)"
    mkpath(dir)
    filename = "$(dir)/sol_ude_tsit5_$(timestamp)_traj$(traj_id).txt"

    open(filename, "w") do io
        println(io, "# n=$(n)")
        println(io, "# target=$(target_count(n))")
        println(io, "# trajectory_id=$(traj_id)")
        println(io, "# R=$(R)")
        println(io, "# T=$(T)")
        println(io, "# seed=$(seed)")
        println(io, "# α=$(α), β=1.0, γ=$(γ), η=$(η)")
        println(io, "# method=CPU Tsit5 Line-Based UDE Gradient Flow")
        println(io, "# schedule_kind=$(schedule_kind)")
        println(io, "# coeff_mode=$(coeff_mode)")
        println(io, "# col_normalization=$(col_normalization)")
        if !isnothing(model_path)
            println(io, "# model_path=$(model_path)")
        end
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
    mode_l, bounds = validate_args(args)

    n = args["n"]
    outdir = args["outdir"]
    quiet = args["quiet"]
    verbose = !quiet
    col_norm_mode = args["col-normalization"]

    seed = if isnothing(args["seed"])
        rand(RandomDevice(), UInt64)
    else
        args["seed"]
    end

    # Precompute line representation once.
    verbose && println("Precomputing lines...")
    t0_lines = time()
    line_offsets, line_points = compute_lines(n)
    lines_elapsed = time() - t0_lines

    line_count, packed_points, max_line_len, triples_equiv = line_stats(line_offsets)
    verbose && @printf("  Lines: %d | Packed points: %d | Max len: %d | Triple-equiv: %d | %.3fs\n",
                       line_count, packed_points, max_line_len, triples_equiv, lines_elapsed)

    point_col_scale, mean_inc, min_inc, max_inc, col_norm_mode_l = compute_point_collinearity_scale(
        n, line_offsets, line_points; mode=col_norm_mode
    )

    if verbose
        if col_norm_mode_l == "mean-incidence"
            @printf("  Collinearity normalization: %s | incidence mean=%.3f, min=%.3f, max=%.3f\n",
                    col_norm_mode_l, mean_inc, min_inc, max_inc)
        else
            @printf("  Collinearity normalization: %s\n", col_norm_mode_l)
        end
        @printf("  Schedule bounds α:[%.3f, %.3f] γ:[%.3f, %.3f] η:[%.3f, %.3f]\n",
                bounds.alpha_min, bounds.alpha_max,
                bounds.gamma_min, bounds.gamma_max,
                bounds.eta_min, bounds.eta_max)
    end

    if mode_l == "train"
        ok, model_path = run_train_mode(
            n,
            args,
            seed,
            outdir,
            line_offsets,
            line_points,
            point_col_scale,
            bounds;
            verbose=verbose,
        )
        if ok
            println()
            println("Training complete.")
            println("Model: $model_path")
            println("Solve with:")
            println("julia --project=. --threads=auto $(PROGRAM_FILE) $(n) --mode solve --model-path $(model_path) --R $(args["R"]) --T $(args["T"]) --check-interval $(args["check-interval"]) --seed $(seed)")
            return 0
        else
            return 1
        end
    else
        success, _, _, _, used_seed = run_solve_mode(
            n,
            args,
            seed,
            outdir,
            line_offsets,
            line_points,
            point_col_scale,
            col_norm_mode_l,
            bounds;
            verbose=verbose,
        )

        if success
            println()
            model_str = isempty(args["model-path"]) ? "" : " --model-path $(args["model-path"])"
            println("Reproduce exact run:")
            println("julia --project=. --threads=auto $(PROGRAM_FILE) $(n) --mode solve --R $(args["R"]) --T $(args["T"]) --check-interval $(args["check-interval"]) --col-normalization $(col_norm_mode_l) --seed $(used_seed)$(model_str)")
            return 0
        else
            return 1
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_with_terminal_log("ude_n3l_tsit5", ARGS) do
        main()
    end)
end
