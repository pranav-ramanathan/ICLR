#!/usr/bin/env julia
#=
N3L Neural Config Learner (UDE-style)
====================================
Learns alpha/gamma schedules with a small neural network while keeping
top-k mask validation for true no-three-in-line correctness checks.
=#

using OrdinaryDiffEq
using DiffEqCallbacks
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using SciMLSensitivity
using Random
using Printf
using Dates
using ArgParse
using JLD2
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
# Core Helpers
# ============================================================================

function topk_mask(x::AbstractVector{<:Real}, k::Int)
    idx = partialsortperm(x, 1:k; rev=true)
    m = falses(length(x))
    @inbounds for i in idx
        m[i] = true
    end
    return BitVector(m)
end

function parse_n_spec(spec::AbstractString)
    ns = Int[]
    for raw in split(spec, ",")
        part = strip(raw)
        isempty(part) && continue
        if occursin(":", part)
            ab = split(part, ":")
            length(ab) == 2 || error("Invalid range spec: $part")
            a = parse(Int, strip(ab[1]))
            b = parse(Int, strip(ab[2]))
            if a <= b
                append!(ns, a:b)
            else
                append!(ns, a:-1:b)
            end
        else
            push!(ns, parse(Int, part))
        end
    end
    isempty(ns) && error("No n values parsed from spec: $spec")
    sort!(unique!(ns))
    return ns
end

@inline default_alpha(n::Int) = n <= 10 ? 10.0 * (n / 6) : 40.0
@inline default_gamma(n::Int) = n <= 10 ? 5.0 : 15.0
@inline logistic(x) = inv(one(x) + exp(-x))
@inline logit(p) = log(p / (one(p) - p))

const ALPHA_SCALE_LO = 0.40
const ALPHA_SCALE_HI = 10.0
const GAMMA_SCALE_LO = 0.30
const GAMMA_SCALE_HI = 3.0

function compute_triples(n::Int)
    triples = NTuple{3,Int}[]
    for x1 in 1:n, y1 in 1:n
        for x2 in 1:n, y2 in 1:n
            (x2, y2) <= (x1, y1) && continue
            for x3 in 1:n, y3 in 1:n
                (x3, y3) <= (x2, y2) && continue
                if x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0
                    push!(triples, (
                        (x1 - 1) * n + y1,
                        (x2 - 1) * n + y2,
                        (x3 - 1) * n + y3
                    ))
                end
            end
        end
    end
    return triples
end

function precompute_triples_cache(ns::Vector{Int}; verbose::Bool = true)
    triples_map = Dict{Int,Vector{NTuple{3,Int}}}()
    unique_ns = sort(unique(ns))
    for n in unique_ns
        t0 = time()
        triples_map[n] = compute_triples(n)
        if verbose
            @printf(
                "Triples for n=%d: %d (%.2fs)\n",
                n, length(triples_map[n]), time() - t0
            )
        end
    end
    return triples_map
end

function get_triples!(triples_map::Dict{Int,Vector{NTuple{3,Int}}}, n::Int)
    if !haskey(triples_map, n)
        triples_map[n] = compute_triples(n)
    end
    return triples_map[n]
end

function count_violations(x_bin::BitVector, triples)
    c = 0
    @inbounds for (i, j, k) in triples
        c += x_bin[i] & x_bin[j] & x_bin[k]
    end
    return c
end

function biased_init(rng, N::Int, target_density::Float64)
    a = max(0.5, 2.0 * target_density)
    b = max(0.5, 2.0 * (1.0 - target_density))
    x0 = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        u = rand(rng)^(1 / a)
        v = rand(rng)^(1 / b)
        x0[i] = u / (u + v)
    end
    return x0
end

function collinearity_penalty(x, triples)
    total = zero(eltype(x))
    @inbounds for (i, j, k) in triples
        total += x[i] * x[j] * x[k]
    end
    return total / max(1, length(triples))
end

function binary_penalty(x)
    s = zero(eltype(x))
    one_x = one(eltype(x))
    @inbounds for i in eachindex(x)
        xi = x[i]
        s += xi * xi * (one_x - xi) * (one_x - xi)
    end
    return s / length(x)
end

function count_penalty(x, target::Int)
    target_x = convert(eltype(x), target)
    d = sum(x) - target_x
    return (d * d) / (target_x * target_x)
end

# ============================================================================
# Tiny MLP for alpha/gamma schedule
# ============================================================================

Base.@kwdef struct NNArch
    in_dim::Int = 6
    hidden::Int = 12
end

@inline function num_params(arch::NNArch)
    return arch.hidden * arch.in_dim + arch.hidden + 2 * arch.hidden + 2
end

function init_theta(rng, arch::NNArch)
    θ = randn(rng, num_params(arch)) .* 0.02
    W1, b1, W2, b2 = unpack_params(θ, arch)
    W2 .*= 0.05

    # Start close to default alpha/gamma instead of the range midpoint.
    pα = (1.0 - ALPHA_SCALE_LO) / (ALPHA_SCALE_HI - ALPHA_SCALE_LO)
    pγ = (1.0 - GAMMA_SCALE_LO) / (GAMMA_SCALE_HI - GAMMA_SCALE_LO)
    b2[1] = logit(pα)
    b2[2] = logit(pγ)

    return Vector{Float64}(θ)
end

function unpack_params(θ::AbstractVector, arch::NNArch)
    idx = 1

    w1_len = arch.hidden * arch.in_dim
    W1 = reshape(@view(θ[idx:idx + w1_len - 1]), arch.hidden, arch.in_dim)
    idx += w1_len

    b1 = @view(θ[idx:idx + arch.hidden - 1])
    idx += arch.hidden

    w2_len = 2 * arch.hidden
    W2 = reshape(@view(θ[idx:idx + w2_len - 1]), 2, arch.hidden)
    idx += w2_len

    b2 = @view(θ[idx:idx + 1])
    return W1, b1, W2, b2
end

function feature_vector(x, t, n::Int, horizon::Float64)
    T = eltype(x)
    one_t = one(T)
    n_norm = convert(T, n) / convert(T, 20)
    t_norm = convert(T, t) / (convert(T, horizon) + convert(T, 1e-8))

    μ = sum(x) / length(x)
    m2 = sum(abs2, x) / length(x)

    m3 = zero(T)
    bin_mass = zero(T)
    @inbounds for i in eachindex(x)
        xi = x[i]
        m3 += xi * xi * xi
        bin_mass += xi * (one_t - xi)
    end
    m3 /= length(x)
    bin_mass /= length(x)

    return T[n_norm, t_norm, μ, m2, m3, bin_mass]
end

function model_forward(θ::AbstractVector, arch::NNArch, feat)
    W1, b1, W2, b2 = unpack_params(θ, arch)
    h = tanh.(W1 * feat .+ b1)
    return W2 * h .+ b2
end

function predict_alpha_gamma(x, t, n::Int, θ::AbstractVector, arch::NNArch, horizon::Float64)
    feat = feature_vector(x, t, n, horizon)
    raw = model_forward(θ, arch, feat)

    α0 = default_alpha(n)
    γ0 = default_gamma(n)
    α_lo, α_hi = ALPHA_SCALE_LO * α0, ALPHA_SCALE_HI * α0
    γ_lo, γ_hi = GAMMA_SCALE_LO * γ0, GAMMA_SCALE_HI * γ0

    α = α_lo + (α_hi - α_lo) * logistic(raw[1])
    γ = γ_lo + (γ_hi - γ_lo) * logistic(raw[2])
    return α, γ
end

# ============================================================================
# Neural ODE Dynamics
# ============================================================================

function make_rhs_nn(n::Int, triples, arch::NNArch, horizon::Float64)
    function rhs!(dx, x, p, t)
        θ = p
        α, γ = predict_alpha_gamma(x, t, n, θ, arch, horizon)
        one_x = one(eltype(x))

        @inbounds for i in eachindex(x)
            xi = x[i]
            dx[i] = one_x - γ * xi * (2 - 6 * xi + 4 * xi * xi)
        end

        @inbounds for (i, j, k) in triples
            dx[i] -= α * x[j] * x[k]
            dx[j] -= α * x[i] * x[k]
            dx[k] -= α * x[i] * x[j]
        end

        @inbounds for i in eachindex(x)
            if x[i] <= 0 && dx[i] < 0
                dx[i] = 0
            elseif x[i] >= 1 && dx[i] > 0
                dx[i] = 0
            end
        end
    end

    return rhs!
end

# ============================================================================
# Differentiable Surrogate Loss (for training)
# ============================================================================

function trajectory_surrogate_loss(
    n::Int,
    triples,
    x0_ref::Vector{Float64},
    θ::AbstractVector,
    arch::NNArch;
    T_train::Float64 = 8.0,
    save_points::Int = 6
)
    target = 2n
    x0 = copy(x0_ref)

    rhs! = make_rhs_nn(n, triples, arch, T_train)
    saveat = range(0.0, T_train; length=save_points)
    prob = ODEProblem(rhs!, x0, (0.0, T_train), θ)

    sol = try
        solve(
            prob,
            Tsit5();
            p=θ,
            sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
            saveat=saveat,
            abstol=1e-5,
            reltol=1e-4,
            maxiters=1_000_000
        )
    catch
        return convert(eltype(θ), 1e6)
    end

    xT = sol.u[end]
    col_T = collinearity_penalty(xT, triples)
    bin_T = binary_penalty(xT)
    cnt_T = count_penalty(xT, target)
    α0 = convert(eltype(xT), default_alpha(n))
    γ0 = convert(eltype(xT), default_gamma(n))
    αT, γT = predict_alpha_gamma(xT, T_train, n, θ, arch, T_train)
    dαT = αT / α0 - one(eltype(xT))
    dγT = γT / γ0 - one(eltype(xT))
    cfg_reg = dαT * dαT + dγT * dγT

    traj_col = zero(eltype(xT))
    @inbounds for u in sol.u
        traj_col += collinearity_penalty(u, triples)
    end
    traj_col /= length(sol.u)

    reg = convert(eltype(θ), 1e-4) * sum(abs2, θ)
    return 12 * col_T + 8 * cnt_T + 2 * bin_T + 4 * traj_col + 0.25 * cfg_reg + reg
end

function training_loss(
    θ::AbstractVector,
    samples::Vector{NamedTuple{(:n, :x0, :seed),Tuple{Int,Vector{Float64},UInt64}}},
    triples_map::Dict{Int,Vector{NTuple{3,Int}}},
    arch::NNArch,
    T_train::Float64
)
    total = zero(eltype(θ))
    @inbounds for sample in samples
        total += trajectory_surrogate_loss(
            sample.n, triples_map[sample.n], sample.x0, θ, arch; T_train=T_train
        )
    end
    return total / length(samples)
end

function training_loss_batch(
    θ::AbstractVector,
    samples::Vector{NamedTuple{(:n, :x0, :seed),Tuple{Int,Vector{Float64},UInt64}}},
    triples_map::Dict{Int,Vector{NTuple{3,Int}}},
    arch::NNArch,
    T_train::Float64,
    batch_ids::Vector{Int}
)
    total = zero(eltype(θ))
    @inbounds for id in batch_ids
        sample = samples[id]
        total += trajectory_surrogate_loss(
            sample.n, triples_map[sample.n], sample.x0, θ, arch; T_train=T_train
        )
    end
    return total / length(batch_ids)
end

# ============================================================================
# True Validity Check (top-k + exact violation count)
# ============================================================================

@enum Status RUNNING SUCCESS TIMEOUT

function run_trajectory_nn(
    n::Int,
    triples,
    seed::UInt64,
    θ::AbstractVector,
    arch::NNArch;
    T::Float64 = 15.0,
    check_interval::Float64 = 0.1
)
    N = n^2
    target = 2n
    rng = Xoshiro(seed)
    x0 = biased_init(rng, N, target / N)
    status = Ref(RUNNING)

    α0, γ0 = predict_alpha_gamma(x0, 0.0, n, θ, arch, T)
    rhs! = make_rhs_nn(n, triples, arch, T)

    function check!(integrator)
        x_bin = topk_mask(integrator.u, target)
        viols = count_violations(x_bin, triples)
        if viols == 0
            status[] = SUCCESS
            terminate!(integrator)
        end
    end

    cb = PeriodicCallback(check!, check_interval; save_positions=(false, false))
    prob = ODEProblem(rhs!, x0, (0.0, T), θ)
    sol = solve(
        prob,
        Tsit5();
        p=θ,
        abstol=1e-6,
        reltol=1e-4,
        callback=cb,
        save_everystep=false,
        save_start=false,
        maxiters=1_000_000
    )

    x_final = sol.u[end]
    x_bin = topk_mask(x_final, target)
    viols = count_violations(x_bin, triples)
    αT, γT = predict_alpha_gamma(x_final, T, n, θ, arch, T)

    if status[] == RUNNING
        status[] = (viols == 0) ? SUCCESS : TIMEOUT
    end

    return status[], x_bin, viols, α0, γ0, αT, γT
end

function evaluate_model(
    θ::AbstractVector,
    arch::NNArch,
    ns::Vector{Int},
    eval_trials::Int,
    base_seed::UInt64;
    triples_map::Union{Nothing,Dict{Int,Vector{NTuple{3,Int}}}} = nothing,
    T::Float64 = 15.0,
    check_interval::Float64 = 0.1,
    verbose::Bool = true
)
    triples_lookup = isnothing(triples_map) ? Dict{Int,Vector{NTuple{3,Int}}}() : triples_map
    total = 0
    success = 0
    viol_sum = 0.0

    for n in ns
        triples = get_triples!(triples_lookup, n)
        local_success = 0
        local_viols = 0

        for trial in 1:eval_trials
            seed = splitmix64(base_seed ⊻ UInt64(n) ⊻ (UInt64(trial) << 20) ⊻ 0x123456789abcdef0)
            _, _, viols, α0, γ0, αT, γT = run_trajectory_nn(
                n, triples, seed, θ, arch; T=T, check_interval=check_interval
            )
            total += 1
            success += (viols == 0)
            viol_sum += viols
            local_success += (viols == 0)
            local_viols += viols

            if verbose
                @printf(
                    "  n=%d trial=%d: viols=%d | α0=%.2f γ0=%.2f | αT=%.2f γT=%.2f\n",
                    n, trial, viols, α0, γ0, αT, γT
                )
            end
        end

        verbose && @printf(
            "eval n=%d: success=%d/%d | avg_viols=%.3f\n",
            n, local_success, eval_trials, local_viols / eval_trials
        )
    end

    success_rate = success / max(total, 1)
    avg_viols = viol_sum / max(total, 1)
    return Dict(
        :total => total,
        :success => success,
        :success_rate => success_rate,
        :avg_viols => avg_viols
    )
end

# ============================================================================
# Training
# ============================================================================

function build_train_samples(ns::Vector{Int}, seeds_per_n::Int, base_seed::UInt64)
    samples = NamedTuple{(:n, :x0, :seed),Tuple{Int,Vector{Float64},UInt64}}[]
    for n in ns, k in 1:seeds_per_n
        seed = splitmix64(base_seed ⊻ UInt64(n) ⊻ (UInt64(k) << 32))
        N = n^2
        target = 2n
        rng = Xoshiro(seed)
        x0 = biased_init(rng, N, target / N)
        push!(samples, (n=n, x0=x0, seed=seed))
    end
    return samples
end

function train_model(
    θ0::Vector{Float64},
    arch::NNArch,
    ns::Vector{Int},
    seeds_per_n::Int,
    base_seed::UInt64;
    triples_map::Union{Nothing,Dict{Int,Vector{NTuple{3,Int}}}} = nothing,
    T_train::Float64 = 8.0,
    train_iters::Int = 200,
    batch_size::Int = 0,
    refine_iters::Int = 75,
    lr::Float64 = 5e-3,
    refine_method::String = "adam",
    refine_lr::Float64 = 1e-3,
    verbose::Bool = true
)
    triples_lookup = isnothing(triples_map) ? precompute_triples_cache(ns; verbose=verbose) : triples_map

    samples = build_train_samples(ns, seeds_per_n, base_seed)
    n_samples = length(samples)
    effective_batch = if batch_size <= 0
        n_samples
    else
        min(batch_size, n_samples)
    end
    use_minibatch = effective_batch < n_samples
    verbose && @printf(
        "Training samples: %d (%d n-values x %d seeds) | batch=%d%s\n",
        n_samples, length(ns), seeds_per_n, effective_batch, use_minibatch ? " (minibatch)" : " (full)"
    )

    adtype = Optimization.AutoZygote()

    batch_rng = Xoshiro(splitmix64(base_seed ⊻ 0x9c2f6a3b8d145e77))
    batch_ids = collect(1:effective_batch)
    batch_perm = randperm(batch_rng, n_samples)
    batch_cursor = Ref(1)
    function sample_batch!()
        if use_minibatch
            if batch_cursor[] + effective_batch - 1 > n_samples
                batch_perm = randperm(batch_rng, n_samples)
                batch_cursor[] = 1
            end
            start_idx = batch_cursor[]
            stop_idx = start_idx + effective_batch - 1
            @inbounds for i in eachindex(batch_ids)
                batch_ids[i] = batch_perm[start_idx + i - 1]
            end
            batch_cursor[] = stop_idx + 1
        else
            @inbounds for i in eachindex(batch_ids)
                batch_ids[i] = i
            end
        end
        return nothing
    end
    sample_batch!()

    objective_full = (x, p) -> training_loss(x, samples, triples_lookup, arch, T_train)
    objective_batch = (x, p) -> training_loss_batch(x, samples, triples_lookup, arch, T_train, batch_ids)
    objective_adam = use_minibatch ? objective_batch : objective_full
    optf = Optimization.OptimizationFunction(objective_adam, adtype)
    optprob = Optimization.OptimizationProblem(optf, θ0)

    iter = Ref(0)
    function callback_adam(θ, l)
        iter[] += 1
        if verbose && (iter[] == 1 || iter[] % 5 == 0)
            tag = use_minibatch ? "mb-loss" : "loss"
            @printf("[ADAM %4d/%4d] %s=%.6f\n", iter[], train_iters, tag, l)
        end
        sample_batch!()
        return false
    end

    res1 = Optimization.solve(
        optprob,
        OptimizationOptimisers.ADAM(lr);
        callback=callback_adam,
        maxiters=train_iters
    )
    θ_best = Vector{Float64}(res1.u)
    if verbose
        @printf("Post-ADAM full loss: %.6f\n", objective_full(θ_best, nothing))
    end

    refine_method_l = lowercase(refine_method)
    if refine_iters > 0 && refine_method_l != "none"
        iter[] = 0

        if refine_method_l == "adam"
            verbose && @printf("Starting ADAM refinement on full batch (lr=%.5f)...\n", refine_lr)
            optf2 = Optimization.OptimizationFunction(objective_full, adtype)
            optprob2 = Optimization.OptimizationProblem(optf2, θ_best)
            function callback_refine_adam(θ, l)
                iter[] += 1
                if verbose && (iter[] == 1 || iter[] % 5 == 0)
                    @printf("[ADAM-R %4d/%4d] loss=%.6f\n", iter[], refine_iters, l)
                end
                return false
            end

            res2 = Optimization.solve(
                optprob2,
                OptimizationOptimisers.ADAM(refine_lr);
                callback=callback_refine_adam,
                maxiters=refine_iters
            )
            θ_best = Vector{Float64}(res2.u)
        elseif refine_method_l == "bfgs"
            if use_minibatch && verbose
                println("BFGS refinement uses full-batch objective for line-search stability.")
            end
            verbose && println("Starting BFGS refinement...")
            optf2 = Optimization.OptimizationFunction(objective_full, adtype)
            optprob2 = Optimization.OptimizationProblem(optf2, θ_best)
            function callback_refine_bfgs(θ, l)
                iter[] += 1
                if verbose && (iter[] == 1 || iter[] % 5 == 0)
                    @printf("[BFGS   %4d/%4d] loss=%.6f\n", iter[], refine_iters, l)
                end
                return false
            end

            res2 = Optimization.solve(
                optprob2,
                OptimizationOptimJL.BFGS();
                callback=callback_refine_bfgs,
                maxiters=refine_iters
            )
            θ_best = Vector{Float64}(res2.u)
        else
            error("Unknown refine method: $(refine_method). Use adam | bfgs | none.")
        end
    end

    final_loss = objective_full(θ_best, nothing)
    return θ_best, final_loss
end

# ============================================================================
# Save / Load weights
# ============================================================================

function _to_string_key_dict(d::AbstractDict)
    out = Dict{String,Any}()
    for (k, v) in d
        key = string(k)
        if v isa AbstractDict
            out[key] = _to_string_key_dict(v)
        else
            out[key] = v
        end
    end
    return out
end

function _to_symbol_key_dict(d::AbstractDict)
    out = Dict{Symbol,Any}()
    for (k, v) in d
        key = Symbol(k)
        if v isa AbstractDict
            out[key] = _to_symbol_key_dict(v)
        else
            out[key] = v
        end
    end
    return out
end

function save_weights(path::String, θ::Vector{Float64}, arch::NNArch, metadata::Dict{Symbol,Any})
    dir = dirname(path)
    if dir != "."
        mkpath(dir)
    end

    metadata_str = _to_string_key_dict(metadata)
    JLD2.jldsave(
        path;
        theta=θ,
        arch_in_dim=arch.in_dim,
        arch_hidden=arch.hidden,
        metadata=metadata_str
    )
end

function load_weights(path::String)
    payload = JLD2.load(path)

    θ = Vector{Float64}(payload["theta"])
    arch = NNArch(
        in_dim=Int(payload["arch_in_dim"]),
        hidden=Int(payload["arch_hidden"])
    )

    metadata = Dict{Symbol,Any}()
    if haskey(payload, "metadata")
        raw_meta = payload["metadata"]
        if raw_meta isa AbstractDict
            metadata = _to_symbol_key_dict(raw_meta)
        end
    end

    return θ, arch, metadata
end

function normalize_weights_path(path::String)
    root, ext = splitext(path)
    if lowercase(ext) == ".bin"
        return root * ".jld2"
    end
    if isempty(ext)
        return path * ".jld2"
    end
    return path
end

# ============================================================================
# Parallel solve with learned model (top-k validity)
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
        mins = floor(Int, (seconds % 3600) / 60)
        return @sprintf("%dh %dm", hours, mins)
    end
end

function print_grid(grid)
    n = size(grid, 1)
    for i in 1:n
        print("  ")
        for j in 1:n
            print(grid[i, j] ? "1 " : "0 ")
        end
        println()
    end
end

function save_solution_nn(
    n::Int,
    grid,
    traj_id::Int,
    R::Int,
    T::Float64,
    seed::UInt64,
    outdir::String,
    weights_path::String,
    α0::Float64,
    γ0::Float64,
    αT::Float64,
    γT::Float64
)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "$(outdir)/$(n)"
    mkpath(dir)
    filename = "$(dir)/sol_$(timestamp)_traj$(traj_id)_nn.txt"

    open(filename, "w") do io
        println(io, "# n=$(n)")
        println(io, "# target=$(2n)")
        println(io, "# mode=nn")
        println(io, "# trajectory_id=$(traj_id)")
        println(io, "# R=$(R)")
        println(io, "# T=$(T)")
        println(io, "# seed=$(seed)")
        println(io, "# weights=$(weights_path)")
        println(io, @sprintf("# alpha0=%.6f, gamma0=%.6f", α0, γ0))
        println(io, @sprintf("# alphaT=%.6f, gammaT=%.6f", αT, γT))
        println(io, "# timestamp=$(Dates.format(now(), "yyyy-mm-ddTHH:MM:SSZ"))")
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

    println("Saved: $filename")
end

function solve_n3l_nn(
    n::Int,
    R::Int,
    T::Float64,
    seed::UInt64,
    outdir::String,
    θ::AbstractVector,
    arch::NNArch;
    triples_map::Union{Nothing,Dict{Int,Vector{NTuple{3,Int}}}} = nothing,
    check_interval::Float64 = 0.1,
    verbose::Bool = true,
    progress_interval::Int = 50,
    weights_path::String = "nn_weights.jld2"
)
    target = 2n
    triples_lookup = isnothing(triples_map) ? Dict{Int,Vector{NTuple{3,Int}}}() : triples_map
    triples = get_triples!(triples_lookup, n)

    verbose && println("="^60)
    verbose && @printf("N3L NN solve n=%d target=%d | R=%d T=%.1f | threads=%d\n", n, target, R, T, Threads.nthreads())
    verbose && @printf("seed=%d | triples=%d\n", seed, length(triples))
    verbose && println("="^60)

    solution_found = Atomic{Bool}(false)
    trajectories_tried = Atomic{Int}(0)
    best_viols = Atomic{Int}(typemax(Int))
    last_progress = Atomic{Int}(0)

    solution_lock = ReentrantLock()
    solution_grid = nothing
    solution_traj_id = 0
    win_α0 = 0.0
    win_γ0 = 0.0
    win_αT = 0.0
    win_γT = 0.0

    thread_histograms = [Dict{Int,Int}() for _ in 1:Threads.nthreads()]
    start_time = time()

    Threads.@threads for id in 1:R
        if solution_found[]
            continue
        end

        traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(id) << 1))
        _, x_bin, viols, α0, γ0, αT, γT = run_trajectory_nn(
            n, triples, traj_seed, θ, arch; T=T, check_interval=check_interval
        )

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
                        win_α0, win_γ0, win_αT, win_γT = α0, γ0, αT, γT
                        elapsed = time() - start_time
                        verbose && @printf(
                            "SOLUTION traj=%d time=%.2fs tried=%d | α0=%.2f γ0=%.2f\n",
                            id, elapsed, tried, α0, γ0
                        )
                    end
                end
            end
        elseif viols < best_viols[]
            old_best = best_viols[]
            if viols < old_best && atomic_cas!(best_viols, old_best, viols) == old_best
                verbose && @printf("[%6d] new best=%d (traj=%d)\n", tried, viols, id)
            end
        end

        if tried - last_progress[] >= progress_interval
            old = last_progress[]
            if atomic_cas!(last_progress, old, tried) == old
                elapsed = time() - start_time
                rate = tried / elapsed
                eta = (R - tried) / max(rate, 1e-9)
                verbose && @printf("[%6d] best=%d | %.1f/s | eta=%s\n", tried, best_viols[], rate, format_time(eta))
            end
        end
    end

    elapsed = time() - start_time

    violation_histogram = Dict{Int,Int}()
    for hist in thread_histograms
        for (v, cnt) in hist
            violation_histogram[v] = get(violation_histogram, v, 0) + cnt
        end
    end

    verbose && println("-"^60)
    if solution_found[]
        verbose && @printf("SUCCESS time=%.2fs tried=%d/%d rate=%.1f/s\n", elapsed, trajectories_tried[], R, trajectories_tried[] / elapsed)
        if !isempty(violation_histogram)
            verbose && println("Violation distribution:")
            for v in sort(collect(keys(violation_histogram)))
                cnt = violation_histogram[v]
                pct = 100 * cnt / trajectories_tried[]
                verbose && @printf("  %2d violations: %5d (%5.2f%%)\n", v, cnt, pct)
            end
        end
        verbose && println("Solution grid:")
        print_grid(solution_grid)
        save_solution_nn(
            n, solution_grid, solution_traj_id, R, T, seed, outdir, weights_path,
            win_α0, win_γ0, win_αT, win_γT
        )
        return true, solution_grid, elapsed, Dict(:success => 1, :tried => trajectories_tried[])
    else
        verbose && @printf("NO SOLUTION time=%.2fs tried=%d best=%d\n", elapsed, trajectories_tried[], best_viols[])
        if !isempty(violation_histogram)
            verbose && println("Violation distribution:")
            for v in sort(collect(keys(violation_histogram)))
                cnt = violation_histogram[v]
                pct = 100 * cnt / trajectories_tried[]
                verbose && @printf("  %2d violations: %5d (%5.2f%%)\n", v, cnt, pct)
            end
        end
        return false, nothing, elapsed, Dict(:success => 0, :tried => trajectories_tried[])
    end
end

# ============================================================================
# CLI
# ============================================================================

function parse_cli_args(args)
    s = ArgParseSettings(
        description = "N3L Neural Config Learner (train/eval/solve)",
        version = "1.0.0",
        add_version = true
    )

    @add_arg_table! s begin
        "--mode"
            help = "train | eval | solve"
            arg_type = String
            default = "train"
        "--n"
            help = "Board size for solve/eval"
            arg_type = Int
        "--R"
            help = "Max trajectories (solve mode)"
            arg_type = Int
            default = 200
        "--T"
            help = "Integration horizon for eval/solve"
            arg_type = Float64
            default = 15.0
        "--T-train"
            help = "Integration horizon during training"
            arg_type = Float64
            default = 8.0
        "--check-interval"
            help = "Validity check interval"
            arg_type = Float64
            default = 0.1
        "--train-ns"
            help = "Training/eval n values (e.g. 8:15 or 8,9,10)"
            arg_type = String
            default = "8:15"
        "--seeds-per-n"
            help = "Training seeds per n"
            arg_type = Int
            default = 2
        "--eval-trials"
            help = "Evaluation trials per n"
            arg_type = Int
            default = 2
        "--train-iters"
            help = "ADAM iterations"
            arg_type = Int
            default = 200
        "--batch-size"
            help = "Mini-batch size for training ADAM (0 = full batch)"
            arg_type = Int
            default = 0
        "--refine-iters"
            help = "Refinement iterations"
            arg_type = Int
            default = 75
        "--refine-method"
            help = "Refinement optimizer: adam | bfgs | none"
            arg_type = String
            default = "adam"
        "--refine-lr"
            help = "Refinement ADAM learning rate (used if --refine-method adam)"
            arg_type = Float64
            default = 0.001
        "--lr"
            help = "ADAM learning rate"
            arg_type = Float64
            default = 0.005
        "--hidden"
            help = "Hidden width of NN"
            arg_type = Int
            default = 12
        "--weights"
            help = "Weight file path"
            arg_type = String
            default = "models/n3l_nn_weights.jld2"
        "--resume"
            help = "Resume training from --weights if file exists"
            action = :store_true
        "--seed"
            help = "Base random seed"
            arg_type = UInt64
        "--outdir"
            help = "Output directory for solved grids"
            arg_type = String
            default = "solutions_nn"
        "--progress-interval"
            help = "Progress print interval (solve mode)"
            arg_type = Int
            default = 50
        "--quiet", "-q"
            help = "Suppress most output"
            action = :store_true
    end

    return parse_args(args, s)
end

function main()
    args = parse_cli_args(ARGS)
    mode = lowercase(args["mode"])
    verbose = !args["quiet"]

    seed = if isnothing(args["seed"])
        rand(RandomDevice(), UInt64)
    else
        args["seed"]
    end

    if mode == "train"
        ns = parse_n_spec(args["train-ns"])
        weights_path = normalize_weights_path(args["weights"])
        triples_map = precompute_triples_cache(ns; verbose=verbose)

        θ0 = nothing
        arch = NNArch(in_dim=6, hidden=args["hidden"])
        if args["resume"] && isfile(weights_path)
            θ0, loaded_arch, meta = load_weights(weights_path)
            arch = loaded_arch
            verbose && println("Resuming from $weights_path")
            verbose && println("Loaded metadata: $meta")
        else
            rng = Xoshiro(seed)
            θ0 = init_theta(rng, arch)
        end

        verbose && @printf(
            "Training mode | ns=%s | hidden=%d | train_iters=%d | batch=%d | refine=%s:%d\n",
            join(ns, ","), arch.hidden, args["train-iters"], args["batch-size"], args["refine-method"], args["refine-iters"]
        )

        θ, final_loss = train_model(
            θ0,
            arch,
            ns,
            args["seeds-per-n"],
            seed;
            triples_map=triples_map,
            T_train=args["T-train"],
            train_iters=args["train-iters"],
            batch_size=args["batch-size"],
            refine_iters=args["refine-iters"],
            lr=args["lr"],
            refine_method=args["refine-method"],
            refine_lr=args["refine-lr"],
            verbose=verbose
        )

        eval_seed = splitmix64(seed ⊻ 0xfacefeed12345678)
        stats = evaluate_model(
            θ,
            arch,
            ns,
            args["eval-trials"],
            eval_seed;
            triples_map=triples_map,
            T=args["T"],
            check_interval=args["check-interval"],
            verbose=verbose
        )

        metadata = Dict{Symbol,Any}()
        metadata[:timestamp] = Dates.format(now(), "yyyy-mm-ddTHH:MM:SSZ")
        metadata[:train_ns] = ns
        metadata[:seeds_per_n] = args["seeds-per-n"]
        metadata[:train_iters] = args["train-iters"]
        metadata[:batch_size] = args["batch-size"]
        metadata[:refine_iters] = args["refine-iters"]
        metadata[:refine_method] = args["refine-method"]
        metadata[:refine_lr] = args["refine-lr"]
        metadata[:lr] = args["lr"]
        metadata[:seed] = seed
        metadata[:final_loss] = final_loss
        metadata[:eval_stats] = stats
        save_weights(weights_path, θ, arch, metadata)

        println("="^60)
        @printf("Training finished. final_loss=%.6f\n", final_loss)
        @printf("Eval success: %d/%d (%.2f%%)\n", stats[:success], stats[:total], 100 * stats[:success_rate])
        @printf("Eval avg violations: %.4f\n", stats[:avg_viols])
        println("Saved weights: $weights_path")
        return 0

    elseif mode == "eval"
        weights_path = normalize_weights_path(args["weights"])
        isfile(weights_path) || error("Weight file not found: $weights_path")
        θ, arch, meta = load_weights(weights_path)
        verbose && println("Loaded metadata: $meta")

        ns = isnothing(args["n"]) ? parse_n_spec(args["train-ns"]) : [args["n"]]
        triples_map = precompute_triples_cache(ns; verbose=verbose)
        stats = evaluate_model(
            θ,
            arch,
            ns,
            args["eval-trials"],
            seed;
            triples_map=triples_map,
            T=args["T"],
            check_interval=args["check-interval"],
            verbose=verbose
        )

        println("="^60)
        @printf("Eval success: %d/%d (%.2f%%)\n", stats[:success], stats[:total], 100 * stats[:success_rate])
        @printf("Eval avg violations: %.4f\n", stats[:avg_viols])
        return 0

    elseif mode == "solve"
        n = args["n"]
        isnothing(n) && error("--n is required in solve mode")
        weights_path = normalize_weights_path(args["weights"])
        isfile(weights_path) || error("Weight file not found: $weights_path")
        triples_map = precompute_triples_cache([n]; verbose=verbose)

        θ, arch, meta = load_weights(weights_path)
        verbose && println("Loaded metadata: $meta")

        success, _, _, _ = solve_n3l_nn(
            n,
            args["R"],
            args["T"],
            seed,
            args["outdir"],
            θ,
            arch;
            triples_map=triples_map,
            check_interval=args["check-interval"],
            verbose=verbose,
            progress_interval=args["progress-interval"],
            weights_path=weights_path
        )

        if success
            println()
            println("Reproduce:")
            println(
                "julia --project=. --threads=$(Threads.nthreads()) $(PROGRAM_FILE) --mode solve --n $(n) --R $(args["R"]) --T $(args["T"]) --weights $(weights_path) --seed $(seed)"
            )
            return 0
        else
            return 1
        end

    else
        error("Invalid --mode '$mode'. Expected train | eval | solve.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
