#!/usr/bin/env julia
#=
N3L UDE Training - Global Energy Correction
===========================================
NN learns a global energy correction based on system-wide features.
Much faster: 1 NN call per timestep instead of N^2 calls.
=#

using Lux, OrdinaryDiffEq, Optimization, OptimizationOptimisers
using OptimizationOptimJL, ComponentArrays, SciMLSensitivity
using Random, Statistics, LinearAlgebra, Printf

# ============================================================================
# Core N3L Functions
# ============================================================================

function topk_mask(x::AbstractVector{<:Real}, k::Int)
    idx = partialsortperm(x, 1:k; rev=true)
    return [i in idx for i in 1:length(x)]
end

function compute_triples(n::Int)
    triples = NTuple{3,Int}[]
    for x1 in 1:n, y1 in 1:n, x2 in 1:n, y2 in 1:n
        (x2, y2) <= (x1, y1) && continue
        for x3 in 1:n, y3 in 1:n
            (x3, y3) <= (x2, y2) && continue
            if x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0
                push!(triples, ((x1-1)*n + y1, (x2-1)*n + y2, (x3-1)*n + y3))
            end
        end
    end
    return triples
end

function count_violations(x_bin, triples)
    count = 0
    @inbounds for (i, j, k) in triples
        count += x_bin[i] & x_bin[j] & x_bin[k]
    end
    return count
end

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

Base.@kwdef struct Config
    n::Int
    α::Float64 = 300.0
    β::Float64 = 1.0
    γ::Float64 = 20.0
    T::Float64 = 60.0
end

function energy(x, triples, cfg::Config)
    E_col = sum(x[i] * x[j] * x[k] for (i, j, k) in triples)
    E_count = -cfg.β * sum(x)
    E_bin = sum(x .^ 2 .* (1 .- x) .^ 2)
    return cfg.α * E_col + E_count + cfg.γ * E_bin
end

function gradient_nonmutating(x, triples, cfg::Config)
    g = similar(x)
    @inbounds for i in eachindex(x)
        xi = x[i]
        g[i] = -cfg.β + cfg.γ * xi * (2 - 6*xi + 4*xi*xi)
    end
    @inbounds for (i, j, k) in triples
        g[i] = g[i] + cfg.α * x[j] * x[k]
        g[j] = g[j] + cfg.α * x[i] * x[k]
        g[k] = g[k] + cfg.α * x[i] * x[j]
    end
    return g
end

# ============================================================================
# Setup
# ============================================================================

rng = Xoshiro(1111)
const n_problem = 14
const N_problem = n_problem^2
const target_points = 2 * n_problem
const cfg = Config(n=n_problem, α=300.0, β=1.0, γ=20.0, T=60.0)
const triples = compute_triples(n_problem)
println("Collinear triples: $(length(triples))")

const u0_samples = [biased_init(Xoshiro(1000 + i), N_problem, target_points/N_problem) 
                    for i in 1:20]
const tspan = (0.0, cfg.T)
const t_eval = [cfg.T]

# ============================================================================
# Global Features Only - AD Compatible
# ============================================================================

function extract_global_features(x, cfg::Config)
    N = length(x)
    
    # Basic statistics
    x_mean = sum(x) / N
    x_var = sum((xi - x_mean)^2 for xi in x) / N
    x_std = sqrt(x_var + 1e-8)
    
    # Binary-related
    binary_count = sum(x .> 0.5)
    binary_density = binary_count / N
    
    # Fuzziness (distance from binary)
    fuzziness = sum(xi * (1 - xi) for xi in x) / N
    
    # Moments
    x_mean2 = sum(x .^ 2) / N
    x_mean3 = sum(x .^ 3) / N
    
    # Distribution shape
    x_min = minimum(x)
    x_max = maximum(x)
    x_range = x_max - x_min
    
    return [
        x_mean,           # 1
        x_std,            # 2
        binary_density,   # 3
        fuzziness,        # 4
        x_mean2,          # 5
        x_mean3,          # 6
        x_min,            # 7
        x_max,            # 8
        x_range,          # 9
    ]
end

# ============================================================================
# Neural Network - Single Global Correction
# ============================================================================

const U = Lux.Chain(
    Lux.Dense(9, 32, tanh),
    Lux.Dense(32, 32, tanh),
    Lux.Dense(32, 1)  # Single scalar output
)

p, st = Lux.setup(rng, U)
const _st = st

# ============================================================================
# UDE Dynamics
# ============================================================================

function ude_dynamics!(du, u, p_nn, t, cfg_known, triples_known)
    N = length(u)
    
    # 1. Physics gradient
    g_physics = gradient_nonmutating(u, triples_known, cfg_known)
    
    # 2. Extract global features
    features = extract_global_features(u, cfg_known)
    
    # 3. NN predicts single energy correction scalar
    correction_scalar = U(features, p_nn, _st)[1][1]
    
    # 4. Apply correction (scale to reasonable magnitude)
    @inbounds for i in 1:N
        du[i] = -g_physics[i] * (1.0 + 0.01 * correction_scalar)
        
        # Box constraints
        if u[i] <= 0 && du[i] < 0
            du[i] = 0.0
        elseif u[i] >= 1 && du[i] > 0
            du[i] = 0.0
        end
    end
end

nn_dynamics!(du, u, p_nn, t) = ude_dynamics!(du, u, p_nn, t, cfg, triples)
prob_nn = ODEProblem(nn_dynamics!, u0_samples[1], tspan, p)

# ============================================================================
# Prediction & Loss
# ============================================================================

function predict_adjoint(θ, initial_condition = u0_samples[1])
    _prob = remake(prob_nn, u0 = initial_condition, p = θ)
    x = Array(solve(_prob, Tsit5(), saveat = t_eval,
                    abstol = 1e-6, reltol = 1e-4,
                    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    return x
end

function loss_adjoint(θ, n_samples = 5)
    total_loss = 0.0
    
    for i in 1:n_samples
        ic = u0_samples[min(i, length(u0_samples))]
        x_final_mat = predict_adjoint(θ, ic)
        x_final = x_final_mat[:, end]
        
        x_bin = topk_mask(x_final, target_points)
        viols = count_violations(x_bin, triples)
        
        E_final = energy(x_final, triples, cfg)
        
        # Heavy violation penalty
        violation_loss = Float32(viols)^2
        energy_loss = Float32(max(0, E_final))
        
        total_loss += violation_loss + 0.001 * energy_loss
    end
    
    reg_loss = 0.00001 * sum(abs2, θ)
    return total_loss / n_samples + reg_loss
end

# ============================================================================
# Testing
# ============================================================================

function test_model(θ, n_tests = 10)
    min_violations = Inf
    success_count = 0
    violation_counts = Int[]
    
    for i in 1:n_tests
        x0 = biased_init(Xoshiro(5000 + i), N_problem, target_points/N_problem)
        x_final_mat = predict_adjoint(θ, x0)
        x_final = x_final_mat[:, end]
        x_bin = topk_mask(x_final, target_points)
        viols = count_violations(x_bin, triples)
        
        push!(violation_counts, viols)
        min_violations = min(min_violations, viols)
        
        if viols == 0
            success_count += 1
        end
    end
    
    avg_viols = mean(violation_counts)
    @printf("  Test: Min=%d, Avg=%.1f, Success=%d/%d\n", 
            min_violations, avg_viols, success_count, n_tests)
    
    return min_violations, success_count
end

# ============================================================================
# Training
# ============================================================================

iter = 0
best_violations = Inf

function callback(state, l)
    global iter, best_violations
    iter += 1
    
    if iter % 10 == 0
        println("Iteration $iter: Loss = $l")
    end
    
    if iter % 25 == 0
        println("\n--- Testing ---")
        min_viols, successes = test_model(state.u, 10)
        
        if min_viols < best_violations
            best_violations = min_viols
            @printf("  → New best: %d violations!\n", best_violations)
        end
        println()
    end
    
    return false
end

println("="^60)
println("N3L UDE Training - Global Energy Correction")
@printf("n=%d, target=%d points\n", n_problem, target_points)
println("="^60)

α = ComponentArray{Float32}(p)

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_adjoint(x, 5), adtype)
optprob = Optimization.OptimizationProblem(optf, α)

println("\n=== Stage 1: ADAM (lr=0.001) ===")
iter = 0
@time res1 = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.001),
    callback = callback,
    maxiters = 300
)

println("\n=== Stage 2: ADAM (lr=0.0001) ===")
iter = 0
optprob2 = remake(optprob, u0 = res1.u)
@time res2 = Optimization.solve(
    optprob2,
    OptimizationOptimisers.Adam(0.0001),
    callback = callback,
    maxiters = 200
)

println("\n=== Final Test (50 trajectories) ===")
test_model(res2.u, 50)