#!/usr/bin/env julia
#=
N3L UDE Training - Selection Policy Approach
============================================
NN learns which 2n points to select, physics refines the selection.
Based on Lotka-Volterra and SIRHD UDE examples.
=#

using Lux, OrdinaryDiffEq, Optimization, OptimizationOptimisers
using OptimizationOptimJL, ComponentArrays, SciMLSensitivity
using Random, Statistics, LinearAlgebra, Plots, Printf, Dates

# ============================================================================
# Base N3L Functions
# ============================================================================

@inline function splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    z = x
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

function topk_mask(x::AbstractVector{<:Real}, k::Int)
    idx = partialsortperm(x, 1:k; rev=true)
    m = falses(length(x))
    @inbounds for i in idx
        m[i] = true
    end
    return BitVector(m)
end

function compute_triples(n::Int)
    triples = NTuple{3,Int}[]
    
    for x1 in 1:n, y1 in 1:n
        for x2 in 1:n, y2 in 1:n
            (x2, y2) <= (x1, y1) && continue
            for x3 in 1:n, y3 in 1:n
                (x3, y3) <= (x2, y2) && continue
                if x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0
                    push!(triples, (
                        (x1-1)*n + y1,
                        (x2-1)*n + y2,
                        (x3-1)*n + y3
                    ))
                end
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
    E_col = 0.0
    @inbounds for (i, j, k) in triples
        E_col += x[i] * x[j] * x[k]
    end
    
    E_count = -cfg.β * sum(x)
    
    E_bin = 0.0
    @inbounds for i in eachindex(x)
        E_bin += x[i]^2 * (1 - x[i])^2
    end
    
    return cfg.α * E_col + E_count + cfg.γ * E_bin
end

function gradient!(g, x, triples, cfg::Config)
    @inbounds for i in eachindex(x)
        xi = x[i]
        g[i] = -cfg.β + cfg.γ * xi * (2 - 6*xi + 4*xi*xi)
    end
    
    @inbounds for (i, j, k) in triples
        g[i] += cfg.α * x[j] * x[k]
        g[j] += cfg.α * x[i] * x[k]
        g[k] += cfg.α * x[i] * x[j]
    end
    
    return g
end

# ============================================================================
# Problem Setup
# ============================================================================

# Set random seed for reproducibility
rng = Xoshiro(1111)

# Problem parameters
const n_problem = 14
const N_problem = n_problem^2
const target_points = 2 * n_problem

# Configuration
const cfg = Config(n=n_problem, α=300.0, β=1.0, γ=20.0, T=60.0)

# Precompute collinear triples
const triples = compute_triples(n_problem)
println("Collinear triples: $(length(triples))")

# Generate initial conditions for training
const u0_samples = [biased_init(Xoshiro(1000 + i), N_problem, target_points/N_problem) 
                    for i in 1:20]

# Time span
const tspan = (0.0, cfg.T)
const t_eval = [cfg.T]  # Only evaluate at final time

# ============================================================================
# Feature Extraction
# ============================================================================

function extract_features(x, triples, cfg::Config)
    N = length(x)
    target = 2 * cfg.n
    x_bin = topk_mask(x, target)
    viols = count_violations(x_bin, triples)
    
    # Count near-violations (2 of 3 active)
    near_viols = 0
    for (i, j, k) in triples
        active = Int(x_bin[i]) + Int(x_bin[j]) + Int(x_bin[k])
        near_viols += (active == 2)
    end
    
    features = Float32[
        sum(x) / N,
        Float32(viols) / length(triples),
        Float32(near_viols) / length(triples),
        maximum(x),
        minimum(x),
        std(x),
        sum(x .> 0.5) / N,
        sum(x .> 0.8) / N,
        sum(x .< 0.2) / N,
        sum(x .* (1 .- x)),
        sum(abs.(x .- 0.5)),
        sum(x[1:div(N,2)]) / max(sum(x), 1e-10),
        sum(x[1:2:end]) / max(sum(x), 1e-10),
        Float32(energy(x, triples, cfg)) / N,
    ]
    
    return features
end

# ============================================================================
# Neural Network Definition
# ============================================================================

# Initialize with smaller weights to prevent explosion
function init_weights(rng, dims...)
    return randn(rng, Float32, dims...) .* 0.01f0
end

# NN outputs selection logits for each grid position
const U = Lux.Chain(
    Lux.Dense(14, 128, tanh; init_weight=init_weights),
    Lux.Dense(128, 128, tanh; init_weight=init_weights),
    Lux.Dense(128, 128, tanh; init_weight=init_weights),
    Lux.Dense(128, N_problem; init_weight=init_weights)
)

# Get the initial parameters and state variables
p, st = Lux.setup(rng, U)
const _st = st

# ============================================================================
# UDE Dynamics: Hybrid Selection + Physics
# ============================================================================

function ude_dynamics!(du, u, p_nn, t, cfg_known, triples_known, g_physics)
    N = length(u)
    target = 2 * cfg_known.n
    
    # 1. Extract features
    features = extract_features(u, triples_known, cfg_known)
    
    # 2. NN outputs selection scores
    selection_scores = U(features, p_nn, _st)[1]
    
    # Normalize to [0, 1]
    selection_scores = (selection_scores .- minimum(selection_scores)) ./ 
                      (maximum(selection_scores) - minimum(selection_scores) .+ 1e-8)
    
    # 3. Physics gradient
    gradient!(g_physics, u, triples_known, cfg_known)
    
    # 4. Time-dependent mixing
    t_norm = t / cfg_known.T
    
    # Phase 1 (0-50%): NN selection dominates
    # Phase 2 (50-100%): Physics refinement dominates
    if t_norm < 0.5
        α_nn = 0.8
        α_physics = 0.2
    else
        α_nn = 0.2
        α_physics = 0.8
    end
    
    # 5. Combined dynamics
    @inbounds for i in eachindex(u)
        # NN selection force
        target_value = selection_scores[i]
        nn_force = α_nn * (target_value - u[i])
        
        # Physics gradient force
        physics_force = -α_physics * g_physics[i]
        
        du[i] = nn_force + physics_force
        
        # Box constraints
        if u[i] <= 0 && du[i] < 0
            du[i] = 0.0
        elseif u[i] >= 1 && du[i] > 0
            du[i] = 0.0
        end
    end
end

# Closure with known parameters
nn_dynamics!(du, u, p_nn, t) = ude_dynamics!(du, u, p_nn, t, cfg, triples, zeros(N_problem))

# Define the ODE problem
prob_nn = ODEProblem(nn_dynamics!, u0_samples[1], tspan, p)

# ============================================================================
# Prediction Function
# ============================================================================

function predict_adjoint(θ, initial_condition = u0_samples[1])
    _prob = remake(prob_nn, u0 = initial_condition, tspan = tspan, p = θ)
    x = Array(solve(_prob, Tsit5(), saveat = t_eval,
                    abstol = 1e-6, reltol = 1e-4,
                    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    return x
end

# ============================================================================
# Loss Function
# ============================================================================

function loss_adjoint(θ, n_samples = 5)
    total_loss = 0.0
    
    for i in 1:n_samples
        # Use different initial conditions
        initial_condition = u0_samples[min(i, length(u0_samples))]
        
        # Predict final state
        x_final_mat = predict_adjoint(θ, initial_condition)
        x_final = x_final_mat[:, end]
        
        # Evaluate violations
        x_bin = topk_mask(x_final, target_points)
        viols = count_violations(x_bin, triples)
        
        # Energy at final state
        E_final = energy(x_final, triples, cfg)
        
        # Loss components
        violation_loss = Float32(viols)
        energy_loss = Float32(max(0, E_final))
        
        total_loss += violation_loss + 0.01 * energy_loss
    end
    
    # Regularization
    reg_loss = 0.0001 * sum(abs2, θ)
    
    return total_loss / n_samples + reg_loss
end

# ============================================================================
# Testing Function
# ============================================================================

function test_model(θ, n_tests = 10)
    min_violations = Inf
    success_count = 0
    violation_counts = Int[]
    
    for i in 1:n_tests
        rng_test = Xoshiro(5000 + i)
        x0 = biased_init(rng_test, N_problem, target_points/N_problem)
        
        x_final_mat = predict_adjoint(θ, x0)
        x_final = x_final_mat[:, end]
        x_bin = topk_mask(x_final, target_points)
        viols = count_violations(x_bin, triples)
        
        push!(violation_counts, viols)
        min_violations = min(min_violations, viols)
        
        if viols == 0
            success_count += 1
            @printf("  ✓ Found solution in test %d!\n", i)
            
            # Save solution
            grid = reshape(x_bin, (cfg.n, cfg.n))
            timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
            outdir = "ude_solutions/$(cfg.n)"
            mkpath(outdir)
            filename = "$(outdir)/ude_sol_$(timestamp)_test$(i).txt"
            
            open(filename, "w") do io
                println(io, "# UDE Solution")
                println(io, "# n=$(cfg.n), target=$(target_points)")
                println(io, "# α=$(cfg.α), β=$(cfg.β), γ=$(cfg.γ)")
                println(io, "#")
                println(io, "# Grid (0/1):")
                for row in 1:cfg.n
                    println(io, join(Int.(grid[row, :]), " "))
                end
            end
        end
    end
    
    avg_viols = mean(violation_counts)
    @printf("  Test: Min=%d, Avg=%.1f, Success=%d/%d\n", 
            min_violations, avg_viols, success_count, n_tests)
    
    return min_violations, success_count
end

# ============================================================================
# Callback
# ============================================================================

iter = 0
best_violations = Inf

function callback(state, l)
    global iter, best_violations
    iter += 1
    
    if iter % 10 == 0
        println("Iteration $iter: Loss = $l")
    end
    
    # Test every 25 iterations
    if iter % 25 == 0
        println("\n--- Testing current model ---")
        min_viols, successes = test_model(state.u, 10)
        
        if min_viols < best_violations
            best_violations = min_viols
            @printf("  → New best: %d violations!\n", best_violations)
        end
        println()
    end
    
    return false
end

# ============================================================================
# Training
# ============================================================================

println("="^60)
println("N3L UDE Training - Hybrid Selection Policy")
@printf("n=%d, target=%d points\n", n_problem, target_points)
@printf("Hyperparameters: α=%.1f, β=%.1f, γ=%.1f\n", cfg.α, cfg.β, cfg.γ)
println("="^60)

# Convert to ComponentArray
α = ComponentArray{Float32}(p)

# Optimization setup
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_adjoint(x, 5), adtype)
optprob = Optimization.OptimizationProblem(optf, α)

# Stage 1: ADAM with higher learning rate
println("\n=== Stage 1: ADAM (lr=0.01) ===")
iter = 0
@time res1 = Optimization.solve(
    optprob,
    OptimizationOptimisers.Adam(0.01),
    callback = callback,
    maxiters = 200
)

println("\nLoss after Stage 1: ", loss_adjoint(res1.u, 20))

# Stage 2: ADAM with lower learning rate
println("\n=== Stage 2: ADAM (lr=0.001) ===")
iter = 0
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
@time res2 = Optimization.solve(
    optprob2,
    OptimizationOptimisers.Adam(0.001),
    callback = callback,
    maxiters = 200
)

println("\nLoss after Stage 2: ", loss_adjoint(res2.u, 20))

# Stage 3: L-BFGS for fine-tuning
println("\n=== Stage 3: L-BFGS ===")
iter = 0
optprob3 = Optimization.OptimizationProblem(optf, res2.u)
@time res3 = Optimization.solve(
    optprob3,
    LBFGS(linesearch = BackTracking()),
    callback = callback,
    maxiters = 100
)

println("\nFinal loss after L-BFGS: ", loss_adjoint(res3.u, 20))

# Store the best parameters
p_trained = res3.u

# ============================================================================
# Final Testing
# ============================================================================

println("\n" * "="^60)
println("FINAL TEST (50 trajectories)")
println("="^60)
min_viols, successes = test_model(p_trained, 50)

println("\n" * "="^60)
println("Training Complete!")
@printf("Best violations achieved: %d\n", best_violations)
@printf("Final test: Min=%d, Success=%d/50\n", min_viols, successes)
println("="^60)

# ============================================================================
# Analysis and Visualization
# ============================================================================

# Test prediction on a few samples
println("\n=== Sample Predictions ===")
for i in 1:3
    test_x0 = u0_samples[i]
    x_final_mat = predict_adjoint(p_trained, test_x0)
    x_final = x_final_mat[:, end]
    x_bin = topk_mask(x_final, target_points)
    viols = count_violations(x_bin, triples)
    
    @printf("Sample %d: Final violations = %d\n", i, viols)
    
    if viols == 0
        println("  ✓ Valid solution!")
        grid = reshape(x_bin, (cfg.n, cfg.n))
        for row in 1:cfg.n
            print("  ")
            for col in 1:cfg.n
                print(grid[row, col] ? "● " : "· ")
            end
            println()
        end
    end
end

# Diagnostic: Check NN output behavior
println("\n=== NN Output Diagnostics ===")
test_x = u0_samples[1]
test_features = extract_features(test_x, triples, cfg)
test_selection = U(test_features, p_trained, _st)[1]
println("Selection scores - Min: $(minimum(test_selection)), Max: $(maximum(test_selection)), Mean: $(mean(test_selection))")
println("Number of high-confidence selections (>0.7): $(count(test_selection .> 0.7))")
println("Number of low-confidence selections (<0.3): $(count(test_selection .< 0.3))")

println("\n✓ Training complete!")