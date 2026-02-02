#!/usr/bin/env julia
#=
N3L Pure Gradient Flow - ACCURATE STATS
========================================
Complete version with gamma tuning support.
All trajectories run to completion for accurate statistics.
=#

using OrdinaryDiffEq
using DiffEqCallbacks
using Random
using Printf
using Dates
using ArgParse

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
# Configuration
# ============================================================================

Base.@kwdef struct Config
    n::Int
    R::Int = 100
    T::Float64 = 10.0
    check_interval::Float64 = 0.1
    α::Float64 = n <= 10 ? 10.0 * (n / 6) : 40.0
    β::Float64 = 1.0
    γ::Float64 = n <= 10 ? 5.0 : 15.0
end

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_cli_args(args)
    s = ArgParseSettings(
        description = "N3L Pure Gradient Flow - Accurate Statistics",
        version = "1.0.0",
        add_version = true
    )
    
    @add_arg_table! s begin
        "n"
            help = "Board size (n x n grid)"
            arg_type = Int
            required = true
        "--R"
            help = "Number of parallel trajectories"
            arg_type = Int
            default = 200
        "--T"
            help = "Max integration time in seconds"
            arg_type = Float64
            default = 15.0
        "--alpha"
            help = "Override alpha penalty coefficient (default: 10*(n/6) for n≤10, 40 for n>10)"
            arg_type = Float64
        "--gamma"
            help = "Override gamma binary regularization (default: 5 for n≤10, 15 for n>10)"
            arg_type = Float64
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
    end
    
    return parse_args(args, s)
end

# ============================================================================
# Precompute Collinear Triples
# ============================================================================

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

# ============================================================================
# Energy and Gradient
# ============================================================================

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
# Validation
# ============================================================================

function count_violations(x_bin, triples)
    count = 0
    @inbounds for (i, j, k) in triples
        count += x_bin[i] & x_bin[j] & x_bin[k]
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
# ODE System - PURE GRADIENT FLOW
# ============================================================================

function make_rhs(triples, cfg::Config)
    g = zeros(cfg.n^2)
    
    function rhs!(dx, x, p, t)
        gradient!(g, x, triples, cfg)
        @inbounds for i in eachindex(x)
            dx[i] = -g[i]
            # Box constraints
            if x[i] <= 0 && dx[i] < 0
                dx[i] = 0.0
            elseif x[i] >= 1 && dx[i] > 0
                dx[i] = 0.0
            end
        end
    end
    
    return rhs!
end

# ============================================================================
# Single Trajectory - NO GLOBAL TERMINATION
# ============================================================================

@enum Status RUNNING SUCCESS TIMEOUT

function run_trajectory(cfg::Config, triples, rng)
    N = cfg.n^2
    target = 2 * cfg.n
    
    x0 = biased_init(rng, N, target / N)
    status = Ref(RUNNING)
    
    rhs! = make_rhs(triples, cfg)
    
    # Check for success, but only terminate THIS trajectory
    function check!(integrator)
        x = integrator.u
        x_bin = topk_mask(x, target)
        viols = count_violations(x_bin, triples)
        
        if viols == 0
            status[] = SUCCESS
            terminate!(integrator)
        end
    end
    
    cb = PeriodicCallback(check!, cfg.check_interval; save_positions=(false, false))
    
    prob = ODEProblem(rhs!, x0, (0.0, cfg.T))
    sol = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-4, callback=cb,
                save_everystep=false, save_start=false, maxiters=1_000_000)
    
    # Final evaluation
    x_final = sol.u[end]
    x_bin = topk_mask(x_final, target)
    viols = count_violations(x_bin, triples)
    
    if status[] == RUNNING
        status[] = (viols == 0) ? SUCCESS : TIMEOUT
    end
    
    return status[], x_bin, viols
end

# ============================================================================
# Parallel Search - ALL TRAJECTORIES COMPLETE
# ============================================================================

function solve_n3l(n::Int, R::Int, T::Float64, seed::UInt64, outdir::String; 
                   alpha_override::Union{Nothing,Float64}=nothing,
                   gamma_override::Union{Nothing,Float64}=nothing,
                   verbose::Bool=true)
    # Create config with optional overrides
    cfg = if !isnothing(alpha_override) && !isnothing(gamma_override)
        Config(n=n, R=R, T=T, α=alpha_override, γ=gamma_override)
    elseif !isnothing(alpha_override)
        Config(n=n, R=R, T=T, α=alpha_override)
    elseif !isnothing(gamma_override)
        Config(n=n, R=R, T=T, γ=gamma_override)
    else
        Config(n=n, R=R, T=T)
    end
    
    target = 2n
    
    verbose && println("="^50)
    verbose && @printf("N3L Pure Gradient Flow - ACCURATE STATS\n")
    verbose && @printf("n=%d, target=%d points\n", n, target)
    verbose && @printf("R=%d trajectories, T=%.1fs, threads=%d\n", R, T, Threads.nthreads())
    verbose && @printf("Hyperparameters: α=%.1f, β=%.1f, γ=%.1f\n", cfg.α, cfg.β, cfg.γ)
    verbose && @printf("seed=%d\n", seed)
    verbose && println("NOTE: All trajectories run to completion for accurate statistics")
    
    triples = compute_triples(n)
    verbose && @printf("Collinear triples: %d\n", length(triples))
    verbose && println("-"^50)
    
    # Store all results
    results_lock = ReentrantLock()
    all_results = Tuple{Status, BitVector, Int}[]
    
    start_time = time()
    
    # Let ALL trajectories complete - no early termination
    Threads.@threads for id in 1:R
        traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(id) << 1))
        rng = Xoshiro(traj_seed)
        status, x_bin, viols = run_trajectory(cfg, triples, rng)
        
        lock(results_lock) do
            push!(all_results, (status, x_bin, viols))
            
            # Progress reporting for first few successes
            if status == SUCCESS && count(r -> r[1] == SUCCESS, all_results) <= 5
                verbose && @printf("  ✓ Worker %d found solution #%d (%.2fs, %d violations final check)\n", 
                                  id, count(r -> r[1] == SUCCESS, all_results), time()-start_time, viols)
            end
        end
    end
    
    elapsed = time() - start_time
    
    # Calculate ACCURATE statistics
    successes = count(r -> r[1] == SUCCESS, all_results)
    timeouts = count(r -> r[1] == TIMEOUT, all_results)
    violation_history = [r[3] for r in all_results]
    
    verbose && println("-"^50)
    verbose && println("ACCURATE STATISTICS (all trajectories completed):")
    verbose && @printf("  Total runs: %d\n", R)
    verbose && @printf("  Successes: %d (%.2f%%)\n", successes, 100*successes/R)
    verbose && @printf("  Timeouts: %d (%.2f%%)\n", timeouts, 100*timeouts/R)
    
    if !isempty(violation_history)
        sorted_viols = sort(violation_history)
        verbose && println("  Violation distribution:")
        verbose && @printf("    Min: %d\n", minimum(sorted_viols))
        verbose && @printf("    25th percentile: %d\n", sorted_viols[max(1, div(length(sorted_viols), 4))])
        verbose && @printf("    Median: %d\n", sorted_viols[div(length(sorted_viols), 2)])
        verbose && @printf("    75th percentile: %d\n", sorted_viols[max(1, 3*div(length(sorted_viols), 4))])
        verbose && @printf("    Max: %d\n", maximum(sorted_viols))
        
        close_calls = count(v -> v <= 5, sorted_viols)
        very_close = count(v -> v <= 2, sorted_viols)
        verbose && @printf("  Close calls (≤5 violations): %d (%.2f%%)\n", 
                          close_calls, 100*close_calls/length(sorted_viols))
        verbose && @printf("  Very close (≤2 violations): %d (%.2f%%)\n", 
                          very_close, 100*very_close/length(sorted_viols))
    end
    
    verbose && println("-"^50)
    
    if successes > 0
        # Save first solution found
        first_success_idx = findfirst(r -> r[1] == SUCCESS, all_results)
        _, first_grid, _ = all_results[first_success_idx]
        grid = reshape(first_grid, (n, n))
        
        verbose && println("✓✓✓ SUCCESS ✓✓✓")
        verbose && @printf("Found %d solution%s in %.3fs\n", successes, successes > 1 ? "s" : "", elapsed)
        verbose && @printf("Success rate: %.2f%%\n", 100*successes/R)
        verbose && println("\nFirst solution:")
        print_grid(grid)
        save_solution(n, grid, R, T, seed, outdir)
        
        return true, grid, elapsed, Dict(:success=>successes, :timeout=>timeouts), seed
    else
        verbose && println("✗✗✗ FAILED ✗✗✗")
        verbose && @printf("Time: %.3fs\n", elapsed)
        
        if !isempty(violation_history)
            min_viol = minimum(violation_history)
            verbose && @printf("\nBest achieved: %d violations\n", min_viol)
        end
        
        return false, nothing, elapsed, Dict(:success=>0, :timeout=>timeouts), seed
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

function save_solution(n, grid, R, T, seed, outdir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "$(outdir)/$(n)"
    mkpath(dir)
    filename = "$(dir)/sol_$(timestamp).txt"
    
    open(filename, "w") do io
        println(io, "# n=$(n)")
        println(io, "# target=$(2n)")
        println(io, "# R=$(R)")
        println(io, "# T=$(T)")
        println(io, "# seed=$(seed)")
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
    alpha_override = args["alpha"]
    gamma_override = args["gamma"]
    outdir = args["outdir"]
    quiet = args["quiet"]
    verbose = !quiet
    
    seed = if isnothing(args["seed"])
        rand(RandomDevice(), UInt64)
    else
        args["seed"]
    end
    
    success, grid, elapsed, stats, used_seed = solve_n3l(n, R, T, seed, outdir; 
                                                         alpha_override=alpha_override,
                                                         gamma_override=gamma_override,
                                                         verbose=verbose)
    
    if success
        println()
        alpha_str = isnothing(alpha_override) ? "" : " --alpha $(alpha_override)"
        gamma_str = isnothing(gamma_override) ? "" : " --gamma $(gamma_override)"
        println("Reproduce: julia --threads=12 main_v2.jl $(n) --R $(R) --T $(T)$(alpha_str)$(gamma_str) --seed $(used_seed)")
    end
    
    return success ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
