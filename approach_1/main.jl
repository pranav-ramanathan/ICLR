#!/usr/bin/env julia
#=
N3L SciML Solver (Single File)
==============================
Scientific Machine Learning approach to No-Three-In-Line problem.
Gradient flow dynamics with parallel search and early termination.

Usage:
    julia --threads=12 n3l_solver.jl           # Run n=3 to 20
    julia --threads=12 n3l_solver.jl 10        # Single grid size
    julia --threads=12 n3l_solver.jl 3 15      # Range n=3 to n=15

Author: Sandy
Style: Chris Rackauckas / SciML conventions
=#

using OrdinaryDiffEq
using DiffEqCallbacks
using Random
using Printf
using Dates

# ============================================================================
# Configuration
# ============================================================================

Base.@kwdef struct Config
    n::Int
    R::Int = 100                        # Parallel trajectories
    T::Float64 = 10.0                   # Max integration time
    τ::Float64 = 0.5                    # Binarization threshold
    check_interval::Float64 = 0.1       # Early termination check frequency
    α::Float64 = 10.0 * (n / 6)         # Collinearity penalty (scales with n)
    β::Float64 = 1.0                    # Point count reward
    γ::Float64 = 5.0                    # Binary regularization
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
                # Collinear if area of triangle = 0
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
    # Collinearity penalty
    E_col = 0.0
    @inbounds for (i, j, k) in triples
        E_col += x[i] * x[j] * x[k]
    end
    
    # Point count reward
    E_count = -cfg.β * sum(x)
    
    # Binary regularization
    E_bin = 0.0
    @inbounds for i in eachindex(x)
        E_bin += x[i]^2 * (1 - x[i])^2
    end
    
    return cfg.α * E_col + E_count + cfg.γ * E_bin
end

function gradient!(g, x, triples, cfg::Config)
    # Initialize with count + binary terms
    @inbounds for i in eachindex(x)
        xi = x[i]
        g[i] = -cfg.β + cfg.γ * xi * (2 - 6*xi + 4*xi*xi)
    end
    
    # Add collinearity gradient
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

binarize(x, τ) = BitVector(x .>= τ)

function count_violations(x_bin, triples)
    count = 0
    @inbounds for (i, j, k) in triples
        count += x_bin[i] & x_bin[j] & x_bin[k]
    end
    return count
end

# ============================================================================
# Initial Condition (biased toward target density)
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
# ODE System
# ============================================================================

function make_rhs(triples, cfg::Config)
    g = zeros(cfg.n^2)
    
    function rhs!(dx, x, p, t)
        gradient!(g, x, triples, cfg)
        @inbounds for i in eachindex(x)
            dx[i] = -g[i]
            # Soft boundary
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
# Single Trajectory
# ============================================================================

@enum Status RUNNING SUCCESS HOPELESS STUCK TIMEOUT

function run_trajectory(cfg::Config, triples, global_success::Ref{Bool}, rng)
    N = cfg.n^2
    target = 2 * cfg.n
    
    # Biased initial condition
    x0 = biased_init(rng, N, target / N)
    
    # Status tracking
    status = Ref(RUNNING)
    last_energy = Ref(Inf)
    stall_count = Ref(0)
    
    # ODE setup
    rhs! = make_rhs(triples, cfg)
    
    # Callback for early termination
    function check!(integrator)
        global_success[] && (terminate!(integrator); return)
        
        x = integrator.u
        t = integrator.t
        
        x_bin = binarize(x, cfg.τ)
        pts = sum(x_bin)
        viols = count_violations(x_bin, triples)
        
        # Success?
        if pts == target && viols == 0
            status[] = SUCCESS
            terminate!(integrator)
            return
        end
        
        # Track energy
        E = energy(x, triples, cfg)
        if E >= last_energy[]
            stall_count[] += 1
        else
            stall_count[] = 0
        end
        last_energy[] = E
        
        # Hopeless? (too few points, stalling)
        if pts < target - cfg.n ÷ 2 && stall_count[] >= 3 && t > cfg.T / 4
            status[] = HOPELESS
            terminate!(integrator)
            return
        end
        
        # Stuck? (violations persist)
        if viols > 0 && stall_count[] >= 5 && t > 0.6 * cfg.T
            status[] = STUCK
            terminate!(integrator)
            return
        end
    end
    
    cb = PeriodicCallback(check!, cfg.check_interval; save_positions=(false, false))
    
    # Solve
    prob = ODEProblem(rhs!, x0, (0.0, cfg.T))
    sol = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-4, callback=cb,
                save_everystep=false, save_start=false, maxiters=1_000_000)
    
    # Final check
    x_final = sol.u[end]
    x_bin = binarize(x_final, cfg.τ)
    pts = sum(x_bin)
    viols = count_violations(x_bin, triples)
    
    if status[] == RUNNING
        status[] = (pts == target && viols == 0) ? SUCCESS : TIMEOUT
    end
    
    return status[], x_bin, pts, viols
end

# ============================================================================
# Parallel Search
# ============================================================================

function solve_n3l(n::Int; R::Int=100, T::Float64=10.0, verbose::Bool=true)
    cfg = Config(n=n, R=R, T=T)
    target = 2n
    
    verbose && println("="^50)
    verbose && @printf("N3L Solver: n=%d, target=%d points\n", n, target)
    verbose && @printf("R=%d trajectories, T=%.1fs, threads=%d\n", R, T, Threads.nthreads())
    
    # Precompute triples
    triples = compute_triples(n)
    verbose && @printf("Collinear triples: %d\n", length(triples))
    verbose && println("-"^50)
    
    # Shared state
    global_success = Ref(false)
    result_grid = Ref{Union{Nothing, BitVector}}(nothing)
    result_lock = ReentrantLock()
    stats = Dict(:success=>0, :hopeless=>0, :stuck=>0, :timeout=>0)
    
    start_time = time()
    
    # Parallel search
    Threads.@threads for id in 1:R
        global_success[] && continue
        
        rng = Xoshiro(id + round(Int, time() * 1000))
        status, x_bin, pts, viols = run_trajectory(cfg, triples, global_success, rng)
        
        lock(result_lock) do
            if status == SUCCESS && !global_success[]
                global_success[] = true
                result_grid[] = x_bin
                verbose && @printf("  ✓ Worker %d found solution (%.2fs)\n", id, time()-start_time)
            end
            
            key = status == SUCCESS ? :success : 
                  status == HOPELESS ? :hopeless :
                  status == STUCK ? :stuck : :timeout
            stats[key] += 1
        end
    end
    
    elapsed = time() - start_time
    
    # Results
    verbose && println("-"^50)
    if global_success[]
        grid = reshape(result_grid[], (n, n))
        verbose && println("SUCCESS!")
        verbose && @printf("Time: %.3fs\n", elapsed)
        verbose && println("Grid:")
        print_grid(grid)
        save_solution(n, grid)
        return true, grid, elapsed, stats
    else
        verbose && println("FAILED")
        verbose && @printf("Stats: %s\n", stats)
        verbose && @printf("Time: %.3fs\n", elapsed)
        return false, nothing, elapsed, stats
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
# Batch Solver
# ============================================================================

function solve_batch(n_range; R::Int=100, T::Float64=10.0)
    println("\n" * "="^50)
    println("N3L BATCH SOLVER")
    println("="^50)
    
    results = []
    max_solved = 0
    
    for n in n_range
        # Scale parameters
        R_n = min(R * (1 + (n - 6) ÷ 4), 500)
        T_n = T * (1 + (n - 6) / 10)
        
        success, grid, elapsed, stats = solve_n3l(n; R=R_n, T=T_n, verbose=true)
        push!(results, (n=n, success=success, grid=grid, time=elapsed))
        
        if success
            max_solved = n
            save_solution(n, grid)
        else
            println("\n⚠ Stopping: failed at n=$n")
            break
        end
        println()
    end
    
    # Summary
    println("="^50)
    println("SUMMARY")
    println("="^50)
    @printf("%-6s %-10s %-10s\n", "n", "Status", "Time (s)")
    println("-"^30)
    for r in results
        status = r.success ? "✓" : "✗"
        @printf("%-6d %-10s %-10.3f\n", r.n, status, r.time)
    end
    println("-"^30)
    @printf("Max solved: n=%d\n", max_solved)
    @printf("Total time: %.2fs\n", sum(r.time for r in results))
    println("="^50)
    
    return results, max_solved
end

# ============================================================================
# Save Solution
# ============================================================================

function save_solution(n, grid)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "solutions/$(n)"
    mkpath(dir)
    filename = "$(dir)/sol_$(timestamp).txt"
    
    open(filename, "w") do io
        println(io, "# N3L Solution: n=$n, points=$(2n)")
        println(io, "# Generated: $(now())")
        for i in 1:n
            println(io, join(Int.(grid[i, :]), " "))
        end
        println(io, "\n# Coordinates (row, col):")
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
    if length(ARGS) == 0
        # Default: n=3 to 20
        solve_batch(3:20; R=100, T=10.0)
    elseif length(ARGS) == 1
        # Single n
        n = parse(Int, ARGS[1])
        solve_n3l(n; R=200, T=15.0)
    else
        # Range
        n_start = parse(Int, ARGS[1])
        n_end = parse(Int, ARGS[2])
        solve_batch(n_start:n_end; R=100, T=10.0)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
