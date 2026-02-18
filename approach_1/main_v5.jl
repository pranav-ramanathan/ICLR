#!/usr/bin/env julia
#=
N3L Pure Gradient Flow - SOLUTION FINDING MODE (OPTIMIZED)
===========================================================
Stops immediately on first solution with lock-free parallel execution.
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
        description = "N3L Pure Gradient Flow - Solution Finding (Optimized)",
        version = "1.0.0",
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
            help = "Max integration time in seconds"
            arg_type = Float64
            default = 15.0
        "--alpha"
            help = "Override alpha penalty coefficient"
            arg_type = Float64
        "--gamma"
            help = "Override gamma binary regularization"
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
        "--progress-interval"
            help = "Print progress every N trajectories"
            arg_type = Int
            default = 50
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
# Single Trajectory
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
# Parallel Search - LOCK-FREE OPTIMIZED
# ============================================================================

function solve_n3l(n::Int, R::Int, T::Float64, seed::UInt64, outdir::String; 
                   alpha_override::Union{Nothing,Float64}=nothing,
                   gamma_override::Union{Nothing,Float64}=nothing,
                   verbose::Bool=true,
                   progress_interval::Int=50)
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
    
    verbose && println("="^60)
    verbose && @printf("n=%d, target=%d | α=%.1f, β=%.1f, γ=%.1f\n", n, target, cfg.α, cfg.β, cfg.γ)
    verbose && @printf("Max: %d traj, T=%.1fs, %d threads\n", R, T, Threads.nthreads())
    verbose && @printf("Seed: %d | Progress: every %d\n", seed, progress_interval)
    verbose && println("="^60)
    
    triples = compute_triples(n)
    verbose && @printf("Triples: %d\n", length(triples))
    verbose && println("-"^60)
    
    # Atomics for lock-free coordination
    solution_found = Atomic{Bool}(false)
    trajectories_tried = Atomic{Int}(0)
    best_viols = Atomic{Int}(typemax(Int))
    last_progress = Atomic{Int}(0)
    
    # Solution storage (only locked when solution found)
    solution_lock = ReentrantLock()
    solution_grid = nothing
    solution_traj_id = 0
    
    # Per-thread histograms (no lock contention)
    thread_histograms = [Dict{Int,Int}() for _ in 1:Threads.nthreads()]
    
    start_time = time()
    
    # Parallel search with early termination
    Threads.@threads for id in 1:R
        # Check if solution already found
        if solution_found[]
            break
        end
        
        traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(id) << 1))
        rng = Xoshiro(traj_seed)
        status, x_bin, viols = run_trajectory(cfg, triples, rng)
        
        # Atomic counter increment (no lock)
        tried = atomic_add!(trajectories_tried, 1)
        
        # Thread-local histogram update (no contention)
        tid = Threads.threadid()
        thread_histograms[tid][viols] = get(thread_histograms[tid], viols, 0) + 1
        
        # Check for solution
        if viols == 0
            if !solution_found[]
                lock(solution_lock) do
                    # Double-check after acquiring lock
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
        # Check for new best (atomic CAS - no lock unless we win)
        elseif viols < best_viols[]
            old_best = best_viols[]
            if viols < old_best && atomic_cas!(best_viols, old_best, viols) == old_best
                verbose && @printf("[%6d] ★ NEW BEST: %d viols (traj %d)\n", tried, viols, id)
            end
        end
        
        # Progress updates (atomic CAS to prevent spam)
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
        
        # Early exit if solution found
        if solution_found[]
            break
        end
    end
    
    elapsed = time() - start_time
    
    # Merge histograms from all threads
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
        
        # Print violation distribution
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
        save_solution(n, solution_grid, solution_traj_id, R, T, seed, cfg.α, cfg.γ, outdir)
        
        return true, solution_grid, elapsed, Dict(:success=>1, :tried=>trajectories_tried[]), seed
    else
        verbose && println("✗✗✗ NO SOLUTION FOUND ✗✗✗")
        verbose && @printf("Time: %.2fs | Tried: %d | Best: %d viols | Rate: %.1f/s\n", 
                          elapsed, trajectories_tried[], best_viols[], trajectories_tried[]/elapsed)
        
        # Print violation distribution
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
        mins = floor(Int, (seconds % 3600) / 60)
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

function save_solution(n, grid, traj_id, R, T, seed, α, γ, outdir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "$(outdir)/$(n)"
    mkpath(dir)
    filename = "$(dir)/sol_$(timestamp)_traj$(traj_id).txt"
    
    open(filename, "w") do io
        println(io, "# n=$(n)")
        println(io, "# target=$(2n)")
        println(io, "# trajectory_id=$(traj_id)")
        println(io, "# R=$(R)")
        println(io, "# T=$(T)")
        println(io, "# seed=$(seed)")
        println(io, "# α=$(α), β=1.0, γ=$(γ)")
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
    progress_interval = args["progress-interval"]
    
    seed = if isnothing(args["seed"])
        rand(RandomDevice(), UInt64)
    else
        args["seed"]
    end
    
    success, grid, elapsed, stats, used_seed = solve_n3l(n, R, T, seed, outdir; 
                                                         alpha_override=alpha_override,
                                                         gamma_override=gamma_override,
                                                         verbose=verbose,
                                                         progress_interval=progress_interval)
    
    if success
        println()
        alpha_str = isnothing(alpha_override) ? "" : " --alpha $(alpha_override)"
        gamma_str = isnothing(gamma_override) ? "" : " --gamma $(gamma_override)"
        println("Reproduce exact solution:")
        println("julia --project=. --threads=12 $(PROGRAM_FILE) $(n) --R $(R) --T $(T)$(alpha_str)$(gamma_str) --seed $(used_seed)")
    end
    
    return success ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_with_terminal_log("cpu_v5", ARGS) do
        main()
    end)
end