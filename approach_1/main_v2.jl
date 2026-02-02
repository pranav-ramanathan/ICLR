#!/usr/bin/env julia
#=
N3L Pure Gradient Flow Baseline
================================
Clean implementation without perturbations.
This will be our baseline to compare UDE approach against.
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
        description = "N3L Pure Gradient Flow Baseline",
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
# Single Trajectory - SIMPLIFIED
# ============================================================================

@enum Status RUNNING SUCCESS TIMEOUT

function run_trajectory(cfg::Config, triples, global_success::Ref{Bool}, rng)
    N = cfg.n^2
    target = 2 * cfg.n
    
    x0 = biased_init(rng, N, target / N)
    status = Ref(RUNNING)
    
    rhs! = make_rhs(triples, cfg)
    
    # Simple early success check only
    function check!(integrator)
        global_success[] && (terminate!(integrator); return)
        
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
# Parallel Search
# ============================================================================

function solve_n3l(n::Int, R::Int, T::Float64, seed::UInt64, outdir::String; 
                   alpha_override::Union{Nothing,Float64}=nothing, verbose::Bool=true)
    # Create config with optional alpha override
    cfg = if isnothing(alpha_override)
        Config(n=n, R=R, T=T)
    else
        Config(n=n, R=R, T=T, α=alpha_override)
    end
    
    target = 2n
    
    verbose && println("="^50)
    verbose && @printf("N3L Pure Gradient Flow Baseline\n")
    verbose && @printf("n=%d, target=%d points\n", n, target)
    verbose && @printf("R=%d trajectories, T=%.1fs, threads=%d\n", R, T, Threads.nthreads())
    verbose && @printf("Hyperparameters: α=%.1f, β=%.1f, γ=%.1f\n", cfg.α, cfg.β, cfg.γ)
    verbose && @printf("seed=%d\n", seed)
    
    triples = compute_triples(n)
    verbose && @printf("Collinear triples: %d\n", length(triples))
    verbose && println("-"^50)
    
    global_success = Ref(false)
    result_grid = Ref{Union{Nothing, BitVector}}(nothing)
    result_lock = ReentrantLock()
    stats = Dict(:success=>0, :timeout=>0)
    
    # Track violation statistics
    violation_lock = ReentrantLock()
    violation_history = Int[]
    
    start_time = time()
    
    Threads.@threads for id in 1:R
        global_success[] && continue
        
        traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(id) << 1))
        rng = Xoshiro(traj_seed)
        status, x_bin, viols = run_trajectory(cfg, triples, global_success, rng)
        
        lock(result_lock) do
            if status == SUCCESS && !global_success[]
                global_success[] = true
                result_grid[] = x_bin
                verbose && @printf("  ✓ Worker %d found solution (%.2fs)\n", id, time()-start_time)
            end
            
            key = status == SUCCESS ? :success : :timeout
            stats[key] += 1
        end
        
        lock(violation_lock) do
            push!(violation_history, viols)
        end
    end
    
    elapsed = time() - start_time
    
    # Statistics
    verbose && println("-"^50)
    verbose && println("STATISTICS:")
    verbose && @printf("  Success: %d/%d (%.1f%%)\n", stats[:success], R, 100*stats[:success]/R)
    
    if !isempty(violation_history)
        sorted_viols = sort(violation_history)
        verbose && println("  Violation distribution:")
        verbose && @printf("    Min: %d\n", minimum(sorted_viols))
        verbose && @printf("    Median: %d\n", sorted_viols[div(length(sorted_viols), 2)])
        verbose && @printf("    Max: %d\n", maximum(sorted_viols))
        
        close_calls = count(v -> v <= 5, sorted_viols)
        verbose && @printf("  Close calls (≤5 violations): %d (%.1f%%)\n", 
                          close_calls, 100*close_calls/length(sorted_viols))
    end
    
    verbose && println("-"^50)
    
    if global_success[]
        grid = reshape(result_grid[], (n, n))
        verbose && println("✓✓✓ SUCCESS ✓✓✓")
        verbose && @printf("Time: %.3fs\n", elapsed)
        verbose && println("Grid:")
        print_grid(grid)
        save_solution(n, grid, R, T, seed, outdir)
        return true, grid, elapsed, stats, seed
    else
        verbose && println("✗✗✗ FAILED ✗✗✗")
        verbose && @printf("Time: %.3fs\n", elapsed)
        
        if !isempty(violation_history)
            min_viol = minimum(violation_history)
            verbose && @printf("\nBest achieved: %d violations\n", min_viol)
            verbose && println("This is the baseline for UDE to improve upon.")
        end
        
        return false, nothing, elapsed, stats, seed
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
    alpha_override = args["alpha"]  # Will be nothing if not provided
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
                                                         verbose=verbose)
    
    if success
        println()
        alpha_str = isnothing(alpha_override) ? "" : " --alpha $(alpha_override)"
        println("Reproduce: julia --threads=12 main_baseline.jl $(n) --R $(R) --T $(T)$(alpha_str) --seed $(used_seed)")
    end
    
    return success ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end