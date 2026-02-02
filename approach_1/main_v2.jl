#!/usr/bin/env julia
#=
N3L Sparse Solver - DIAGNOSTIC VERSION
======================================
Adds detailed logging to understand why n>10 fails
=#

using OrdinaryDiffEq
using DiffEqCallbacks
using SparseArrays
using LinearAlgebra
using Random
using Printf
using Dates
using ArgParse
using Statistics

# ============================================================================
# [Keep all the helper functions from before: splitmix64, topk_mask, Config, parse_cli_args]
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

Base.@kwdef struct Config
    n::Int
    R::Int = 100
    T::Float64 = 10.0
    τ::Float64 = 0.5
    check_interval::Float64 = 0.1
    # MUCH MORE AGGRESSIVE SCALING
    α::Float64 = begin
        if n <= 10
            10.0 * (n / 6)
        elseif n <= 15
            100.0 * (n / 10)  # 10x increase for n=11-15
        else
            200.0 * (n / 10)  # Even more for n>15
        end
    end
    β::Float64 = 1.0
    γ::Float64 = begin
        if n <= 10
            5.0
        elseif n <= 15
            50.0 * (n / 10)  # Much stronger binary pressure
        else
            100.0 * (n / 10)
        end
    end
end

function parse_cli_args(args)
    s = ArgParseSettings(
        description = "N3L Diagnostic Solver",
        version = "2.1.0",
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
            help = "Max integration time"
            arg_type = Float64
            default = 20.0
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
        "--debug"
            help = "Enable debug logging"
            action = :store_true
    end
    
    return parse_args(args, s)
end

# ============================================================================
# Sparse Structure (VERIFIED VERSION)
# ============================================================================

struct SparseTripleData
    point_to_pairs::Vector{Vector{Tuple{Int,Int}}}
    n_triples::Int
end

function build_sparse_triples(n::Int; verbose=false)
    N = n^2
    point_to_pairs = [Tuple{Int,Int}[] for _ in 1:N]
    
    # Collect all unique triplets first
    triplet_set = Set{Tuple{Int,Int,Int}}()
    
    for x1 in 1:n, y1 in 1:n
        for x2 in 1:n, y2 in 1:n
            (x2, y2) <= (x1, y1) && continue
            for x3 in 1:n, y3 in 1:n
                (x3, y3) <= (x2, y2) && continue
                
                if x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0
                    i = (x1-1)*n + y1
                    j = (x2-1)*n + y2
                    k = (x3-1)*n + y3
                    
                    triple = tuple(sort([i, j, k])...)
                    push!(triplet_set, triple)
                end
            end
        end
    end
    
    # Now build the point_to_pairs structure
    for (i, j, k) in triplet_set
        push!(point_to_pairs[i], (j, k))
        push!(point_to_pairs[j], (i, k))
        push!(point_to_pairs[k], (i, j))
    end
    
    if verbose
        println("  Total unique triplets: $(length(triplet_set))")
        pairs_per_point = [length(pairs) for pairs in point_to_pairs]
        println("  Pairs/point - min: $(minimum(pairs_per_point)), max: $(maximum(pairs_per_point)), avg: $(mean(pairs_per_point))")
    end
    
    return SparseTripleData(point_to_pairs, length(triplet_set))
end

# ============================================================================
# Energy and Gradient with VERIFICATION
# ============================================================================

function energy_sparse(x, triples::SparseTripleData, cfg::Config)
    # Count each triplet exactly once
    E_col = 0.0
    counted = Set{Tuple{Int,Int,Int}}()
    
    @inbounds for i in eachindex(triples.point_to_pairs)
        xi = x[i]
        for (j, k) in triples.point_to_pairs[i]
            triple = tuple(sort([i, j, k])...)
            if !(triple in counted)
                E_col += xi * x[j] * x[k]
                push!(counted, triple)
            end
        end
    end
    
    E_count = -cfg.β * sum(x)
    E_bin = sum(x .^ 2 .* (1 .- x) .^ 2)
    
    return cfg.α * E_col + E_count + cfg.γ * E_bin
end

function gradient_sparse!(g, x, triples::SparseTripleData, cfg::Config)
    # Binary + count terms
    @inbounds for i in eachindex(x)
        xi = x[i]
        g[i] = -cfg.β + cfg.γ * xi * (2 - 6*xi + 4*xi*xi)
    end
    
    # Collinearity gradient
    @inbounds for i in eachindex(triples.point_to_pairs)
        contrib = 0.0
        for (j, k) in triples.point_to_pairs[i]
            contrib += x[j] * x[k]
        end
        g[i] += cfg.α * contrib
    end
    
    return g
end

# ============================================================================
# Validation
# ============================================================================

binarize(x, τ) = BitVector(x .>= τ)

function count_violations_sparse(x_bin, triples::SparseTripleData)
    count = 0
    counted = Set{Tuple{Int,Int,Int}}()
    
    @inbounds for i in eachindex(triples.point_to_pairs)
        x_bin[i] || continue
        
        for (j, k) in triples.point_to_pairs[i]
            if x_bin[j] && x_bin[k]
                triple = tuple(sort([i, j, k])...)
                if !(triple in counted)
                    count += 1
                    push!(counted, triple)
                end
            end
        end
    end
    
    return count
end

# ============================================================================
# IMPROVED Initial Condition
# ============================================================================

function biased_init(rng, N, target_density, n)
    # More sophisticated initialization for larger n
    if n <= 10
        # Original approach
        a = max(0.5, 2.0 * target_density)
        b = max(0.5, 2.0 * (1.0 - target_density))
        
        x0 = Vector{Float64}(undef, N)
        @inbounds for i in 1:N
            u = rand(rng)^(1/a)
            v = rand(rng)^(1/b)
            x0[i] = u / (u + v)
        end
    else
        # For n>10: Start with more structure
        # Place points on a jittered grid pattern
        x0 = zeros(Float64, N)
        target_count = 2 * n
        
        # Select random positions with some spatial spreading
        positions = shuffle(rng, 1:N)
        for i in 1:target_count
            x0[positions[i]] = 0.7 + 0.25 * rand(rng)  # Start high
        end
        
        # Add noise to remaining positions
        for i in (target_count+1):N
            x0[positions[i]] = 0.1 * rand(rng)  # Start low
        end
    end
    
    return x0
end

# ============================================================================
# ODE System with DIAGNOSTICS
# ============================================================================

function make_rhs_sparse(triples::SparseTripleData, cfg::Config; debug=false)
    g = zeros(cfg.n^2)
    
    function rhs!(dx, x, p, t)
        gradient_sparse!(g, x, triples, cfg)
        
        if debug && t > 0 && mod(t, 1.0) < 0.01
            E = energy_sparse(x, triples, cfg)
            grad_norm = norm(g)
            println("  [t=$(round(t,digits=2))] E=$(round(E,digits=2)), |∇E|=$(round(grad_norm,digits=2)), sum(x)=$(round(sum(x),digits=1))")
        end
        
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
# Single Trajectory with BETTER DIAGNOSTICS
# ============================================================================

@enum Status RUNNING SUCCESS STUCK TIMEOUT

function run_trajectory(cfg::Config, triples::SparseTripleData, 
                       global_success::Ref{Bool}, rng, traj_seed::UInt64; debug=false)
    N = cfg.n^2
    target = 2 * cfg.n
    
    # Improved initialization
    x0 = biased_init(rng, N, target / N, cfg.n)
    
    status = Ref(RUNNING)
    last_energy = Ref(Inf)
    stall_count = Ref(0)
    min_violations = Ref(1000000)
    
    rhs! = make_rhs_sparse(triples, cfg; debug=debug)
    
    function check!(integrator)
        global_success[] && (terminate!(integrator); return)
        
        x = integrator.u
        t = integrator.t
        
        # Evaluate
        x_bin = topk_mask(x, target)
        viols = count_violations_sparse(x_bin, triples)
        
        # Track best violations seen
        if viols < min_violations[]
            min_violations[] = viols
            if debug
                println("  [t=$(round(t,digits=2))] New best: $(viols) violations")
            end
        end
        
        # Success?
        if viols == 0
            status[] = SUCCESS
            terminate!(integrator)
            return
        end
        
        # Energy tracking
        E = energy_sparse(x, triples, cfg)
        if E > last_energy[] - 1e-8
            stall_count[] += 1
        else
            stall_count[] = 0
        end
        last_energy[] = E
        
        # More aggressive perturbation for larger n
        perturb_threshold = cfg.n <= 10 ? 5 : 3  # Perturb sooner for large n
        
        if viols > 0 && stall_count[] >= perturb_threshold && t > 0.3 * cfg.T
            # Stronger perturbation for larger n
            perturb_scale = cfg.n <= 10 ? 0.25 : 0.4
            u_modified = false
            
            @inbounds for i in eachindex(integrator.u)
                if integrator.u[i] > 0.05 && integrator.u[i] < 0.95
                    integrator.u[i] += perturb_scale * (2.0 * rand(rng) - 1.0)
                    integrator.u[i] = clamp(integrator.u[i], 0.0, 1.0)
                    u_modified = true
                end
            end
            
            if u_modified
                stall_count[] = 0
                last_energy[] = Inf
                if debug
                    println("  [t=$(round(t,digits=2))] Perturbed! Continuing search...")
                end
                return
            else
                status[] = STUCK
                terminate!(integrator)
                return
            end
        end
    end
    
    cb = PeriodicCallback(check!, cfg.check_interval; save_positions=(false, false))
    
    # Solve
    prob = ODEProblem(rhs!, x0, (0.0, cfg.T))
    sol = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-4, callback=cb,
                save_everystep=false, save_start=false, maxiters=1_000_000)
    
    # Final check
    x_final = sol.u[end]
    x_bin = topk_mask(x_final, target)
    viols = count_violations_sparse(x_bin, triples)
    pts = target
    
    if status[] == RUNNING
        status[] = (viols == 0) ? SUCCESS : TIMEOUT
    end
    
    if debug
        println("  Final: $(viols) violations (best seen: $(min_violations[]))")
    end
    
    return status[], x_bin, pts, viols, min_violations[]
end

# ============================================================================
# Parallel Search with STATISTICS
# ============================================================================

function solve_n3l(n::Int, R::Int, T::Float64, seed::UInt64, outdir::String; 
                  verbose::Bool=true, debug::Bool=false)
    cfg = Config(n=n, R=R, T=T)
    target = 2n
    
    verbose && println("="^70)
    verbose && @printf("N3L DIAGNOSTIC Solver: n=%d, target=%d points\n", n, target)
    verbose && @printf("R=%d trajectories, T=%.1fs, threads=%d\n", R, T, Threads.nthreads())
    verbose && @printf("Hyperparameters: α=%.1f, β=%.1f, γ=%.1f\n", cfg.α, cfg.β, cfg.γ)
    verbose && @printf("seed=%d\n", seed)
    verbose && println("-"^70)
    
    # Build sparse structure
    verbose && print("Building sparse structure... ")
    build_time = @elapsed triples = build_sparse_triples(n; verbose=verbose)
    verbose && @printf("done (%.3fs)\n", build_time)
    
    # Shared state
    global_success = Ref(false)
    result_grid = Ref{Union{Nothing, BitVector}}(nothing)
    result_lock = ReentrantLock()
    stats = Dict(:success=>0, :stuck=>0, :timeout=>0)
    violation_history = Int[]
    
    start_time = time()
    
    # Run first trajectory with debug if requested
    if debug && R > 0
        println("\n" * "="^70)
        println("DEBUG: Running first trajectory with detailed logging")
        println("="^70)
        rng = Xoshiro(splitmix64(seed ⊻ UInt64(n) ⊻ UInt64(1)))
        status, x_bin, pts, viols, min_viols = run_trajectory(cfg, triples, global_success, rng, splitmix64(seed); debug=true)
        println("="^70)
        println("First trajectory: status=$status, violations=$viols")
        println("="^70 * "\n")
        
        if status == SUCCESS
            global_success[] = true
            result_grid[] = x_bin
        end
        
        R = R - 1  # One less to run
    end
    
    # Parallel search
    violation_lock = ReentrantLock()
    
    Threads.@threads for id in 1:(R)
        global_success[] && continue
        
        traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(id + (debug ? 1 : 0)) << 1))
        rng = Xoshiro(traj_seed)
        status, x_bin, pts, viols, min_viols = run_trajectory(cfg, triples, global_success, rng, traj_seed; debug=false)
        
        lock(result_lock) do
            if status == SUCCESS && !global_success[]
                global_success[] = true
                result_grid[] = x_bin
                verbose && @printf("  ✓ Worker %d found solution (%.2fs)\n", id, time()-start_time)
            end
            
            key = status == SUCCESS ? :success : 
                  status == STUCK ? :stuck : :timeout
            stats[key] += 1
        end
        
        lock(violation_lock) do
            push!(violation_history, min_viols)
        end
    end
    
    elapsed = time() - start_time
    
    # Detailed statistics
    verbose && println("-"^70)
    verbose && println("STATISTICS:")
    verbose && @printf("  Success: %d/%d (%.1f%%)\n", stats[:success], R, 100*stats[:success]/R)
    verbose && @printf("  Stuck: %d, Timeout: %d\n", stats[:stuck], stats[:timeout])
    
    if !isempty(violation_history)
        sorted_viols = sort(violation_history)
        verbose && println("  Best violations achieved:")
        verbose && @printf("    Min: %d\n", minimum(sorted_viols))
        verbose && @printf("    25th percentile: %d\n", sorted_viols[max(1, div(length(sorted_viols), 4))])
        verbose && @printf("    Median: %d\n", sorted_viols[max(1, div(length(sorted_viols), 2))])
        verbose && @printf("    75th percentile: %d\n", sorted_viols[max(1, 3*div(length(sorted_viols), 4))])
        verbose && @printf("    Max: %d\n", maximum(sorted_viols))
        
        # Critical metric: how close are we getting?
        close_calls = count(v -> v <= 5, sorted_viols)
        very_close = count(v -> v <= 2, sorted_viols)
        verbose && @printf("  Close calls (≤5 violations): %d (%.1f%%)\n", close_calls, 100*close_calls/length(sorted_viols))
        verbose && @printf("  Very close (≤2 violations): %d (%.1f%%)\n", very_close, 100*very_close/length(sorted_viols))
    end
    
    verbose && println("-"^70)
    
    if global_success[]
        grid = reshape(result_grid[], (n, n))
        verbose && println("✓✓✓ SUCCESS! ✓✓✓")
        verbose && @printf("Time: %.3fs\n", elapsed)
        verbose && println("Grid:")
        print_grid(grid)
        save_solution(n, grid, R, T, seed, outdir)
        return true, grid, elapsed, stats, seed
    else
        verbose && println("✗✗✗ FAILED ✗✗✗")
        verbose && @printf("Time: %.3fs\n", elapsed)
        verbose && println("\nRECOMMENDATIONS:")
        
        if !isempty(violation_history)
            min_viol = minimum(violation_history)
            if min_viol <= 10
                verbose && println("  → Getting close! Try:")
                verbose && println("     • Increase R to $(2*R) or $(5*R)")
                verbose && println("     • Increase T to $(2*T)")
                verbose && println("     • Increase α to $(2*cfg.α)")
            elseif min_viol <= 50
                verbose && println("  → Making progress but stuck. Try:")
                verbose && println("     • Much higher R ($(5*R)+)")
                verbose && println("     • Stronger penalties: α=$(3*cfg.α), γ=$(3*cfg.γ)")
            else
                verbose && println("  → Not converging. This size may need UDE (Phase 2)")
                verbose && println("     • Current approach likely insufficient for n=$(n)")
            end
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
    outdir = args["outdir"]
    quiet = args["quiet"]
    debug = args["debug"]
    verbose = !quiet
    
    seed = if isnothing(args["seed"])
        rand(RandomDevice(), UInt64)
    else
        args["seed"]
    end
    
    success, grid, elapsed, stats, used_seed = solve_n3l(n, R, T, seed, outdir; verbose=verbose, debug=debug)
    
    if success
        println()
        println("Reproduce: julia --threads=12 main_sparse.jl $(n) --R $(R) --T $(T) --seed $(used_seed)")
    end
    
    return success ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end