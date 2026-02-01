#!/usr/bin/env julia
#=
N3L SciML Solver v2
===================
Scientific Machine Learning approach to No-Three-In-Line problem.
Gradient flow dynamics with parallel search, annealing, and local repair.

Usage:
    julia --threads=12 main.jl 11              # Single grid size
    julia --threads=12 main.jl 11 --R 500      # Custom trajectories
    julia --threads=12 main.jl 11 --seed 42    # Reproducible run
    julia --threads=12 main.jl --help          # Show help

Author: Sandy
Style: Chris Rackauckas / SciML conventions
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
    R::Int = 200
    T::Float64 = 30.0
    τ::Float64 = 0.5
    check_interval::Float64 = 0.05
    
    # Energy coefficients (smooth scaling)
    α_base::Float64 = 8.0
    α_scale::Float64 = 1.8               # Grows faster with n
    β::Float64 = 1.0
    γ::Float64 = 3.0 + 0.8 * n           # Gradual increase
    
    # Annealing settings
    anneal::Bool = true
    α_start_mult::Float64 = 0.05         # Start very low
    α_end_mult::Float64 = 3.0            # End high
end

function get_α(cfg::Config, t::Float64)
    α_target = cfg.α_base * (cfg.n / 6)^cfg.α_scale
    if !cfg.anneal
        return α_target
    end
    progress = clamp(t / cfg.T, 0.0, 1.0)
    # Smooth S-curve annealing
    smooth_progress = progress^2 * (3 - 2*progress)
    mult = cfg.α_start_mult + smooth_progress * (cfg.α_end_mult - cfg.α_start_mult)
    return α_target * mult
end

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_cli_args(args)
    s = ArgParseSettings(
        description = "N3L Solver v2 - No-Three-In-Line problem solver using SciML",
        version = "2.0.0",
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
            default = 500
        "--T"
            help = "Max integration time per trajectory"
            arg_type = Float64
            default = 30.0
        "--seed"
            help = "Random seed for reproducibility"
            arg_type = UInt64
        "--outdir"
            help = "Output directory base path"
            arg_type = String
            default = "solutions"
        "--no-anneal"
            help = "Disable annealing (use fixed α)"
            action = :store_true
        "--no-repair"
            help = "Disable local repair phase"
            action = :store_true
        "--quiet", "-q"
            help = "Suppress most output"
            action = :store_true
    end
    
    return parse_args(args, s)
end

# ============================================================================
# Collinear Triples with Adjacency List
# ============================================================================

struct TripleData
    triples::Vector{NTuple{3,Int}}
    point_to_triples::Vector{Vector{Int}}
end

function compute_triples(n::Int)
    triples = NTuple{3,Int}[]
    N = n * n
    point_to_triples = [Int[] for _ in 1:N]
    
    triple_idx = 0
    for x1 in 1:n, y1 in 1:n
        for x2 in 1:n, y2 in 1:n
            (x2, y2) <= (x1, y1) && continue
            for x3 in 1:n, y3 in 1:n
                (x3, y3) <= (x2, y2) && continue
                if x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0
                    i1 = (x1-1)*n + y1
                    i2 = (x2-1)*n + y2
                    i3 = (x3-1)*n + y3
                    push!(triples, (i1, i2, i3))
                    triple_idx += 1
                    push!(point_to_triples[i1], triple_idx)
                    push!(point_to_triples[i2], triple_idx)
                    push!(point_to_triples[i3], triple_idx)
                end
            end
        end
    end
    
    return TripleData(triples, point_to_triples)
end

# ============================================================================
# Energy and Gradient
# ============================================================================

function energy(x, td::TripleData, α, β, γ)
    E_col = 0.0
    @inbounds for (i, j, k) in td.triples
        E_col += x[i] * x[j] * x[k]
    end
    
    E_count = -β * sum(x)
    
    E_bin = 0.0
    @inbounds for i in eachindex(x)
        E_bin += x[i]^2 * (1 - x[i])^2
    end
    
    return α * E_col + E_count + γ * E_bin
end

function gradient!(g, x, td::TripleData, α, β, γ)
    @inbounds for i in eachindex(x)
        xi = x[i]
        g[i] = -β + γ * xi * (2 - 6*xi + 4*xi*xi)
    end
    
    @inbounds for (i, j, k) in td.triples
        g[i] += α * x[j] * x[k]
        g[j] += α * x[i] * x[k]
        g[k] += α * x[i] * x[j]
    end
    
    return g
end

# ============================================================================
# Validation
# ============================================================================

binarize(x, τ) = BitVector(x .>= τ)

function count_violations(x_bin, td::TripleData)
    count = 0
    @inbounds for (i, j, k) in td.triples
        count += x_bin[i] & x_bin[j] & x_bin[k]
    end
    return count
end

function find_violation_points(x_bin, td::TripleData)
    violation_points = Set{Int}()
    @inbounds for (i, j, k) in td.triples
        if x_bin[i] && x_bin[j] && x_bin[k]
            push!(violation_points, i, j, k)
        end
    end
    return violation_points
end

# ============================================================================
# Local Repair (greedy search when close to solution)
# ============================================================================

function local_repair!(x_bin::BitVector, td::TripleData, n::Int, rng; max_iters::Int=2000)
    target = 2n
    N = n * n
    
    for iter in 1:max_iters
        pts = sum(x_bin)
        viols = count_violations(x_bin, td)
        
        # Success?
        if pts == target && viols == 0
            return true, iter
        end
        
        viol_pts = find_violation_points(x_bin, td)
        
        # Strategy 1: Remove worst violation point (if at or above target)
        if !isempty(viol_pts) && pts >= target
            worst_pt = 0
            worst_count = 0
            for pt in viol_pts
                count = 0
                for tidx in td.point_to_triples[pt]
                    i, j, k = td.triples[tidx]
                    if x_bin[i] && x_bin[j] && x_bin[k]
                        count += 1
                    end
                end
                if count > worst_count
                    worst_count = count
                    worst_pt = pt
                end
            end
            
            if worst_pt > 0
                x_bin[worst_pt] = false
                continue
            end
        end
        
        # Strategy 2: Add safe point if under target
        if pts < target
            best_pt = 0
            best_viols = typemax(Int)
            
            # Shuffle to avoid bias
            candidates = shuffle(rng, collect(1:N))
            
            for pt in candidates
                x_bin[pt] && continue
                
                new_viols = 0
                for tidx in td.point_to_triples[pt]
                    i, j, k = td.triples[tidx]
                    others = (i == pt ? 0 : x_bin[i]) + 
                             (j == pt ? 0 : x_bin[j]) + 
                             (k == pt ? 0 : x_bin[k])
                    if others == 2
                        new_viols += 1
                    end
                end
                
                if new_viols < best_viols
                    best_viols = new_viols
                    best_pt = pt
                    new_viols == 0 && break  # Can't do better
                end
            end
            
            if best_pt > 0 && best_viols == 0
                x_bin[best_pt] = true
                continue
            end
        end
        
        # Strategy 3: Swap - remove violation point, add safe point
        if !isempty(viol_pts)
            viol_list = collect(viol_pts)
            remove_pt = viol_list[rand(rng, 1:length(viol_list))]
            x_bin[remove_pt] = false
            
            # Find safe point to add
            candidates = shuffle(rng, collect(1:N))
            for pt in candidates
                x_bin[pt] && continue
                
                safe = true
                for tidx in td.point_to_triples[pt]
                    i, j, k = td.triples[tidx]
                    others = (i == pt ? 0 : x_bin[i]) + 
                             (j == pt ? 0 : x_bin[j]) + 
                             (k == pt ? 0 : x_bin[k])
                    if others == 2
                        safe = false
                        break
                    end
                end
                
                if safe
                    x_bin[pt] = true
                    break
                end
            end
            continue
        end
        
        # No progress
        break
    end
    
    pts = sum(x_bin)
    viols = count_violations(x_bin, td)
    return pts == target && viols == 0, max_iters
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
# ODE System
# ============================================================================

function make_rhs(td::TripleData, cfg::Config)
    N = cfg.n^2
    g = zeros(N)
    
    function rhs!(dx, x, p, t)
        α = get_α(cfg, t)
        gradient!(g, x, td, α, cfg.β, cfg.γ)
        
        @inbounds for i in eachindex(x)
            dx[i] = -g[i]
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

@enum Status RUNNING SUCCESS REPAIRED STUCK TIMEOUT

function run_trajectory(cfg::Config, td::TripleData, global_success::Ref{Bool}, 
                        rng, traj_seed::UInt64; use_repair::Bool=true)
    N = cfg.n^2
    target = 2 * cfg.n
    
    x0 = biased_init(rng, N, target / N)
    
    status = Ref(RUNNING)
    best_viols = Ref(typemax(Int))
    best_x = Ref{Union{Nothing, Vector{Float64}}}(nothing)
    stall_count = Ref(0)
    last_viols = Ref(typemax(Int))
    perturb_count = Ref(0)
    
    rhs! = make_rhs(td, cfg)
    
    function check!(integrator)
        global_success[] && (terminate!(integrator); return)
        
        x = integrator.u
        t = integrator.t
        
        x_bin = topk_mask(x, target)
        viols = count_violations(x_bin, td)
        
        # Track best
        if viols < best_viols[]
            best_viols[] = viols
            best_x[] = copy(x)
        end
        
        # Direct success
        if viols == 0
            status[] = SUCCESS
            terminate!(integrator)
            return
        end
        
        # Track stalling
        if viols >= last_viols[]
            stall_count[] += 1
        else
            stall_count[] = 0
        end
        last_viols[] = viols
        
        # Violation-targeted perturbation when stuck
        if stall_count[] >= 8 && t > 0.2 * cfg.T && t < 0.85 * cfg.T && perturb_count[] < 5
            viol_pts = find_violation_points(x_bin, td)
            
            if !isempty(viol_pts)
                # Push violation points toward 0
                for pt in viol_pts
                    integrator.u[pt] *= 0.3 + 0.4 * rand(rng)
                end
                
                # Boost some random non-violation points
                non_viol = [i for i in 1:N if !(i in viol_pts) && integrator.u[i] < 0.6]
                for _ in 1:min(length(viol_pts), length(non_viol))
                    if !isempty(non_viol)
                        idx = rand(rng, 1:length(non_viol))
                        pt = non_viol[idx]
                        integrator.u[pt] = 0.6 + 0.3 * rand(rng)
                        deleteat!(non_viol, idx)
                    end
                end
            end
            
            stall_count[] = 0
            perturb_count[] += 1
        end
        
        # Truly stuck - too many perturbations without progress
        if perturb_count[] >= 5 && stall_count[] >= 10
            status[] = STUCK
            terminate!(integrator)
            return
        end
    end
    
    cb = PeriodicCallback(check!, cfg.check_interval; save_positions=(false, false))
    
    prob = ODEProblem(rhs!, x0, (0.0, cfg.T))
    sol = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-4, callback=cb,
                save_everystep=false, save_start=false, maxiters=1_000_000)
    
    # Use best solution
    x_final = best_x[] !== nothing ? best_x[] : sol.u[end]
    x_bin = topk_mask(x_final, target)
    viols = count_violations(x_bin, td)
    
    # Try local repair if close
    if status[] == RUNNING && viols > 0 && viols <= 15 && use_repair
        repair_rng = Xoshiro(splitmix64(traj_seed ⊻ UInt64(viols)))
        repaired, _ = local_repair!(x_bin, td, cfg.n, repair_rng; max_iters=3000)
        if repaired
            status[] = REPAIRED
            viols = 0
        end
    end
    
    # Final status
    if status[] == RUNNING
        status[] = (viols == 0) ? SUCCESS : TIMEOUT
    end
    
    return status[], x_bin, target, viols
end

# ============================================================================
# Parallel Search
# ============================================================================

function solve_n3l(n::Int, R::Int, T::Float64, seed::UInt64, outdir::String;
                   verbose::Bool=true, use_anneal::Bool=true, use_repair::Bool=true)
    cfg = Config(n=n, R=R, T=T, anneal=use_anneal)
    target = 2n
    
    verbose && println("="^60)
    verbose && @printf("N3L Solver v2: n=%d, target=%d points\n", n, target)
    verbose && @printf("R=%d trajectories, T=%.1fs, threads=%d\n", R, T, Threads.nthreads())
    verbose && @printf("seed=%d\n", seed)
    verbose && @printf("anneal=%s, repair=%s\n", use_anneal, use_repair)
    
    td = compute_triples(n)
    verbose && @printf("Collinear triples: %d\n", length(td.triples))
    if use_anneal
        verbose && @printf("α: %.2f → %.2f (annealed)\n", get_α(cfg, 0.0), get_α(cfg, cfg.T))
    else
        verbose && @printf("α: %.2f (fixed)\n", get_α(cfg, cfg.T/2))
    end
    verbose && @printf("γ: %.2f\n", cfg.γ)
    verbose && println("-"^60)
    
    global_success = Ref(false)
    result_grid = Ref{Union{Nothing, BitVector}}(nothing)
    result_status = Ref(TIMEOUT)
    result_lock = ReentrantLock()
    stats = Dict(:success=>0, :repaired=>0, :stuck=>0, :timeout=>0)
    best_viols_seen = Ref(typemax(Int))
    
    start_time = time()
    
    Threads.@threads for id in 1:R
        global_success[] && continue
        
        traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(id) << 1))
        rng = Xoshiro(traj_seed)
        status, x_bin, pts, viols = run_trajectory(cfg, td, global_success, rng, traj_seed;
                                                    use_repair=use_repair)
        
        lock(result_lock) do
            if viols < best_viols_seen[]
                best_viols_seen[] = viols
            end
            
            if (status == SUCCESS || status == REPAIRED) && !global_success[]
                global_success[] = true
                result_grid[] = x_bin
                result_status[] = status
                verbose && @printf("  ✓ Worker %d: %s (%.2fs)\n", id, status, time()-start_time)
            end
            
            key = status == SUCCESS ? :success :
                  status == REPAIRED ? :repaired :
                  status == STUCK ? :stuck : :timeout
            stats[key] += 1
            
            total = sum(values(stats))
            if verbose && total % max(1, R ÷ 10) == 0 && !global_success[]
                @printf("  [%d/%d] best_viols=%d\n", total, R, best_viols_seen[])
            end
        end
    end
    
    elapsed = time() - start_time
    
    verbose && println("-"^60)
    if global_success[]
        grid = reshape(result_grid[], (n, n))
        verbose && @printf("SUCCESS via %s!\n", result_status[])
        verbose && @printf("Time: %.3fs\n", elapsed)
        verbose && println("Grid:")
        print_grid(grid)
        save_solution(n, grid, R, T, seed, outdir)
        return true, grid, elapsed, stats, seed
    else
        verbose && @printf("FAILED (best_viols=%d)\n", best_viols_seen[])
        verbose && @printf("Stats: %s\n", stats)
        verbose && @printf("Time: %.3fs\n", elapsed)
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
    verbose = !quiet
    use_anneal = !args["no-anneal"]
    use_repair = !args["no-repair"]
    
    seed = if isnothing(args["seed"])
        rand(RandomDevice(), UInt64)
    else
        args["seed"]
    end
    
    success, grid, elapsed, stats, used_seed = solve_n3l(n, R, T, seed, outdir;
                                                          verbose=verbose,
                                                          use_anneal=use_anneal,
                                                          use_repair=use_repair)
    
    if success
        println()
        println("Reproduce: julia --threads=$(Threads.nthreads()) main.jl $(n) --R $(R) --T $(T) --seed $(used_seed)")
    end
    
    return success ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end