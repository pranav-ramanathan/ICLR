#!/usr/bin/env julia
#=
N3L Pure Gradient Flow — Metal GPU Accelerated (EnsembleGPUKernel)
===================================================================
Uses DiffEqGPU's EnsembleGPUKernel with Metal backend.
All trajectories run to completion on GPU, then validated on CPU.

Key constraints for Metal:
  - EnsembleGPUKernel (NOT EnsembleGPUArray — broken on Metal)
  - Out-of-place ODE: f(u, p, t) → SVector
  - Float32 everywhere (Metal has limited Float64)
  - No heap allocation in RHS
  - Parameters as flat SVector
=#

using Metal
using DiffEqGPU
using OrdinaryDiffEq
using StaticArrays
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
# Top-k Mask (CPU only — uses sorting)
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
    α::Float64 = n <= 10 ? 10.0 * (n / 6) : 40.0
    β::Float64 = 1.0
    γ::Float64 = n <= 10 ? 5.0 : 15.0
end

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_cli_args(args)
    s = ArgParseSettings(
        description = "N3L Gradient Flow — Metal GPU (EnsembleGPUKernel)",
        version = "2.0.0",
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
            default = 5000
        "--T"
            help = "Max integration time"
            arg_type = Float64
            default = 10.0
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
        "--batch-size"
            help = "Trajectories per GPU batch"
            arg_type = Int
            default = 1024
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
# Validation (CPU)
# ============================================================================

function count_violations(x_bin, triples)
    count = 0
    @inbounds for (i, j, k) in triples
        count += x_bin[i] & x_bin[j] & x_bin[k]
    end
    return count
end

# ============================================================================
# Initial Condition (CPU, then transferred to GPU via prob_func)
# ============================================================================

function biased_init_svector(::Val{N}, rng, target_density) where {N}
    a = max(0.5f0, 2.0f0 * Float32(target_density))
    b = max(0.5f0, 2.0f0 * (1.0f0 - Float32(target_density)))
    
    vals = ntuple(N) do _
        u = Float32(rand(rng))^(1.0f0/a)
        v = Float32(rand(rng))^(1.0f0/b)
        u / (u + v)
    end
    return SVector{N, Float32}(vals)
end

# ============================================================================
# Build ODE function and problem using @generated for the RHS
# ============================================================================
#
# The key trick: we encode the triple indices and config into the *parameter
# vector* as a flat SVector{3M+3, Float32}:
#   p[1]    = α
#   p[2]    = β
#   p[3]    = γ
#   p[4]    ... p[3+M]     = triple_i indices (as Float32)
#   p[3+M+1] ... p[3+2M]  = triple_j indices
#   p[3+2M+1]... p[3+3M]  = triple_k indices
#
# The RHS reads these at runtime. Since SVector indexing is O(1) and
# stack-allocated, this compiles cleanly to a GPU kernel.
#

"""
Build the parameter SVector encoding α, β, γ and all triples.
Returns (p::SVector, M::Int) where M is the number of triples.
"""
function build_params(cfg::Config, triples)
    M = length(triples)
    α = Float32(cfg.α)
    β = Float32(cfg.β)
    γ = Float32(cfg.γ)
    
    # Build flat vector: [α, β, γ, ti..., tj..., tk...]
    vals = Vector{Float32}(undef, 3 + 3M)
    vals[1] = α
    vals[2] = β
    vals[3] = γ
    for (idx, (i, j, k)) in enumerate(triples)
        vals[3 + idx]      = Float32(i)
        vals[3 + M + idx]  = Float32(j)
        vals[3 + 2M + idx] = Float32(k)
    end
    
    return SVector{3 + 3M, Float32}(vals...), M
end

"""
GPU-compatible RHS: f(u, p, t) → SVector{N, Float32}

Encodes negative gradient flow with box constraints.
All parameters read from the SVector `p`.
M (number of triples) is a Val parameter for type stability.
"""
function make_rhs(::Val{N}, ::Val{M}) where {N, M}
    function rhs(u::SVector{N, Float32}, p, t) where {}
        α = p[1]
        β = p[2]
        γ = p[3]
        
        # Compute gradient: binary regularization part
        g = MVector{N, Float32}(undef)
        @inbounds for i in 1:N
            xi = u[i]
            g[i] = -β + γ * xi * (2.0f0 - 6.0f0*xi + 4.0f0*xi*xi)
        end
        
        # Add collinearity penalty from triples
        @inbounds for idx in 1:M
            # Read triple indices from parameter vector
            ti = unsafe_trunc(Int32, p[3 + idx])
            tj = unsafe_trunc(Int32, p[3 + M + idx])
            tk = unsafe_trunc(Int32, p[3 + 2*M + idx])
            
            g[ti] += α * u[tj] * u[tk]
            g[tj] += α * u[ti] * u[tk]
            g[tk] += α * u[ti] * u[tj]
        end
        
        # Negative gradient with box constraints
        dx = MVector{N, Float32}(undef)
        @inbounds for i in 1:N
            di = -g[i]
            if u[i] <= 0.0f0 && di < 0.0f0
                di = 0.0f0
            elseif u[i] >= 1.0f0 && di > 0.0f0
                di = 0.0f0
            end
            dx[i] = di
        end
        
        return SVector{N, Float32}(dx)
    end
    return rhs
end

# ============================================================================
# GPU Solve
# ============================================================================

function solve_metal(n::Int, R::Int, T::Float64, seed::UInt64, outdir::String;
                     alpha_override::Union{Nothing,Float64}=nothing,
                     gamma_override::Union{Nothing,Float64}=nothing,
                     verbose::Bool=true,
                     batch_size::Int=1024)
    # Build config
    cfg = if !isnothing(alpha_override) && !isnothing(gamma_override)
        Config(n=n, R=R, T=T, α=alpha_override, γ=gamma_override)
    elseif !isnothing(alpha_override)
        Config(n=n, R=R, T=T, α=alpha_override)
    elseif !isnothing(gamma_override)
        Config(n=n, R=R, T=T, γ=gamma_override)
    else
        Config(n=n, R=R, T=T)
    end
    
    N = n^2
    target = 2n
    
    verbose && println("=" ^ 60)
    verbose && println("N3L Gradient Flow — Metal GPU (EnsembleGPUKernel)")
    verbose && println("=" ^ 60)
    verbose && @printf("n=%d, target=%d | α=%.1f, β=%.1f, γ=%.1f\n", n, target, cfg.α, cfg.β, cfg.γ)
    verbose && @printf("Max: %d traj, T=%.1fs, batch=%d\n", R, T, batch_size)
    verbose && @printf("Seed: %d\n", seed)
    verbose && println("-" ^ 60)
    
    # Precompute triples
    triples = compute_triples(n)
    M = length(triples)
    verbose && @printf("Triples: %d | State dim: %d\n", M, N)
    
    # Build parameter vector
    p_sv, _ = build_params(cfg, triples)
    verbose && @printf("Parameter SVector: %d elements\n", length(p_sv))
    verbose && println("-" ^ 60)
    
    # Build the RHS function specialized on N and M
    rhs = make_rhs(Val(N), Val(M))
    
    # Create base ODE problem (out-of-place for EnsembleGPUKernel)
    rng_base = Xoshiro(splitmix64(seed ⊻ UInt64(n) ⊻ UInt64(1) << 1))
    u0_base = biased_init_svector(Val(N), rng_base, target / N)
    tspan = (0.0f0, Float32(T))
    
    prob = ODEProblem{false}(rhs, u0_base, tspan, p_sv)
    
    # Warmup
    verbose && println("Warming up Metal GPU (compiling kernel)...")
    warmup_prob_func = let seed=seed, n=n, N=N, target=target, p_sv=p_sv
        (prob, i, repeat) -> begin
            traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(i) << 1))
            rng = Xoshiro(traj_seed)
            u0_new = biased_init_svector(Val(N), rng, target / N)
            remake(prob, u0 = u0_new, p = p_sv)
        end
    end
    try
        warmup_ensemble = EnsembleProblem(prob, prob_func=warmup_prob_func, safetycopy=false)
        solve(warmup_ensemble, GPUTsit5(), EnsembleGPUKernel(Metal.MetalBackend()),
              trajectories=2, adaptive=true, dt=0.01f0,
              abstol=1.0f-4, reltol=1.0f-3, save_everystep=false)
        verbose && println("  ✓ Metal warmup OK")
    catch e
        verbose && println("  ⚠ Metal warmup failed: $e")
        verbose && println("  Continuing anyway.")
    end
    verbose && println("-" ^ 60)
    
    # Batched GPU solve with CPU validation
    start_time = time()
    total_tried = 0
    best_viols = typemax(Int)
    solution_found = false
    solution_grid = nothing
    solution_batch_id = 0
    solution_traj_id = 0
    violation_histogram = Dict{Int,Int}()
    
    n_batches = cld(R, batch_size)
    
    for batch in 1:n_batches
        remaining = R - total_tried
        batch_n = min(batch_size, remaining)
        batch_offset = (batch - 1) * batch_size
        
        verbose && @printf("[Batch %d/%d] Launching %d trajectories on GPU...\n", 
                          batch, n_batches, batch_n)
        
        # Build prob_func for this batch
        prob_func = let seed=seed, n=n, N=N, target=target, p_sv=p_sv, offset=batch_offset
            (prob, i, repeat) -> begin
                id = offset + i
                traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(id) << 1))
                rng = Xoshiro(traj_seed)
                u0_new = biased_init_svector(Val(N), rng, target / N)
                remake(prob, u0 = u0_new, p = p_sv)
            end
        end
        
        ensemble_prob = EnsembleProblem(prob, prob_func=prob_func, safetycopy=false)
        
        local sol
        try
            sol = solve(ensemble_prob, GPUTsit5(), EnsembleGPUKernel(Metal.MetalBackend()),
                       trajectories=batch_n, adaptive=true, dt=0.01f0,
                       abstol=1.0f-4, reltol=1.0f-3, save_everystep=false)
        catch e
            verbose && @printf("  ⚠ GPU batch failed: %s\n", string(e)[1:min(200,end)])
            verbose && println("  Skipping batch.")
            total_tried += batch_n
            continue
        end
        
        # CPU validation of all trajectories in this batch
        batch_best = typemax(Int)
        for i in 1:batch_n
            x_final = Float64.(sol[i].u[end])  # Back to Float64 for validation
            x_bin = topk_mask(x_final, target)
            viols = count_violations(x_bin, triples)
            
            violation_histogram[viols] = get(violation_histogram, viols, 0) + 1
            
            if viols < batch_best
                batch_best = viols
            end
            
            if viols == 0 && !solution_found
                solution_found = true
                solution_grid = reshape(x_bin, (n, n))
                solution_batch_id = batch
                solution_traj_id = batch_offset + i
                elapsed = time() - start_time
                verbose && @printf("  🎉 SOLUTION! batch=%d, traj=%d, time=%.2fs\n", 
                                  batch, solution_traj_id, elapsed)
                break
            end
        end
        
        total_tried += batch_n
        
        if batch_best < best_viols
            best_viols = batch_best
            verbose && @printf("  ★ Best violations: %d\n", best_viols)
        end
        
        elapsed = time() - start_time
        rate = total_tried / elapsed
        verbose && @printf("  Batch done: best=%d, tried=%d, %.1f traj/s\n",
                          best_viols, total_tried, rate)
        
        if solution_found
            break
        end
    end
    
    elapsed = time() - start_time
    
    verbose && println("-" ^ 60)
    
    if solution_found
        verbose && println("✓✓✓ SUCCESS ✓✓✓")
        verbose && @printf("Time: %.2fs | Tried: %d/%d (%.1f%%) | Rate: %.1f/s\n",
                          elapsed, total_tried, R,
                          100 * total_tried / R, total_tried / elapsed)
        
        if !isempty(violation_histogram)
            verbose && println("\nViolation distribution:")
            for v in sort(collect(keys(violation_histogram)))
                count = violation_histogram[v]
                pct = 100 * count / total_tried
                bar = repeat("█", min(40, round(Int, pct/2)))
                verbose && @printf("  %2d violations: %5d (%5.2f%%) %s\n", v, count, pct, bar)
            end
        end
        
        verbose && println("\nSolution:")
        print_grid(solution_grid)
        save_solution(n, solution_grid, solution_traj_id, R, T, seed, cfg.α, cfg.γ, outdir)
        
        return true, solution_grid, elapsed, Dict(:success=>1, :tried=>total_tried), seed
    else
        verbose && println("✗✗✗ NO SOLUTION FOUND ✗✗✗")
        verbose && @printf("Time: %.2fs | Tried: %d | Best: %d viols | Rate: %.1f/s\n",
                          elapsed, total_tried, best_viols, total_tried / elapsed)
        
        if !isempty(violation_histogram)
            verbose && println("\nViolation distribution:")
            for v in sort(collect(keys(violation_histogram)))
                count = violation_histogram[v]
                pct = 100 * count / total_tried
                bar = repeat("█", min(40, round(Int, pct/2)))
                verbose && @printf("  %2d violations: %5d (%5.2f%%) %s\n", v, count, pct, bar)
            end
        end
        
        return false, nothing, elapsed, Dict(:success=>0, :tried=>total_tried), seed
    end
end

# ============================================================================
# Utilities
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
            print(grid[i,j] ? "● " : "· ")
        end
        println()
    end
end

function save_solution(n, grid, traj_id, R, T, seed, α, γ, outdir)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "$(outdir)/$(n)"
    mkpath(dir)
    filename = "$(dir)/sol_metal_$(timestamp)_traj$(traj_id).txt"
    
    open(filename, "w") do io
        println(io, "# n=$(n)")
        println(io, "# target=$(2n)")
        println(io, "# trajectory_id=$(traj_id)")
        println(io, "# R=$(R)")
        println(io, "# T=$(T)")
        println(io, "# seed=$(seed)")
        println(io, "# α=$(α), β=1.0, γ=$(γ)")
        println(io, "# solver=Metal GPU (EnsembleGPUKernel + GPUTsit5)")
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
# Metal Check
# ============================================================================

function check_metal()
    println("Checking Metal GPU...")
    try
        dev = Metal.current_device()
        println("  Device: $(dev)")
        return true
    catch e
        println("  ✗ Metal not available: $e")
        println("  This script requires an Apple Silicon Mac with Metal support.")
        return false
    end
end

# ============================================================================
# Main
# ============================================================================

function main()
    if !check_metal()
        return 1
    end
    
    args = parse_cli_args(ARGS)
    
    n = args["n"]
    R = args["R"]
    T = args["T"]
    alpha_override = args["alpha"]
    gamma_override = args["gamma"]
    outdir = args["outdir"]
    quiet = args["quiet"]
    verbose = !quiet
    batch_size = args["batch-size"]
    
    seed = if isnothing(args["seed"])
        rand(RandomDevice(), UInt64)
    else
        args["seed"]
    end
    
    success, grid, elapsed, stats, used_seed = solve_metal(n, R, T, seed, outdir;
                                                           alpha_override=alpha_override,
                                                           gamma_override=gamma_override,
                                                           verbose=verbose,
                                                           batch_size=batch_size)
    
    if success
        println()
        alpha_str = isnothing(alpha_override) ? "" : " --alpha $(alpha_override)"
        gamma_str = isnothing(gamma_override) ? "" : " --gamma $(gamma_override)"
        println("Reproduce exact solution:")
        println("julia --project=. $(PROGRAM_FILE) $(n) --R $(R) --T $(T)$(alpha_str)$(gamma_str) --seed $(used_seed)")
    end
    
    return success ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
