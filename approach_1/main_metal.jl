#!/usr/bin/env julia
#=
N3L Pure Gradient Flow — Metal GPU Accelerated (Custom Kernel)
===============================================================
Uses a custom KernelAbstractions @kernel implementing RK4 directly on Metal.
Each GPU thread handles one independent ODE trajectory.
Works for ALL board sizes (no SVector size limit).

Architecture:
  - State: N×R MtlArray matrix (each column = one trajectory)
  - Triples: three Int32 MtlArrays (ti, tj, tk) read by all threads
  - Kernel: fixed-step RK4 with box constraints and gradient flow
  - Validation: CPU-side topk_mask + violation check after GPU solve
=#

using Metal
using KernelAbstractions
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
# Top-k Mask Helper (CPU)
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
    T::Float32 = 10.0f0
    dt::Float32 = 0.005f0
    α::Float32 = n <= 10 ? Float32(10.0 * (n / 6)) : 40.0f0
    β::Float32 = 1.0f0
    γ::Float32 = n <= 10 ? 5.0f0 : 15.0f0
end

# ============================================================================
# CLI Argument Parsing
# ============================================================================

function parse_cli_args(args)
    s = ArgParseSettings(
        description = "N3L Gradient Flow — Metal GPU (Custom Kernel)",
        version = "2.0.0",
        add_version = true
    )

    @add_arg_table! s begin
        "n"
            help = "Board size (n x n grid)"
            arg_type = Int
            required = true
        "--R"
            help = "Number of trajectories per batch"
            arg_type = Int
            default = 5000
        "--T"
            help = "Integration time"
            arg_type = Float64
            default = 10.0
        "--dt"
            help = "RK4 timestep"
            arg_type = Float64
            default = 0.005
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
        "--max-batches"
            help = "Maximum number of batches (0 = unlimited until R)"
            arg_type = Int
            default = 0
    end

    return parse_args(args, s)
end

# ============================================================================
# Precompute Collinear Triples
# ============================================================================

function compute_triples(n::Int)
    triples_i = Int32[]
    triples_j = Int32[]
    triples_k = Int32[]

    for x1 in 1:n, y1 in 1:n
        for x2 in 1:n, y2 in 1:n
            (x2, y2) <= (x1, y1) && continue
            for x3 in 1:n, y3 in 1:n
                (x3, y3) <= (x2, y2) && continue
                if x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) == 0
                    push!(triples_i, Int32((x1-1)*n + y1))
                    push!(triples_j, Int32((x2-1)*n + y2))
                    push!(triples_k, Int32((x3-1)*n + y3))
                end
            end
        end
    end

    return triples_i, triples_j, triples_k
end

# ============================================================================
# GPU Kernel: RK4 Gradient Flow Integrator
# ============================================================================
# Each thread integrates one trajectory from t=0 to t=T using fixed-step RK4.
# The negative gradient of the energy function drives the flow.
# Box constraints [0,1] are applied after each RK4 step.
#
# Memory layout:
#   state[i, traj]  — state variable i for trajectory traj
#   ti[m], tj[m], tk[m] — m-th collinear triple indices
#
# The gradient for variable i is:
#   g[i] = -β + γ * x[i] * (2 - 6*x[i] + 4*x[i]^2)
#          + α * Σ_{triples containing i} (product of other two)

@kernel function rk4_gradient_flow_kernel!(
    state,          # N × R matrix (Float32) — state, modified in-place
    @Const(ti),     # M vector (Int32) — triple index i
    @Const(tj),     # M vector (Int32) — triple index j
    @Const(tk),     # M vector (Int32) — triple index k
    α::Float32,
    β::Float32,
    γ::Float32,
    dt::Float32,
    nsteps::Int32,
    N::Int32,       # state dimension (n^2)
    M::Int32        # number of triples
)
    traj = @index(Global, Linear)

    # Temporary arrays in registers (KernelAbstractions private memory)
    # For large N, these will spill to device memory — that's fine
    k1 = @private Float32 (512,)
    k2 = @private Float32 (512,)
    k3 = @private Float32 (512,)
    k4 = @private Float32 (512,)
    xtmp = @private Float32 (512,)

    # Time integration loop
    for step in Int32(1):nsteps
        # ---- Compute k1 = f(x) ----
        # Base gradient (per-variable terms)
        for i in Int32(1):N
            xi = state[i, traj]
            @inbounds k1[i] = β - γ * xi * (2.0f0 - 6.0f0*xi + 4.0f0*xi*xi)
        end
        # Triple contributions
        for m in Int32(1):M
            @inbounds begin
                ii = ti[m]; jj = tj[m]; kk = tk[m]
                xi = state[ii, traj]; xj = state[jj, traj]; xk = state[kk, traj]
                k1[ii] -= α * xj * xk
                k1[jj] -= α * xi * xk
                k1[kk] -= α * xi * xj
            end
        end

        # ---- Compute k2 = f(x + dt/2 * k1) ----
        half_dt = dt * 0.5f0
        for i in Int32(1):N
            @inbounds xtmp[i] = state[i, traj] + half_dt * k1[i]
        end
        for i in Int32(1):N
            xi = @inbounds xtmp[i]
            @inbounds k2[i] = β - γ * xi * (2.0f0 - 6.0f0*xi + 4.0f0*xi*xi)
        end
        for m in Int32(1):M
            @inbounds begin
                ii = ti[m]; jj = tj[m]; kk = tk[m]
                xi = xtmp[ii]; xj = xtmp[jj]; xk = xtmp[kk]
                k2[ii] -= α * xj * xk
                k2[jj] -= α * xi * xk
                k2[kk] -= α * xi * xj
            end
        end

        # ---- Compute k3 = f(x + dt/2 * k2) ----
        for i in Int32(1):N
            @inbounds xtmp[i] = state[i, traj] + half_dt * k2[i]
        end
        for i in Int32(1):N
            xi = @inbounds xtmp[i]
            @inbounds k3[i] = β - γ * xi * (2.0f0 - 6.0f0*xi + 4.0f0*xi*xi)
        end
        for m in Int32(1):M
            @inbounds begin
                ii = ti[m]; jj = tj[m]; kk = tk[m]
                xi = xtmp[ii]; xj = xtmp[jj]; xk = xtmp[kk]
                k3[ii] -= α * xj * xk
                k3[jj] -= α * xi * xk
                k3[kk] -= α * xi * xj
            end
        end

        # ---- Compute k4 = f(x + dt * k3) ----
        for i in Int32(1):N
            @inbounds xtmp[i] = state[i, traj] + dt * k3[i]
        end
        for i in Int32(1):N
            xi = @inbounds xtmp[i]
            @inbounds k4[i] = β - γ * xi * (2.0f0 - 6.0f0*xi + 4.0f0*xi*xi)
        end
        for m in Int32(1):M
            @inbounds begin
                ii = ti[m]; jj = tj[m]; kk = tk[m]
                xi = xtmp[ii]; xj = xtmp[jj]; xk = xtmp[kk]
                k4[ii] -= α * xj * xk
                k4[jj] -= α * xi * xk
                k4[kk] -= α * xi * xj
            end
        end

        # ---- RK4 update + box constraint [0, 1] ----
        sixth_dt = dt / 6.0f0
        for i in Int32(1):N
            @inbounds begin
                xnew = state[i, traj] + sixth_dt * (k1[i] + 2.0f0*k2[i] + 2.0f0*k3[i] + k4[i])
                # Clamp to [0, 1]
                state[i, traj] = min(1.0f0, max(0.0f0, xnew))
            end
        end
    end
end

# ============================================================================
# Biased Initial Conditions (CPU, then transfer to GPU)
# ============================================================================

function generate_initial_conditions(n::Int, R::Int, seed::UInt64)
    N = n * n
    target_density = Float32(2 * n) / Float32(N)
    a = max(0.5f0, 2.0f0 * target_density)
    b = max(0.5f0, 2.0f0 * (1.0f0 - target_density))

    state = Matrix{Float32}(undef, N, R)
    for traj in 1:R
        traj_seed = splitmix64(seed ⊻ UInt64(n) ⊻ (UInt64(traj) << 1))
        rng = Xoshiro(traj_seed)
        for i in 1:N
            u = Float32(rand(rng))^(1.0f0/a)
            v = Float32(rand(rng))^(1.0f0/b)
            state[i, traj] = u / (u + v)
        end
    end
    return state
end

# ============================================================================
# Validation (CPU)
# ============================================================================

function count_violations(x_bin, ti, tj, tk)
    count = 0
    @inbounds for m in eachindex(ti)
        count += x_bin[ti[m]] & x_bin[tj[m]] & x_bin[tk[m]]
    end
    return count
end

# ============================================================================
# GPU Solver
# ============================================================================

function solve_metal(cfg::Config, ti_cpu, tj_cpu, tk_cpu, seed::UInt64, outdir::String;
                     verbose::Bool=true, batch_size::Int=1024, max_batches::Int=0)
    n = cfg.n
    N = Int32(n * n)
    M = Int32(length(ti_cpu))
    target = 2 * n
    nsteps = Int32(round(Int, cfg.T / cfg.dt))

    verbose && println("="^60)
    verbose && println("N3L Gradient Flow — Metal GPU (Custom RK4 Kernel)")
    verbose && println("="^60)
    verbose && @printf("n=%d, target=%d | α=%.1f, β=%.1f, γ=%.1f\n", n, target, cfg.α, cfg.β, cfg.γ)
    verbose && @printf("Max: %d traj, T=%.1fs, dt=%.4f, steps=%d\n", cfg.R, cfg.T, cfg.dt, nsteps)
    verbose && @printf("Batch size: %d | Seed: %d\n", batch_size, seed)
    verbose && println("-"^60)
    verbose && @printf("Triples: %d | State dim: %d\n", M, N)
    verbose && println("-"^60)

    # Upload triples to GPU (shared across all batches)
    d_ti = MtlArray(ti_cpu)
    d_tj = MtlArray(tj_cpu)
    d_tk = MtlArray(tk_cpu)

    # Compile kernel
    backend = Metal.MetalBackend()

    # Warmup with tiny batch
    verbose && print("Warming up Metal GPU (compiling kernel)...")
    try
        warmup_state = MtlArray(rand(Float32, N, 2))
        kern = rk4_gradient_flow_kernel!(backend, 64)
        kern(warmup_state, d_ti, d_tj, d_tk,
             cfg.α, cfg.β, cfg.γ, cfg.dt, Int32(1), N, M;
             ndrange=2)
        KernelAbstractions.synchronize(backend)
        verbose && println("\n  ✓ Metal warmup OK")
    catch e
        verbose && println("\n  ⚠ Metal warmup failed: $(typeof(e)): $(sprint(showerror, e))")
        verbose && println("  This is a fatal error — kernel cannot compile.")
        return false, nothing, 0.0, Dict(:success=>0, :tried=>0), seed
    end
    verbose && println("-"^60)

    # Batch loop
    total_tried = 0
    total_batches = cld(cfg.R, batch_size)
    if max_batches > 0
        total_batches = min(total_batches, max_batches)
    end
    total_traj = min(cfg.R, total_batches * batch_size)

    solution_found = false
    solution_grid = nothing
    solution_traj_id = 0
    best_viols = typemax(Int)
    violation_histogram = Dict{Int,Int}()

    start_time = time()

    for batch_idx in 1:total_batches
        this_batch = min(batch_size, total_traj - total_tried)
        this_batch <= 0 && break

        verbose && @printf("[Batch %d/%d] Launching %d trajectories on GPU...\n",
                          batch_idx, total_batches, this_batch)

        # Generate initial conditions on CPU, upload to GPU
        batch_seed = splitmix64(seed ⊻ UInt64(batch_idx) ⊻ (UInt64(0xBA7C4) << 16))
        ic_cpu = generate_initial_conditions(n, this_batch, batch_seed)
        d_state = MtlArray(ic_cpu)

        # Launch kernel — one thread per trajectory
        kern = rk4_gradient_flow_kernel!(backend, min(64, this_batch))
        kern(d_state, d_ti, d_tj, d_tk,
             cfg.α, cfg.β, cfg.γ, cfg.dt, nsteps, N, M;
             ndrange=this_batch)
        KernelAbstractions.synchronize(backend)

        # Pull results back to CPU
        result_cpu = Array(d_state)

        # Validate each trajectory
        batch_best = typemax(Int)
        for traj in 1:this_batch
            x_final = @view result_cpu[:, traj]
            x_bin = topk_mask(x_final, target)
            viols = count_violations(x_bin, ti_cpu, tj_cpu, tk_cpu)

            violation_histogram[viols] = get(violation_histogram, viols, 0) + 1

            if viols < batch_best
                batch_best = viols
            end

            if viols == 0 && !solution_found
                solution_found = true
                solution_grid = reshape(x_bin, (n, n))
                solution_traj_id = total_tried + traj
                elapsed = time() - start_time
                verbose && @printf("  🎉 SOLUTION! batch=%d, traj=%d, time=%.2fs\n",
                                  batch_idx, traj, elapsed)
            end
        end

        total_tried += this_batch

        if batch_best < best_viols
            best_viols = batch_best
        end

        elapsed = time() - start_time
        rate = total_tried / elapsed
        verbose && @printf("  ★ Best violations: %d | %d tried | %.1f traj/s\n",
                          best_viols, total_tried, rate)

        if solution_found
            break
        end
    end

    elapsed = time() - start_time
    verbose && println("-"^60)

    if solution_found
        verbose && println("✓✓✓ SUCCESS ✓✓✓")
        verbose && @printf("Time: %.2fs | Tried: %d/%d (%.1f%%) | Rate: %.1f/s\n",
                          elapsed, total_tried, total_traj,
                          100*total_tried/total_traj, total_tried/elapsed)

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
        save_solution(n, solution_grid, solution_traj_id, cfg.R, cfg.T, seed, cfg.α, cfg.γ, outdir)

        return true, solution_grid, elapsed, Dict(:success=>1, :tried=>total_tried), seed
    else
        verbose && println("✗✗✗ NO SOLUTION FOUND ✗✗✗")
        verbose && @printf("Time: %.2fs | Tried: %d | Best: %d viols | Rate: %.1f/s\n",
                          elapsed, total_tried, best_viols, total_tried/elapsed)

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
# Utility
# ============================================================================

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
        println(io, "# method=Metal GPU Custom RK4 Kernel")
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
    T = Float32(args["T"])
    dt = Float32(args["dt"])
    outdir = args["outdir"]
    quiet = args["quiet"]
    verbose = !quiet
    batch_size = args["batch-size"]
    max_batches = args["max-batches"]

    seed = if isnothing(args["seed"])
        rand(RandomDevice(), UInt64)
    else
        args["seed"]
    end

    # Build config with optional overrides
    alpha_override = args["alpha"]
    gamma_override = args["gamma"]

    cfg = if !isnothing(alpha_override) && !isnothing(gamma_override)
        Config(n=n, R=R, T=T, dt=dt, α=Float32(alpha_override), γ=Float32(gamma_override))
    elseif !isnothing(alpha_override)
        Config(n=n, R=R, T=T, dt=dt, α=Float32(alpha_override))
    elseif !isnothing(gamma_override)
        Config(n=n, R=R, T=T, dt=dt, γ=Float32(gamma_override))
    else
        Config(n=n, R=R, T=T, dt=dt)
    end

    # Check Metal
    verbose && println("Checking Metal GPU...")
    try
        dev = Metal.current_device()
        verbose && println("  Device: $(dev)")
    catch e
        println("ERROR: No Metal GPU available: $e")
        return 1
    end

    # Compute triples
    ti_cpu, tj_cpu, tk_cpu = compute_triples(n)

    success, grid, elapsed, stats, used_seed = solve_metal(
        cfg, ti_cpu, tj_cpu, tk_cpu, seed, outdir;
        verbose=verbose, batch_size=batch_size, max_batches=max_batches)

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
