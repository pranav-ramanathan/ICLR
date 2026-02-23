#!/usr/bin/env julia
#=
UDE_ROCm.jl
===========
ROCm-first UDE runner for N3L.

Modes:
- adaptive: original heuristic UDE-lite rounds.
- neural:   full neural UDE v1 (chunked coefficient schedule from trained model).

Neural v1 is deployment-compatible with ROCm-first/no-hybrid requirements:
- model inference happens once up-front to produce (α, γ, δ=β) per chunk,
- rollout runs entirely on GPU in chunked fixed-coefficient launches,
- no CPU controller calls inside the integration loop.

Examples:
  # 1) train a neural schedule model
  julia --project=. train_ude.jl --epochs 400 --hidden 64 --chunks 16 --out models/ude_v1_schedule.jld2

  # 2) run ROCm neural UDE
  julia --project=. UDE_ROCm.jl 16 --mode neural --model models/ude_v1_schedule.jld2 --chunk-steps 20 --R 5000 --T 30 --dt 0.005

  # 3) keep existing adaptive wrapper behavior
  julia --project=. UDE_ROCm.jl 16 --mode adaptive --rounds 6 --R 5000 --T 30 --dt 0.005
=#

include("main_rocm.jl")
include("ude_model.jl")

using .UDEModel
using Random
using Printf
using ArgParse
using AMDGPU
using KernelAbstractions
include("logging_utils.jl")

Base.@kwdef mutable struct UDEConfig
    n::Int
    R::Int = 5000
    T::Float32 = 30.0f0
    dt::Float32 = 0.005f0
    α::Float32 = 8.0f0
    γ::Float32 = 4.0f0
    rounds::Int = 6
    T_growth::Float32 = 1.15f0
    α_step::Float32 = 0.08f0
    γ_step::Float32 = 0.06f0
    α_min::Float32 = 4.0f0
    α_max::Float32 = 20.0f0
    γ_min::Float32 = 2.0f0
    γ_max::Float32 = 10.0f0
end

clampf(x, lo, hi) = min(max(x, lo), hi)

function adapt_params!(c::UDEConfig, round::Int)
    if isodd(round)
        c.α = clampf(c.α * (1.0f0 - c.α_step), c.α_min, c.α_max)
        c.γ = clampf(c.γ * (1.0f0 - c.γ_step), c.γ_min, c.γ_max)
    else
        c.α = clampf(c.α * (1.0f0 + c.α_step), c.α_min, c.α_max)
        c.γ = clampf(c.γ * (1.0f0 + 0.5f0 * c.γ_step), c.γ_min, c.γ_max)
    end
    c.T = c.T * c.T_growth
end

function parse_ude_args(args)
    s = ArgParseSettings(description="ROCm UDE runner for N3L (adaptive + neural v1)")
    @add_arg_table! s begin
        "n"
            arg_type = Int
            required = true
        "--mode"
            arg_type = String
            default = "adaptive"
        "--model"
            arg_type = String
            default = "models/ude_v1_schedule.jld2"
        "--chunk-steps"
            arg_type = Int
            default = 20
        "--R"
            arg_type = Int
            default = 5000
        "--T"
            arg_type = Float64
            default = 30.0
        "--dt"
            arg_type = Float64
            default = 0.005
        "--alpha"
            arg_type = Float64
            default = 8.0
        "--gamma"
            arg_type = Float64
            default = 4.0
        "--rounds"
            arg_type = Int
            default = 6
        "--seed"
            arg_type = UInt64
        "--batch-size"
            arg_type = Int
            default = 1024
        "--max-batches"
            arg_type = Int
            default = 0
        "--outdir"
            arg_type = String
            default = "solutions"
        "--quiet", "-q"
            action = :store_true
    end
    return parse_args(args, s)
end

function _build_neural_schedule(n::Int, T::Float32, dt::Float32, chunk_steps::Int, model_path::String)
    model, ps, st, cfg, _ = load_checkpoint(model_path)
    total_steps = max(1, floor(Int, T / dt))
    K = max(1, cld(total_steps, chunk_steps))
    αs, γs, δs, _ = predict_schedule(model, ps, st, n, K, cfg)
    return αs, γs, δs, total_steps, K
end

function solve_rocm_neural_schedule(
    n::Int,
    R::Int,
    T::Float32,
    dt::Float32,
    seed::UInt64,
    outdir::String,
    line_offsets::Vector{Int32},
    line_points::Vector{Int32},
    point_col_scale::Vector{Float32},
    αs::Vector{Float32},
    γs::Vector{Float32},
    δs::Vector{Float32};
    chunk_steps::Int=20,
    batch_size::Int=1024,
    max_batches::Int=0,
    verbose::Bool=true,
    col_normalization::String="mean-incidence",
)
    N = Int32(n * n)
    L = Int32(length(line_offsets) - 1)
    target = 2 * n

    Int(N) > MAX_STATE_DIM && error("n^2=$(Int(N)) exceeds MAX_STATE_DIM=$MAX_STATE_DIM")

    d_line_offsets = ROCArray(line_offsets)
    d_line_points = ROCArray(line_points)
    d_point_col_scale = ROCArray(point_col_scale)

    backend = AMDGPU.ROCBackend()
    kern = rk4_gradient_flow_lines_kernel!(backend, 64)

    # warmup
    warmup_state = ROCArray(rand(Float32, N, 2))
    kern(warmup_state, d_line_offsets, d_line_points, d_point_col_scale,
         αs[1], δs[1], γs[1], dt, Int32(1), N, L; ndrange=2)
    KernelAbstractions.synchronize(backend)

    total_batches = cld(R, batch_size)
    if max_batches > 0
        total_batches = min(total_batches, max_batches)
    end
    total_traj = min(R, total_batches * batch_size)

    total_tried = 0
    start_time = time()
    solution_found = false
    solution_grid = nothing
    solution_traj_id = 0
    best_viols = typemax(Int)

    for batch_idx in 1:total_batches
        this_batch = min(batch_size, total_traj - total_tried)
        this_batch <= 0 && break

        batch_seed = splitmix64(seed ⊻ UInt64(batch_idx) ⊻ (UInt64(0xBADA55) << 8))
        ic_cpu = generate_initial_conditions(n, this_batch, batch_seed)
        d_state = ROCArray(ic_cpu)

        rem_steps = max(1, floor(Int, T / dt))
        k = 1
        while rem_steps > 0
            ns = Int32(min(chunk_steps, rem_steps))
            α = αs[min(k, length(αs))]
            γ = γs[min(k, length(γs))]
            δ = δs[min(k, length(δs))]
            kern(d_state, d_line_offsets, d_line_points, d_point_col_scale,
                 α, δ, γ, dt, ns, N, L; ndrange=this_batch)
            rem_steps -= Int(ns)
            k += 1
        end
        KernelAbstractions.synchronize(backend)

        result_cpu = Array(d_state)
        batch_best = typemax(Int)
        for traj in 1:this_batch
            x_final = @view result_cpu[:, traj]
            x_bin = topk_mask(x_final, target)
            viols = count_violations_lines(x_bin, line_offsets, line_points)
            batch_best = min(batch_best, viols)
            if viols == 0 && !solution_found
                solution_found = true
                solution_grid = reshape(x_bin, (n, n))
                solution_traj_id = total_tried + traj
            end
        end

        total_tried += this_batch
        best_viols = min(best_viols, batch_best)

        if verbose
            elapsed = time() - start_time
            @printf("[neural batch %d/%d] tried=%d best=%d rate=%.1f traj/s\n",
                    batch_idx, total_batches, total_tried, best_viols, total_tried / max(elapsed, 1e-6))
        end

        solution_found && break
    end

    elapsed = time() - start_time

    if solution_found
        verbose && println("UDE-ROCm neural SUCCESS")
        print_grid(solution_grid)
        save_solution(n, solution_grid, solution_traj_id, R, T, seed,
                      αs[1], γs[1], outdir; col_normalization=col_normalization)
        return true, solution_grid, elapsed, Dict(:success => 1, :tried => total_tried), seed
    else
        verbose && @printf("UDE-ROCm neural: no solution found. tried=%d best=%d\n", total_tried, best_viols)
        return false, nothing, elapsed, Dict(:success => 0, :tried => total_tried), seed
    end
end

function run_adaptive(args, seed, line_offsets, line_points, point_col_scale, col_norm_mode, verbose)
    n = args["n"]
    ude = UDEConfig(
        n=n,
        R=args["R"],
        T=Float32(args["T"]),
        dt=Float32(args["dt"]),
        α=Float32(args["alpha"]),
        γ=Float32(args["gamma"]),
        rounds=args["rounds"],
    )

    for round in 1:ude.rounds
        round_seed = splitmix64(seed ⊻ UInt64(round))
        cfg = Config(n=ude.n, R=ude.R, T=ude.T, dt=ude.dt, α=ude.α, γ=ude.γ)

        verbose && @printf("\n[UDE-ROCm round %d/%d] α=%.3f γ=%.3f T=%.2f seed=%d\n",
                           round, ude.rounds, ude.α, ude.γ, ude.T, round_seed)

        ok, _, elapsed, _, _ = solve_rocm(
            cfg,
            line_offsets,
            line_points,
            point_col_scale,
            round_seed,
            args["outdir"];
            verbose=verbose,
            batch_size=args["batch-size"],
            max_batches=args["max-batches"],
            col_normalization=col_norm_mode,
        )

        if ok
            verbose && @printf("UDE-ROCm adaptive SUCCESS at round %d (%.2fs)\n", round, elapsed)
            return 0
        end

        adapt_params!(ude, round)
    end

    verbose && println("UDE-ROCm adaptive: no solution found within budget.")
    return 1
end

function run_neural(args, seed, line_offsets, line_points, point_col_scale, col_norm_mode, verbose)
    n = args["n"]
    model_path = args["model"]
    isfile(model_path) || error("Neural model not found: $model_path")

    αs, γs, δs, total_steps, K = _build_neural_schedule(
        n, Float32(args["T"]), Float32(args["dt"]), args["chunk-steps"], model_path
    )

    if verbose
        @printf("[UDE-ROCm neural] model=%s\n", model_path)
        @printf("  chunks=%d, chunk_steps=%d, total_steps=%d\n", K, args["chunk-steps"], total_steps)
        @printf("  α range: [%.3f, %.3f]\n", minimum(αs), maximum(αs))
        @printf("  γ range: [%.3f, %.3f]\n", minimum(γs), maximum(γs))
        @printf("  δ(β) range: [%.3f, %.3f]\n", minimum(δs), maximum(δs))
    end

    ok, _, _, _, _ = solve_rocm_neural_schedule(
        n,
        args["R"],
        Float32(args["T"]),
        Float32(args["dt"]),
        seed,
        args["outdir"],
        line_offsets,
        line_points,
        point_col_scale,
        αs,
        γs,
        δs;
        chunk_steps=args["chunk-steps"],
        batch_size=args["batch-size"],
        max_batches=args["max-batches"],
        verbose=verbose,
        col_normalization=col_norm_mode,
    )

    return ok ? 0 : 1
end

function main()
    args = parse_ude_args(ARGS)
    verbose = !args["quiet"]

    seed = isnothing(args["seed"]) ? rand(RandomDevice(), UInt64) : args["seed"]
    n = args["n"]

    # Precompute static geometry once.
    line_offsets, line_points = compute_lines(n)
    point_col_scale, _, _, _, col_norm_mode = compute_point_collinearity_scale(
        n, line_offsets, line_points; mode="mean-incidence"
    )

    mode = lowercase(args["mode"])
    if mode == "adaptive"
        return run_adaptive(args, seed, line_offsets, line_points, point_col_scale, col_norm_mode, verbose)
    elseif mode == "neural"
        return run_neural(args, seed, line_offsets, line_points, point_col_scale, col_norm_mode, verbose)
    else
        error("Invalid --mode '$mode' (expected: adaptive | neural)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_with_terminal_log("ude_rocm_v1", ARGS) do
        main()
    end)
end
