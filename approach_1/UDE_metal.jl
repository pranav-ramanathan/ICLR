#!/usr/bin/env julia
#=
UDE_metal.jl
============
Metal backend counterpart of UDE_ROCm.
No GPU+CPU hybrid mode.
=#

include("main_metal_v3.jl")
using Random
using Printf
using ArgParse

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
    s = ArgParseSettings(description="Adaptive UDE-lite Metal runner for N3L")
    @add_arg_table! s begin
        "n"
            arg_type = Int
            required = true
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

function main()
    args = parse_ude_args(ARGS)
    verbose = !args["quiet"]

    seed = isnothing(args["seed"]) ? rand(RandomDevice(), UInt64) : args["seed"]
    n = args["n"]

    line_offsets, line_points = compute_lines(n)
    point_col_scale, _, _, _, col_norm_mode = compute_point_collinearity_scale(
        n, line_offsets, line_points; mode="mean-incidence"
    )

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

        verbose && @printf("\n[UDE-Metal round %d/%d] α=%.3f γ=%.3f T=%.2f seed=%d\n",
                           round, ude.rounds, ude.α, ude.γ, ude.T, round_seed)

        ok, grid, elapsed, stats, _ = solve_metal(
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
            verbose && @printf("UDE-Metal SUCCESS at round %d (%.2fs)\n", round, elapsed)
            return 0
        end

        adapt_params!(ude, round)
    end

    verbose && println("UDE-Metal: no solution found within adaptive budget.")
    return 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
