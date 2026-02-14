#!/usr/bin/env julia
#=
UDE_cpu.jl
==========
CPU backend adaptive UDE-lite runner for N3L.
No GPU+CPU hybrid mode.
=#

include("main_v5.jl")
using Random
using Printf
using ArgParse

Base.@kwdef mutable struct UDEConfig
    n::Int
    R::Int = 4000
    T::Float64 = 20.0
    α::Float64 = 8.0
    γ::Float64 = 4.0
    rounds::Int = 6
    T_growth::Float64 = 1.15
    α_step::Float64 = 0.08
    γ_step::Float64 = 0.06
    α_min::Float64 = 4.0
    α_max::Float64 = 20.0
    γ_min::Float64 = 2.0
    γ_max::Float64 = 10.0
end

clampf(x, lo, hi) = min(max(x, lo), hi)

function adapt_params!(c::UDEConfig, round::Int)
    if isodd(round)
        c.α = clampf(c.α * (1.0 - c.α_step), c.α_min, c.α_max)
        c.γ = clampf(c.γ * (1.0 - c.γ_step), c.γ_min, c.γ_max)
    else
        c.α = clampf(c.α * (1.0 + c.α_step), c.α_min, c.α_max)
        c.γ = clampf(c.γ * (1.0 + 0.5 * c.γ_step), c.γ_min, c.γ_max)
    end
    c.T = c.T * c.T_growth
end

function parse_ude_args(args)
    s = ArgParseSettings(description="Adaptive UDE-lite CPU runner for N3L")
    @add_arg_table! s begin
        "n"
            arg_type = Int
            required = true
        "--R"
            arg_type = Int
            default = 4000
        "--T"
            arg_type = Float64
            default = 20.0
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

    ude = UDEConfig(
        n=args["n"],
        R=args["R"],
        T=args["T"],
        α=args["alpha"],
        γ=args["gamma"],
        rounds=args["rounds"],
    )

    for round in 1:ude.rounds
        round_seed = splitmix64(seed ⊻ UInt64(round))

        verbose && @printf("\n[UDE-CPU round %d/%d] α=%.3f γ=%.3f T=%.2f seed=%d\n",
                           round, ude.rounds, ude.α, ude.γ, ude.T, round_seed)

        ok, grid, elapsed, stats, _ = solve_n3l(
            ude.n,
            ude.R,
            ude.T,
            round_seed,
            args["outdir"];
            alpha_override=ude.α,
            gamma_override=ude.γ,
            verbose=verbose,
        )

        if ok
            verbose && @printf("UDE-CPU SUCCESS at round %d (%.2fs)\n", round, elapsed)
            return 0
        end

        adapt_params!(ude, round)
    end

    verbose && println("UDE-CPU: no solution found within adaptive budget.")
    return 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
