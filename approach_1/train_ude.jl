#!/usr/bin/env julia

include("ude_model.jl")
using .UDEModel
using ArgParse
using Dates

function parse_args(args)
    s = ArgParseSettings(description="Train neural UDE v1 schedule model (ROCm deployment-compatible)")
    @add_arg_table! s begin
        "--epochs"
            arg_type = Int
            default = 400
        "--hidden"
            arg_type = Int
            default = 64
        "--chunks"
            arg_type = Int
            default = 16
        "--lr"
            arg_type = Float64
            default = 1e-3
        "--seed"
            arg_type = UInt64
            default = UInt64(20260215)
        "--n-min"
            arg_type = Int
            default = 8
        "--n-max"
            arg_type = Int
            default = 20
        "--out"
            arg_type = String
            default = "models/ude_v1_schedule.jld2"
        "--quiet", "-q"
            action = :store_true
    end
    parse_args(args, s)
end

function main()
    args = parse_args(ARGS)
    verbose = !args["quiet"]

    model, ps, st, cfg = train_schedule_model!(;
        hidden=args["hidden"],
        chunks=args["chunks"],
        epochs=args["epochs"],
        lr=Float32(args["lr"]),
        seed=args["seed"],
        n_min=args["n-min"],
        n_max=args["n-max"],
        verbose=verbose,
    )

    meta = Dict(
        "trained_at_utc" => Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SSZ"),
        "impl" => "ude_v1_neural_schedule",
        "notes" => "Teacher-distilled neural schedule; deployment uses ROCm chunked fixed-coeff launches",
    )

    out = save_checkpoint(args["out"], model, ps, st, cfg; meta=meta)
    println("Saved neural UDE model: $out")

    println("\nExample:")
    println("  julia --project=. UDE_ROCm.jl 16 --mode neural --model $out --chunk-steps 20 --R 5000 --T 30 --dt 0.005")

    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
