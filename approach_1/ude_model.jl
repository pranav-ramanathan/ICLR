module UDEModel

using Random
using Lux
using Optimisers
using Zygote
using JLD2
using Printf

export NeuralScheduleConfig, build_model, predict_schedule, train_schedule_model!, save_checkpoint, load_checkpoint

Base.@kwdef struct NeuralScheduleConfig
    hidden::Int = 64
    chunks::Int = 16
    alpha_min::Float32 = 4.0f0
    alpha_max::Float32 = 20.0f0
    gamma_min::Float32 = 2.0f0
    gamma_max::Float32 = 10.0f0
    delta_min::Float32 = 0.5f0
    delta_max::Float32 = 2.0f0
end

build_model(hidden::Int=64) = Chain(
    Dense(5, hidden, tanh),
    Dense(hidden, hidden, tanh),
    Dense(hidden, 3)
)

@inline σ(x) = 1.0f0 / (1.0f0 + exp(-x))

function _feature(n::Int, k::Int, K::Int)
    τ = Float32((k - 1) / max(1, K - 1))
    n̂ = Float32(n / 32.0)
    return Float32[τ, sinpi(2f0 * τ), cospi(2f0 * τ), τ * τ, n̂]
end

function _scale_triplet(raw::AbstractVector{<:Real}, cfg::NeuralScheduleConfig)
    α = cfg.alpha_min + (cfg.alpha_max - cfg.alpha_min) * σ(Float32(raw[1]))
    γ = cfg.gamma_min + (cfg.gamma_max - cfg.gamma_min) * σ(Float32(raw[2]))
    δ = cfg.delta_min + (cfg.delta_max - cfg.delta_min) * σ(Float32(raw[3]))
    return α, γ, δ
end

function predict_schedule(model, ps, st, n::Int, K::Int, cfg::NeuralScheduleConfig)
    αs = Vector{Float32}(undef, K)
    γs = Vector{Float32}(undef, K)
    δs = Vector{Float32}(undef, K)
    st_local = st

    for k in 1:K
        x = _feature(n, k, K)
        y, st_local = Lux.apply(model, x, ps, st_local)
        αs[k], γs[k], δs[k] = _scale_triplet(y, cfg)
    end

    return αs, γs, δs, st_local
end

# v1 supervised training target: distill the existing adaptive heuristic into a neural schedule.
# This gives a true neural model path while staying deterministic + lightweight.
function _teacher_schedule(n::Int, K::Int, cfg::NeuralScheduleConfig)
    α = 8.0f0
    γ = 4.0f0
    δ = 1.0f0

    αs = Vector{Float32}(undef, K)
    γs = Vector{Float32}(undef, K)
    δs = Vector{Float32}(undef, K)

    for k in 1:K
        αs[k] = clamp(α, cfg.alpha_min, cfg.alpha_max)
        γs[k] = clamp(γ, cfg.gamma_min, cfg.gamma_max)
        δs[k] = clamp(δ, cfg.delta_min, cfg.delta_max)

        if isodd(k)
            α *= 0.92f0
            γ *= 0.94f0
        else
            α *= 1.08f0
            γ *= 1.03f0
        end
    end

    return αs, γs, δs
end

function train_schedule_model!(; hidden::Int=64, chunks::Int=16, epochs::Int=400, lr::Float32=1e-3f0,
                               seed::UInt64=0xBEEF, n_min::Int=8, n_max::Int=20, verbose::Bool=true)
    cfg = NeuralScheduleConfig(hidden=hidden, chunks=chunks)
    rng = Xoshiro(seed)

    model = build_model(hidden)
    ps, st = Lux.setup(rng, model)
    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, ps)

    ns = collect(n_min:n_max)

    loss_of(p) = begin
        total = 0.0f0
        st_tmp = st
        for n in ns
            α_pred, γ_pred, δ_pred, st_tmp = predict_schedule(model, p, st_tmp, n, chunks, cfg)
            α_t, γ_t, δ_t = _teacher_schedule(n, chunks, cfg)
            total += sum((α_pred .- α_t).^2) / chunks
            total += sum((γ_pred .- γ_t).^2) / chunks
            total += sum((δ_pred .- δ_t).^2) / chunks
        end
        total / length(ns)
    end

    for ep in 1:epochs
        grads = Zygote.gradient(loss_of, ps)[1]
        opt_state, ps = Optimisers.update(opt_state, ps, grads)
        if verbose && (ep == 1 || ep % 50 == 0 || ep == epochs)
            @printf("[train_ude] epoch=%d/%d loss=%.6f\n", ep, epochs, loss_of(ps))
        end
    end

    return model, ps, st, cfg
end

function save_checkpoint(path::String, model, ps, st, cfg::NeuralScheduleConfig; meta=Dict{String,Any}())
    mkpath(dirname(path))
    jldsave(path; model, ps, st, cfg, meta)
    return path
end

function load_checkpoint(path::String)
    d = load(path)
    return d["model"], d["ps"], d["st"], d["cfg"], get(d, "meta", Dict{String,Any}())
end

end # module
