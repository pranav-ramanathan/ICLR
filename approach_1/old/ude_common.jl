module UDECommon

export UDETrainConfig, UDEDeployConfig

Base.@kwdef struct UDETrainConfig
    epochs::Int = 400
    hidden::Int = 64
    chunks::Int = 16
    lr::Float32 = 1e-3f0
    n_min::Int = 8
    n_max::Int = 20
    seed::UInt64 = UInt64(20260215)
end

Base.@kwdef struct UDEDeployConfig
    mode::String = "neural"
    chunk_steps::Int = 20
    model_path::String = "models/ude_v1_schedule.jld2"
end

end # module
