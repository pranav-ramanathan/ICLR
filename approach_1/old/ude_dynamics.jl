module UDEDynamics

export chunk_count

"""
    chunk_count(T, dt, chunk_steps)

Utility used by neural schedule deployment.
"""
function chunk_count(T::Real, dt::Real, chunk_steps::Integer)
    total_steps = max(1, floor(Int, T / dt))
    return max(1, cld(total_steps, chunk_steps)), total_steps
end

end # module
