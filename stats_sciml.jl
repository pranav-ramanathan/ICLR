module N3L

using Random: MersenneTwister, shuffle!
using OrdinaryDiffEq: ODEProblem, solve, Tsit5, terminate!, DiscreteCallback, CallbackSet
using Printf: @printf, @sprintf
using Dates: now, format

export N3LProblem, solve_n3l, race_search, analyze_solution

struct N3LProblem{T<:AbstractFloat}
    n::Int
    k::Int
    lines::Vector{Vector{Int}}
    α::T
    β::T
    λ::T
    ρ::T
    grad::Vector{T}
end

mutable struct ConvergenceState{T<:AbstractFloat}
    best_energy::T
    last_improve_t::T
end

ConvergenceState(E₀::T) where {T} = ConvergenceState{T}(E₀, zero(T))

function N3LProblem(n::Int; k::Int=2n, α=50.0, β=5.0, λ=10.0, ρ=10.0)
    lines = _generate_lines(n)
    T = promote_type(typeof(α), typeof(β), typeof(λ), typeof(ρ))
    N3LProblem{T}(n, k, lines, T(α), T(β), T(λ), T(ρ), zeros(T, n^2))
end

function _generate_lines(n::Int)
    @inline idx(i, j) = (i - 1) * n + j
    
    dirs = Set{Tuple{Int,Int}}()
    for dx in 0:n-1, dy in 0:n-1
        (dx == 0 && dy == 0) && continue
        g = gcd(dx, dy)
        dxp, dyp = dx ÷ g, dy ÷ g
        if dxp > 0 || (dxp == 0 && dyp > 0)
            push!(dirs, (dxp, dyp))
        end
    end
    
    seen = Set{Vector{Int}}()
    lines = Vector{Vector{Int}}()
    
    for (dx, dy) in dirs
        for i in 1:n, j in 1:n
            (1 ≤ i - dx ≤ n && 1 ≤ j - dy ≤ n) && continue
            
            pts = Int[]
            ii, jj = i, j
            while 1 ≤ ii ≤ n && 1 ≤ jj ≤ n
                push!(pts, idx(ii, jj))
                ii += dx
                jj += dy
            end
            
            if length(pts) ≥ 3
                key = sort(pts)
                if key ∉ seen
                    push!(seen, key)
                    push!(lines, pts)
                end
            end
        end
    end
    
    lines
end

function energy(u::AbstractVector, p::N3LProblem{T}) where {T}
    E_line = _energy_lines(u, p.lines)
    E_bin = _energy_binary(u)
    E_mass = _energy_mass(u, p.k)
    E_box = _energy_box(u)
    p.α * E_line + p.β * E_bin + p.λ * E_mass + p.ρ * E_box
end

@inline function _energy_lines(u, lines)
    E = zero(eltype(u))
    @inbounds for L in lines
        s = zero(eltype(u))
        for i in L
            s += u[i]
        end
        r = max(s - 2, zero(eltype(u)))
        E += r * r
    end
    E
end

@inline function _energy_binary(u)
    E = zero(eltype(u))
    @inbounds for i in eachindex(u)
        x = u[i]
        E += x * x * (1 - x) * (1 - x)
    end
    E
end

@inline _energy_mass(u, k) = (sum(u) - k)^2

@inline function _energy_box(u)
    E = zero(eltype(u))
    @inbounds for i in eachindex(u)
        x = u[i]
        lo = max(-x, zero(eltype(u)))
        hi = max(x - 1, zero(eltype(u)))
        E += lo * lo + hi * hi
    end
    E
end

function grad_energy!(p::N3LProblem{T}, u::AbstractVector) where {T}
    g = p.grad
    fill!(g, zero(T))
    
    @inbounds for L in p.lines
        s = zero(T)
        for i in L
            s += u[i]
        end
        z = s - 2
        if z > 0
            c = p.α * 2 * z
            for i in L
                g[i] += c
            end
        end
    end
    
    @inbounds for i in eachindex(u)
        x = u[i]
        g[i] += p.β * (2x - 6x^2 + 4x^3)
    end
    
    mass_grad = p.λ * 2 * (sum(u) - p.k)
    @inbounds for i in eachindex(u)
        g[i] += mass_grad
    end
    
    @inbounds for i in eachindex(u)
        x = u[i]
        if x < 0
            g[i] += p.ρ * 2x
        elseif x > 1
            g[i] += p.ρ * 2(x - 1)
        end
    end
    
    g
end

function rhs!(du, u, p::N3LProblem, t)
    grad_energy!(p, u)
    @inbounds @simd for i in eachindex(du)
        du[i] = -p.grad[i]
    end
    nothing
end

function solve_n3l(p::N3LProblem{T};
                   tspan::Real = 3.0,
                   tol::Real = 1e-6,
                   patience::Real = 0.5,
                   check_dt::Real = 0.05,
                   seed::Int = 0) where {T}
    
    rng = MersenneTwister(seed)
    u0 = rand(rng, T, p.n^2)
    
    state = ConvergenceState(energy(u0, p))
    
    cb_success = DiscreteCallback(
        (u, t, int) -> energy(u, int.p) ≤ tol,
        terminate!;
        save_positions = (false, false)
    )
    
    last_check_t = Ref(zero(T))
    function stall_condition(u, t, int)
        t - last_check_t[] ≥ check_dt
    end
    function stall_affect!(int)
        last_check_t[] = int.t
        E = energy(int.u, int.p)
        if E < 0.999 * state.best_energy
            state.best_energy = E
            state.last_improve_t = int.t
        end
        if int.t - state.last_improve_t > patience && state.best_energy > tol
            terminate!(int)
        end
        nothing
    end
    cb_stall = DiscreteCallback(stall_condition, stall_affect!; 
                                save_positions = (false, false))
    
    prob = ODEProblem(rhs!, u0, (zero(T), T(tspan)), p)
    sol = solve(prob, Tsit5();
                callback = CallbackSet(cb_success, cb_stall),
                abstol = 1e-6, reltol = 1e-6,
                save_everystep = false)
    
    u_final = sol.u[end]
    (u = u_final, E = energy(u_final, p), best_E = state.best_energy, t = sol.t[end])
end

function threshold_board(u::AbstractVector, n::Int; thr=0.5)
    reshape([x > thr ? 1 : 0 for x in u], n, n)
end

function count_violations(board::AbstractMatrix, lines::Vector{Vector{Int}})
    flat = vec(board)
    violations = 0
    @inbounds for L in lines
        s = 0
        for i in L
            s += flat[i]
        end
        if s >= 3
            violations += 1
        end
    end
    violations
end

function race_search(n::Int;
                     k::Int = 2n,
                     α_list = [50.0, 100.0, 200.0],
                     β_list = [2.0, 5.0, 10.0],
                     λ_list = [5.0, 10.0, 20.0],
                     ρ_list = [5.0, 10.0],
                     seeds = 1:200,
                     tspan::Real = 3.0,
                     tol::Real = 1e-6,
                     patience::Real = 0.5,
                     check_dt::Real = 0.05,
                     verbose::Bool = false)
    
    base = N3LProblem(n; k, α=first(α_list), β=first(β_list), λ=first(λ_list), ρ=first(ρ_list))
    lines = base.lines
    
    best_solutions = Vector{NamedTuple{(:u, :E, :cfg, :board, :pts, :viol), 
                                        Tuple{Vector{Float64}, Float64, NamedTuple, Matrix{Int}, Int, Int}}}()
    
    cfgs = [(α, β, λ, ρ) for α in α_list for β in β_list for λ in λ_list for ρ in ρ_list]
    
    verbose && @printf("n=%d k=%d lines=%d configs=%d seeds=%d\n",
                       n, k, length(lines), length(cfgs), length(seeds))
    
    for (ci, (α, β, λ, ρ)) in enumerate(cfgs)
        p = N3LProblem{Float64}(n, k, lines, α, β, λ, ρ, zeros(n^2))
        
        verbose && @printf("\nConfig %d/%d: a=%.1f b=%.1f l=%.1f r=%.1f\n",
                           ci, length(cfgs), α, β, λ, ρ)
        
        for s in seeds
            out = solve_n3l(p; tspan, tol, patience, check_dt, seed=s)
            
            board = threshold_board(out.u, n)
            npts = sum(board)
            nviol = count_violations(board, lines)
            cfg = (α=α, β=β, λ=λ, ρ=ρ, seed=s, t=out.t)
            
            verbose && @printf("  seed=%d t=%.3f E=%.3e pts=%d viol=%d\n",
                               s, out.t, out.E, npts, nviol)
            
            if out.E ≤ tol && nviol == 0
                push!(best_solutions, (u=copy(out.u), E=out.E, cfg=cfg, 
                                       board=board, pts=npts, viol=nviol))
            end
        end
    end
    
    (solutions = best_solutions, n = n, k = k, lines = length(lines))
end

function analyze_solution(result; file::Union{String,Nothing}=nothing)
    if isempty(result.solutions)
        println("No valid solutions found")
        return nothing
    end
    
    sol = result.solutions[rand(1:length(result.solutions))]
    
    @printf("n=%d k=%d | points=%d violations=%d | E=%.2e\n", 
            result.n, result.k, sol.pts, sol.viol, sol.E)
    display(sol.board)
    
    if file !== nothing
        timestamp = format(now(), "HHMMss")
        filename = "n_$(result.n)_$(timestamp).txt"
        open(filename, "w") do io
            println(io, "# n=$(result.n) k=$(result.k) points=$(sol.pts) violations=$(sol.viol)")
            println(io, "# E=$(sol.E) cfg=$(sol.cfg)")
            for row in eachrow(sol.board)
                println(io, join(row, " "))
            end
        end
        println("\nSaved to: $filename")
    end
    
    sol
end

end # module

using .N3L

function main()
    n = 5
    seeds = 1:50
    tspan = 3.0
    tol = 1e-6
    save_file = false
    verbose = false
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "-n" && i < length(ARGS)
            n = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--seeds" && i < length(ARGS)
            seeds = 1:parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--tspan" && i < length(ARGS)
            tspan = parse(Float64, ARGS[i+1])
            i += 2
        elseif arg == "--tol" && i < length(ARGS)
            tol = parse(Float64, ARGS[i+1])
            i += 2
        elseif arg == "--file"
            save_file = true
            i += 1
        elseif arg == "-v" || arg == "--verbose"
            verbose = true
            i += 1
        elseif arg == "-h" || arg == "--help"
            println("Usage: julia --project=. stats_sciml.jl [options]")
            println("  -n N          Grid size (default: 5)")
            println("  --seeds N     Number of seeds (default: 50)")
            println("  --tspan T     Integration time (default: 3.0)")
            println("  --tol T       Energy tolerance (default: 1e-6)")
            println("  --file        Save best solution to n_N_HHMMSS.txt")
            println("  -v, --verbose Print per-seed progress")
            return
        else
            i += 1
        end
    end
    
    result = race_search(n; k=2n, seeds=seeds, tspan=tspan, tol=tol, verbose=verbose)
    analyze_solution(result; file = save_file ? "" : nothing)
end

main()
