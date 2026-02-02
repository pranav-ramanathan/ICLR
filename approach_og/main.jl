using OrdinaryDiffEq, ForwardDiff
using LinearAlgebra, Random, Statistics
using Dates, Printf
using DiffEqCallbacks

# -------------------------------
# Grid & collinearity setup
# -------------------------------
const n = 8
const N = n * n
const target_points = 2 * n  # Want exactly 2n points

# CRITICAL FIX: Much stronger collinearity penalty
const α = 200.0  # Was 13.33, now 200.0!
const β = 50.0   # Point count penalty
const γ = 5.0    # Binary regularization

linear_index(i, j) = (i - 1) * n + j

function compute_collinear_triples(n::Int)
    triples = Tuple{Int,Int,Int}[]
    coords = [(i, j) for i in 1:n for j in 1:n]
    
    function collinear(p1, p2, p3)
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) == 0
    end
    
    L = length(coords)
    for a in 1:(L-2), b in (a+1):(L-1), c in (b+1):L
        if collinear(coords[a], coords[b], coords[c])
            push!(triples, (a, b, c))
        end
    end
    
    return triples
end

const L_triples = compute_collinear_triples(n)
println("Found $(length(L_triples)) collinear triples for n=$n")

# -------------------------------
# Top-K mask (guarantees exactly k points)
# -------------------------------
function topk_mask(x::AbstractVector{<:Real}, k::Int)
    idx = partialsortperm(x, 1:k; rev=true)
    m = falses(length(x))
    @inbounds for i in idx
        m[i] = true
    end
    return BitVector(m)
end

# -------------------------------
# Better energy function
# -------------------------------
function energy(x; α=α, β=β, γ=γ)
    # Collinearity penalty - STRONG
    E_col = α * sum((x[i1] * x[i2] * x[i3] for (i1, i2, i3) in L_triples))
    
    # Point count constraint - soft quadratic penalty around target
    count = sum(x)
    E_count = β * (count - target_points)^2
    
    # Binary regularization (push toward 0 or 1)
    E_bin = γ * sum(xi^2 * (1 - xi)^2 for xi in x)
    
    return E_col + E_count + E_bin
end

# -------------------------------
# Pure gradient flow
# -------------------------------
function f!(dx, x, p, t)
    α, β, γ = p
    
    # Clamp x to [0, 1] to prevent numerical issues
    clamp!(x, 0.0, 1.0)
    
    # Compute gradient using AD
    g = ForwardDiff.gradient(x -> energy(x; α=α, β=β, γ=γ), x)
    
    # Pure gradient descent
    @. dx = -g
end

# -------------------------------
# Better initialization (Beta-like distribution)
# -------------------------------
function biased_init(n; target_points=2n)
    N = n * n
    target_density = target_points / N
    
    # Beta-like distribution centered at target density
    a = max(0.5, 2.0 * target_density)
    b = max(0.5, 2.0 * (1.0 - target_density))
    
    x0 = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        u = rand()^(1/a)
        v = rand()^(1/b)
        x0[i] = u / (u + v)
    end
    return x0
end

# -------------------------------
# Validation using top-k (always returns exactly target_points)
# -------------------------------
function count_violations(x; k=target_points)
    x_bin = topk_mask(x, k)
    violations = 0
    for (i1, i2, i3) in L_triples
        if x_bin[i1] && x_bin[i2] && x_bin[i3]
            violations += 1
        end
    end
    return violations, x_bin
end

# -------------------------------
# Single trajectory with callback
# -------------------------------
function solve_single_trajectory(seed, max_time=100.0)
    Random.seed!(seed)
    x0 = biased_init(n)
    
    # Success flag
    success_ref = Ref(false)
    best_x_ref = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    
    # CRITICAL: Callback to check top-k violations DURING solve
    function check_success(integrator)
        x = integrator.u
        violations, x_bin = count_violations(x)
        
        if violations == 0
            success_ref[] = true
            best_x_ref[] = copy(x)
            terminate!(integrator)
            return
        end
    end
    
    # Check every 0.5 seconds
    cb = PeriodicCallback(check_success, 0.5; save_positions=(false, false))
    
    tspan = (0.0, max_time)
    p = (α, β, γ)
    
    prob = ODEProblem(f!, x0, tspan, p)
    sol = solve(prob, Tsit5(); reltol=1e-6, abstol=1e-8, callback=cb,
                save_everystep=false, maxiters=1_000_000)
    
    # Get final result
    if success_ref[]
        xT = best_x_ref[]
    else
        xT = sol.u[end]
    end
    
    violations, x_bin = count_violations(xT)
    
    return success_ref[], xT, violations, x_bin, sol.t[end]
end

# -------------------------------
# Multi-restart strategy with early termination
# -------------------------------
function solve_with_restarts(n_restarts=50, max_time=100.0)
    best_sol = nothing
    best_violations = Inf
    best_seed = 0
    best_grid = nothing
    
    for seed in 1:n_restarts
        success, xT, violations, x_bin, time_taken = solve_single_trajectory(seed, max_time)
        
        points = sum(x_bin)  # Always equals target_points
        energy_val = energy(xT; α=α, β=β, γ=γ)
        
        status = success ? "✓" : " "
        println("Seed $seed: $points points, $violations violations, E=$(round(energy_val, digits=2)), t=$(round(time_taken, digits=1))s $status")
        
        # Early termination if we find optimal solution
        if violations == 0
            println("  ✓✓✓ Found optimal solution! Terminating early.")
            return xT, violations, seed, x_bin
        end
        
        # Track best solution
        if best_sol === nothing
            best_violations = violations
            best_sol = xT
            best_seed = seed
            best_grid = x_bin
        elseif violations < best_violations
            best_violations = violations
            best_sol = xT
            best_seed = seed
            best_grid = x_bin
        end
    end
    
    return best_sol, best_violations, best_seed, best_grid
end

# -------------------------------
# Save solution
# -------------------------------
function save_solution(n, grid, seed, violations, energy_val, outdir="solutions")
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    dir = "$(outdir)/$(n)"
    mkpath(dir)
    filename = "$(dir)/sol_$(timestamp).txt"
    
    open(filename, "w") do io
        println(io, "# n=$(n)")
        println(io, "# target=$(2n)")
        println(io, "# seed=$(seed)")
        println(io, "# violations=$(violations)")
        println(io, "# energy=$(energy_val)")
        println(io, "# timestamp=$(Dates.format(now(), "yyyy-mm-dd'T'HH:MM:SS'Z'"))")
        println(io, "#")
        println(io, "# Grid (0/1):")
        for i in 1:n
            println(io, join(Int.(grid[i, :]), " "))
        end
        println(io, "# ")
        println(io, "# Coordinates (row, col):")
        for i in 1:n, j in 1:n
            grid[i,j] == 1 && println(io, "($i, $j)")
        end
    end
    
    println("\n  Saved: $filename")
end

# -------------------------------
# Pretty print grid
# -------------------------------
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

# -------------------------------
# Run optimization
# -------------------------------
println("\nRunning multi-restart optimization...")
println("Parameters: α=$α, β=$β, γ=$γ")
println("-"^50)

best_sol, best_violations, best_seed, best_grid = solve_with_restarts(50, 100.0)

grid = reshape(best_grid, (n, n))

println("\n" * "="^50)
println("BEST SOLUTION (seed $best_seed):")
println("="^50)
print_grid(grid)

points_placed = sum(best_grid)
println("\nPoints placed: $points_placed (target: $target_points)")
println("Violations: $best_violations")
energy_val = round(energy(best_sol; α=α, β=β, γ=γ), digits=4)
println("Energy: $energy_val")

# Save solution to file
if best_violations == 0
    save_solution(n, grid, best_seed, best_violations, energy_val)
else
    println("\n⚠️  No valid solution found. Try increasing α or n_restarts.")
end