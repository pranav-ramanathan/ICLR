using OrdinaryDiffEq, ForwardDiff
using LinearAlgebra, Random, Statistics
using Dates, Printf

# -------------------------------
# Grid & collinearity setup
# -------------------------------
const n = 8
const N = n * n
const target_points = 2 * n  # Want exactly 2n points

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
# Better energy function
# -------------------------------
function energy(x; α=100.0, β=10.0, γ=2.0)
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
# Pure gradient flow (no noise)
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
# Smart initialization
# -------------------------------
function smart_init(n; target_points=2n)
    N = n * n
    density = target_points / N
    # Start near target density with noise
    x0 = fill(density, N) .+ 0.2 * randn(N)
    clamp!(x0, 0.0, 1.0)
    return x0
end

# -------------------------------
# Validation
# -------------------------------
function count_violations(x; threshold=0.5)
    x_bin = x .>= threshold
    violations = 0
    for (i1, i2, i3) in L_triples
        if x_bin[i1] && x_bin[i2] && x_bin[i3]
            violations += 1
        end
    end
    return violations
end

# -------------------------------
# Multi-restart strategy
# -------------------------------
function solve_with_restarts(n_restarts=10)
    best_sol = nothing
    best_violations = Inf
    best_seed = 0
    
    for seed in 1:n_restarts
        Random.seed!(seed)
        x0 = smart_init(n)
        
        tspan = (0.0, 50.0)
        p = (100.0, 50.0, 2.0)  # (α, β, γ) - heavy collinearity penalty
        
        prob = ODEProblem(f!, x0, tspan, p)
        sol = solve(prob, Tsit5(); reltol=1e-6, abstol=1e-8)
        
        xT = sol[:, end]
        violations = count_violations(xT)
        points = sum(Int.(xT .>= 0.5))
        
        println("Seed $seed: $points points, $violations violations, E=$(round(energy(xT; α=p[1], β=p[2], γ=p[3]), digits=2))")
        
        if violations < best_violations || (violations == best_violations && abs(points - target_points) < abs(sum(Int.(best_sol .>= 0.5)) - target_points))
            best_violations = violations
            best_sol = xT
            best_seed = seed
        end
    end
    
    return best_sol, best_violations, best_seed
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
# Run optimization
# -------------------------------
println("\nRunning multi-restart optimization...")
best_sol, best_violations, best_seed = solve_with_restarts(20)

x_binary = Int.(best_sol .>= 0.5)
grid = reshape(x_binary, (n, n))

println("\n" * "="^50)
println("BEST SOLUTION (seed $best_seed):")
println("="^50)
for i in 1:n
    println(grid[i, :])
end

points_placed = sum(x_binary)
println("\nPoints placed: $points_placed (target: $target_points)")
println("Violations: $best_violations")
energy_val = round(energy(best_sol; α=100.0, β=10.0, γ=2.0), digits=4)
println("Energy: $energy_val")

# Save solution to file
save_solution(n, grid, best_seed, best_violations, energy_val)