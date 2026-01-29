module N3L

using Random: MersenneTwister, shuffle!
using OrdinaryDiffEq: ODEProblem, solve, Tsit5, terminate!, DiscreteCallback, CallbackSet

using Printf: @printf, @sprintf
using Dates: now, format

export N3LProblem, solve_n3l, race_search, race_search_parallel, analyze_solution

# ============================================================================
# Mutable problem struct with CSR-style line storage and preallocated buffers
# ============================================================================
mutable struct N3LProblem{T<:AbstractFloat}
    const n::Int
    const k::Int
    # CSR-style packed line representation (immutable after construction)
    const line_ptr::Vector{Int}   # length m+1, indices into line_idx
    const line_idx::Vector{Int}   # flattened cell indices for all lines
    const nlines::Int             # number of lines (m)
    # Tunable parameters (updated per-job)
    α::T
    β::T
    λ::T
    ρ::T
    # Preallocated workspace (per-worker, never shared)
    const grad::Vector{T}
    const linesums::Vector{T}     # per-line sums cache
    # Cached energy (updated during gradient computation)
    last_energy::T
end

mutable struct ConvergenceState{T<:AbstractFloat}
    best_energy::T
    last_improve_t::T
end

ConvergenceState(E₀::T) where {T} = ConvergenceState{T}(E₀, zero(T))

# Constructor with CSR line generation
function N3LProblem(n::Int; k::Int=2n, α=50.0, β=5.0, λ=10.0, ρ=10.0)
    line_ptr, line_idx = _generate_lines_csr(n)
    nlines = length(line_ptr) - 1
    T = promote_type(typeof(α), typeof(β), typeof(λ), typeof(ρ))
    N3LProblem{T}(n, k, line_ptr, line_idx, nlines,
                  T(α), T(β), T(λ), T(ρ),
                  zeros(T, n^2), zeros(T, nlines), zero(T))
end

# Update only tunable parameters (reuse buffers)
@inline function update_params!(p::N3LProblem{T}, α::T, β::T, λ::T, ρ::T) where {T}
    p.α = α
    p.β = β
    p.λ = λ
    p.ρ = ρ
    nothing
end

# ============================================================================
# CSR-style line generation (replaces Vector{Vector{Int}})
# ============================================================================
function _generate_lines_csr(n::Int)
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
    lines_temp = Vector{Vector{Int}}()
    
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
                    push!(lines_temp, pts)
                end
            end
        end
    end
    
    # Convert to CSR format
    m = length(lines_temp)
    line_ptr = Vector{Int}(undef, m + 1)
    total_len = sum(length, lines_temp)
    line_idx = Vector{Int}(undef, total_len)
    
    ptr = 1
    for (li, L) in enumerate(lines_temp)
        line_ptr[li] = ptr
        for idx in L
            line_idx[ptr] = idx
            ptr += 1
        end
    end
    line_ptr[m + 1] = ptr
    
    line_ptr, line_idx
end

# Legacy helper for count_violations (returns Vector{Vector{Int}} view-equivalent)
function _lines_from_csr(line_ptr::Vector{Int}, line_idx::Vector{Int})
    m = length(line_ptr) - 1
    lines = Vector{Vector{Int}}(undef, m)
    @inbounds for li in 1:m
        lines[li] = line_idx[line_ptr[li]:line_ptr[li+1]-1]
    end
    lines
end

# ============================================================================
# Energy computation with CSR lines (fused line processing)
# ============================================================================
function energy(u::AbstractVector, p::N3LProblem{T}) where {T}
    E_line = _energy_lines_csr(u, p.line_ptr, p.line_idx, p.nlines)
    E_bin = _energy_binary(u)
    E_mass = _energy_mass(u, p.k)
    E_box = _energy_box(u)
    p.α * E_line + p.β * E_bin + p.λ * E_mass + p.ρ * E_box
end

@inline function _energy_lines_csr(u, line_ptr, line_idx, nlines)
    E = zero(eltype(u))
    @inbounds for li in 1:nlines
        s = zero(eltype(u))
        for pi in line_ptr[li]:line_ptr[li+1]-1
            s += u[line_idx[pi]]
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

# ============================================================================
# Fused gradient computation with energy caching
# Computes gradient AND caches current energy in p.last_energy
# Single pass over lines: compute sum, accumulate energy, add gradient
# ============================================================================
function grad_energy!(p::N3LProblem{T}, u::AbstractVector) where {T}
    g = p.grad
    line_ptr = p.line_ptr
    line_idx = p.line_idx
    nlines = p.nlines
    linesums = p.linesums
    
    fill!(g, zero(T))
    
    # === Line penalty: fused sum + gradient + energy accumulation ===
    E_line = zero(T)
    @inbounds for li in 1:nlines
        s = zero(T)
        for pi in line_ptr[li]:line_ptr[li+1]-1
            s += u[line_idx[pi]]
        end
        linesums[li] = s  # cache for potential reuse
        z = s - 2
        if z > 0
            E_line += z * z
            c = p.α * 2 * z
            for pi in line_ptr[li]:line_ptr[li+1]-1
                g[line_idx[pi]] += c
            end
        end
    end
    
    # === Binary penalty ===
    E_bin = zero(T)
    @inbounds for i in eachindex(u)
        x = u[i]
        E_bin += x * x * (1 - x) * (1 - x)
        g[i] += p.β * (2x - 6x^2 + 4x^3)
    end
    
    # === Mass penalty ===
    mass_sum = zero(T)
    @inbounds for i in eachindex(u)
        mass_sum += u[i]
    end
    mass_diff = mass_sum - p.k
    E_mass = mass_diff * mass_diff
    mass_grad = p.λ * 2 * mass_diff
    @inbounds for i in eachindex(u)
        g[i] += mass_grad
    end
    
    # === Box penalty ===
    E_box = zero(T)
    @inbounds for i in eachindex(u)
        x = u[i]
        if x < 0
            E_box += x * x
            g[i] += p.ρ * 2x
        elseif x > 1
            d = x - 1
            E_box += d * d
            g[i] += p.ρ * 2d
        end
    end
    
    # Cache total energy for callback use
    p.last_energy = p.α * E_line + p.β * E_bin + p.λ * E_mass + p.ρ * E_box
    
    g
end

function rhs!(du, u, p::N3LProblem, t)
    grad_energy!(p, u)
    @inbounds @simd for i in eachindex(du)
        du[i] = -p.grad[i]
    end
    nothing
end

# ============================================================================
# Solver with DiscreteCallback for stall detection
# Uses cached energy from grad_energy! to avoid recomputation
# ============================================================================
function solve_n3l(p::N3LProblem{T};
                   tspan::Real = 3.0,
                   tol::Real = 1e-6,
                   patience::Real = 0.5,
                   check_dt::Real = 0.05,
                   seed::Integer = 0) where {T}
    
    rng = MersenneTwister(seed)
    u0 = rand(rng, T, p.n^2)
    
    # Initialize cached energy
    p.last_energy = energy(u0, p)
    state = ConvergenceState(p.last_energy)
    
    # Success callback: uses cached energy (updated after each RHS call)
    cb_success = DiscreteCallback(
        (u, t, int) -> int.p.last_energy ≤ tol,
        terminate!;
        save_positions = (false, false)
    )
    
    # Stall detection via DiscreteCallback with time-based condition
    last_check_t = Ref(zero(T))
    function stall_condition(u, t, int)
        t - last_check_t[] ≥ check_dt
    end
    function stall_affect!(int)
        last_check_t[] = int.t
        E = int.p.last_energy
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
    E_final = p.last_energy
    (u = u_final, E = E_final, best_E = state.best_energy, t = sol.t[end])
end

# ============================================================================
# Thresholding and violation counting (only called when E ≤ tol)
# ============================================================================
function threshold_board(u::AbstractVector, n::Int; thr=0.5)
    board = Matrix{Int}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        board[i, j] = u[(i-1)*n + j] > thr ? 1 : 0
    end
    board
end

function count_violations_csr(board::AbstractMatrix, line_ptr::Vector{Int}, line_idx::Vector{Int})
    flat = vec(board)
    nlines = length(line_ptr) - 1
    violations = 0
    @inbounds for li in 1:nlines
        s = 0
        for pi in line_ptr[li]:line_ptr[li+1]-1
            s += flat[line_idx[pi]]
        end
        if s >= 3
            violations += 1
        end
    end
    violations
end

# Legacy wrapper
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

# ============================================================================
# Sequential race search
# ============================================================================
function race_search(n::Int;
                     k::Int = 2n,
                     α_list = [50.0, 100.0, 200.0],
                     β_list = [2.0, 5.0, 10.0],
                     λ_list = [5.0, 10.0, 20.0],
                     ρ_list = [5.0, 10.0],
                     n_seeds::Int = 200,
                     master_seed::Int = 42,
                     tspan::Real = 3.0,
                     tol::Real = 1e-6,
                     patience::Real = 0.5,
                     check_dt::Real = 0.05,
                     verbose::Bool = false)
    
    # Pre-generate random seeds using master RNG (reproducible)
    master_rng = MersenneTwister(master_seed)
    random_seeds = rand(master_rng, UInt32, n_seeds)
    
    # Reusable problem instance
    base = N3LProblem(n; k, α=first(α_list), β=first(β_list), λ=first(λ_list), ρ=first(ρ_list))
    
    goal_solutions = Vector{NamedTuple{(:u, :E, :cfg, :board, :pts, :viol), 
                                        Tuple{Vector{Float64}, Float64, NamedTuple, Matrix{Int}, Int, Int}}}()
    
    cfgs = [(α, β, λ, ρ) for α in α_list for β in β_list for λ in λ_list for ρ in ρ_list]
    goal_configs = Set{Tuple{Float64,Float64,Float64,Float64}}()
    total_configs = length(cfgs)
    
    verbose && @printf("n=%d k=%d lines=%d configs=%d seeds=%d master_seed=%d\n",
                       n, k, base.nlines, length(cfgs), n_seeds, master_seed)
    
    for (ci, (α, β, λ, ρ)) in enumerate(cfgs)
        # Reuse problem, just update parameters
        update_params!(base, Float64(α), Float64(β), Float64(λ), Float64(ρ))
        
        verbose && @printf("\nConfig %d/%d: a=%.1f b=%.1f l=%.1f r=%.1f\n",
                           ci, length(cfgs), α, β, λ, ρ)
        
        for s in random_seeds
            out = solve_n3l(base; tspan, tol, patience, check_dt, seed=Int(s))
            
            # Only build board and count violations if energy is low enough
            if out.E ≤ tol
                board = threshold_board(out.u, n)
                npts = sum(board)
                nviol = count_violations_csr(board, base.line_ptr, base.line_idx)
                cfg = (α=α, β=β, λ=λ, ρ=ρ, seed=s, t=out.t)
                
                verbose && @printf("  seed=%u t=%.3f E=%.3e pts=%d viol=%d\n",
                                   s, out.t, out.E, npts, nviol)
                
                if nviol == 0 && npts == k
                    push!(goal_solutions, (u=copy(out.u), E=out.E, cfg=cfg, 
                                           board=board, pts=npts, viol=nviol))
                    push!(goal_configs, (α, β, λ, ρ))
                end
            else
                verbose && @printf("  seed=%u t=%.3f E=%.3e (skip)\n",
                                   s, out.t, out.E)
            end
        end
    end
    
    (solutions = goal_solutions, n = n, k = k, lines = base.nlines, 
     goal_configs = length(goal_configs), total_configs = total_configs,
     total_seeds = n_seeds)
end

# ============================================================================
# Parallel race search with worker-local problem reuse
# ============================================================================
struct Job
    config_id::Int
    α::Float64
    β::Float64
    λ::Float64
    ρ::Float64
    seed::UInt32  # Actual random seed value (not sequential index)
end

struct WorkerResult
    success::Bool
    u::Vector{Float64}
    E::Float64
    cfg::NamedTuple
    board::Matrix{Int}
    pts::Int
    viol::Int
end

function race_search_parallel(n::Int;
                               k::Int = 2n,
                               α_list = [50.0, 100.0, 200.0],
                               β_list = [2.0, 5.0, 10.0],
                               λ_list = [5.0, 10.0, 20.0],
                               ρ_list = [5.0, 10.0],
                               n_seeds::Int = 200,
                               master_seed::Int = 42,
                               tspan::Real = 3.0,
                               tol::Real = 1e-6,
                               patience::Real = 0.5,
                               check_dt::Real = 0.05,
                               verbose::Bool = false,
                               nworkers::Int = Threads.nthreads(),
                               early_stop::Bool = true)
    
    # Generate CSR lines once (shared read-only across workers)
    line_ptr, line_idx = _generate_lines_csr(n)
    nlines = length(line_ptr) - 1
    
    cfgs = [(α, β, λ, ρ) for α in α_list for β in β_list for λ in λ_list for ρ in ρ_list]
    total_configs = length(cfgs)
    total_jobs = total_configs * n_seeds
    
    # Pre-generate random seeds using master RNG (reproducible)
    master_rng = MersenneTwister(master_seed)
    random_seeds = rand(master_rng, UInt32, n_seeds)
    
    verbose && @printf("n=%d k=%d lines=%d configs=%d seeds=%d workers=%d total_jobs=%d\n",
                       n, k, nlines, total_configs, n_seeds, nworkers, total_jobs)
    
    # Thread-safe channels for job distribution and result collection
    jobs = Channel{Job}(min(1000, total_jobs))
    results = Channel{WorkerResult}(min(1000, total_jobs))
    
    # Atomic counters for progress tracking
    jobs_completed = Threads.Atomic{Int}(0)
    solutions_count = Threads.Atomic{Int}(0)
    solution_found = Threads.Atomic{Bool}(false)
    start_time = time()
    
    # Print start message (always, not just verbose)
    @printf("Starting: n=%d k=%d | %d configs × %d seeds = %d jobs | %d workers | master_seed=%d\n",
            n, k, total_configs, n_seeds, total_jobs, nworkers, master_seed)
    
    # Producer: enqueue all jobs
    producer = @async begin
        for (ci, (α, β, λ, ρ)) in enumerate(cfgs)
            early_stop && solution_found[] && break
            for s in random_seeds
                put!(jobs, Job(ci, α, β, λ, ρ, s))
            end
        end
        close(jobs)
    end
    
    # Worker threads (each with own reusable problem instance)
    workers = [@async worker_task_opt(jobs, results, n, k, line_ptr, line_idx, nlines,
                                       tspan, tol, patience, check_dt,
                                       jobs_completed, solutions_count, solution_found, 
                                       early_stop, total_jobs, start_time)
               for _ in 1:nworkers]
    
    # Result aggregator
    goal_solutions = Vector{NamedTuple{(:u, :E, :cfg, :board, :pts, :viol), 
                                        Tuple{Vector{Float64}, Float64, NamedTuple, Matrix{Int}, Int, Int}}}()
    goal_configs = Set{Tuple{Float64,Float64,Float64,Float64}}()
    
    aggregator = @async begin
        job_num = 0
        while !solution_found[] || isready(results)
            # Poll for result availability
            if !isready(results)
                sleep(0.001)
                continue
            end
            
            res = try
                take!(results)
            catch e
                if isa(e, InvalidStateException) || !isopen(results)
                    break
                end
                rethrow(e)
            end
            
            job_num += 1
            if res.success
                push!(goal_solutions, (u=res.u, E=res.E, cfg=res.cfg, 
                                       board=res.board, pts=res.pts, viol=res.viol))
                push!(goal_configs, (res.cfg.α, res.cfg.β, res.cfg.λ, res.cfg.ρ))
                Threads.atomic_add!(solutions_count, 1)
                # Print which job succeeded (shows actual random seed, not sequential index)
                @printf("\n✓ Job %d succeeded: α=%.0f β=%.0f λ=%.0f ρ=%.0f seed=%u E=%.2e\n",
                        job_num, res.cfg.α, res.cfg.β, res.cfg.λ, res.cfg.ρ, res.cfg.seed, res.E)
                early_stop && (solution_found[] = true)
            end
        end
    end
    
    # Wait for completion (with early stop support)
    if early_stop
        # When early_stop is enabled, don't wait for producer to finish all jobs
        # Just wait for workers and aggregator to complete after solution_found is set
        while !solution_found[]
            sleep(0.01)
        end
        # Give workers time to exit their polling loops
        sleep(0.1)
        # Close channels to unblock any waiting tasks
        close(jobs)
        close(results)
        # Wait for tasks with timeout
        @async (sleep(5); close(jobs); close(results))
        try
            wait(producer)
        catch
        end
        for w in workers
            try
                wait(w)
            catch
            end
        end
        try
            wait(aggregator)
        catch
        end
    else
        wait(producer)
        for w in workers
            wait(w)
        end
        close(results)
        wait(aggregator)
    end
    
    verbose && println("\nCompleted $(jobs_completed[]) jobs, found $(solutions_count[]) solutions")
    println()  # Clear the progress line
    
    (solutions = goal_solutions, n = n, k = k, lines = nlines, 
     goal_configs = length(goal_configs), total_configs = total_configs,
     total_seeds = n_seeds)
end

# Optimized worker: reuses a single N3LProblem instance per worker
function worker_task_opt(jobs, results, n, k, line_ptr, line_idx, nlines,
                          tspan, tol, patience, check_dt,
                          jobs_completed, solutions_count, solution_found, 
                          early_stop, total_jobs, start_time)
    # Create one problem instance per worker (with own grad/linesums buffers)
    p = N3LProblem{Float64}(n, k, line_ptr, line_idx, nlines,
                            50.0, 5.0, 10.0, 10.0,  # placeholder params
                            zeros(Float64, n^2),     # grad buffer
                            zeros(Float64, nlines),  # linesums buffer
                            0.0)                     # last_energy
    
    # Print progress every ~5% (20 updates total)
    progress_interval = max(1, div(total_jobs, 20))
    
    while !solution_found[]
        # Check early termination
        early_stop && solution_found[] && break
        
        # Poll for job availability (avoid blocking on take! which hangs after early_stop)
        if !isready(jobs)
            sleep(0.001)  # Small sleep to avoid busy-waiting
            continue
        end
        
        # Take job (should not block now)
        job = try
            take!(jobs)
        catch e
            if isa(e, InvalidStateException) || !isopen(jobs)
                break  # Channel closed, no more jobs
            end
            rethrow(e)
        end
        
        # Update parameters (reuse buffers)
        update_params!(p, job.α, job.β, job.λ, job.ρ)
        
        # Run ODE solve
        out = solve_n3l(p; tspan, tol, patience, check_dt, seed=job.seed)
        
        # Only build board and count violations if E ≤ tol
        cfg = (α=job.α, β=job.β, λ=job.λ, ρ=job.ρ, seed=job.seed, t=out.t)
        
        if out.E ≤ tol
            board = threshold_board(out.u, n)
            npts = sum(board)
            nviol = count_violations_csr(board, line_ptr, line_idx)
            success = nviol == 0 && npts == k
            # Set early stop flag immediately in worker (not aggregator) to avoid race
            if success && early_stop
                solution_found[] = true
            end
            put!(results, WorkerResult(success, copy(out.u), out.E, cfg, board, npts, nviol))
        else
            # Create empty board for non-successful result
            board = zeros(Int, n, n)
            put!(results, WorkerResult(false, out.u, out.E, cfg, board, 0, -1))
        end
        
        # Update progress
        completed = Threads.atomic_add!(jobs_completed, 1) + 1
        
        if completed % progress_interval == 0 || completed == total_jobs
            elapsed = time() - start_time
            rate = completed / elapsed
            eta = (total_jobs - completed) / rate
            @printf("\r[%d/%d] %.1f jobs/s | ETA %.0fs   ", completed, total_jobs, rate, eta)
        end
    end
end

# ============================================================================
# Solution analysis (unchanged output format)
# ============================================================================
function analyze_solution(result; file::Union{String,Nothing}=nothing, elapsed::Float64=0.0)
    nsol = length(result.solutions)
    
    if nsol == 0
        println("No valid solutions found")
        return nothing
    end
    
    sol = result.solutions[rand(1:nsol)]
    
    total_runs = result.total_configs * result.total_seeds
    
    @printf("n=%d k=%d | pts=%d viol=%d | E=%.2e | time=%.2fs\n", 
            result.n, result.k, sol.pts, sol.viol, sol.E, elapsed)
    @printf("Configs with valid solutions: %d/%d\n",
            result.goal_configs, result.total_configs)
    @printf("Valid solutions: %d/%d runs (%.1f%%)\n", 
            nsol, total_runs, 100*nsol/total_runs)
    display(sol.board)
    
    if file !== nothing
        timestamp = format(now(), "HHMMss")
        filename = "n_$(result.n)_$(timestamp).txt"
        open(filename, "w") do io
            println(io, "# n=$(result.n) k=$(result.k)")
            println(io, "# points=$(sol.pts) violations=$(sol.viol)")
            println(io, "# E=$(sol.E)")
            println(io, "# time=$(round(elapsed, digits=2))s")
            println(io, "# cfg: α=$(sol.cfg.α) β=$(sol.cfg.β) λ=$(sol.cfg.λ) ρ=$(sol.cfg.ρ)")
            println(io, "# seed=$(sol.cfg.seed) t=$(round(sol.cfg.t, digits=3))")
            println(io)
            for row in eachrow(sol.board)
                println(io, join(row, " "))
            end
            println(io)
            println(io, "# SUMMARY")
            println(io, "# Total configs tested: $(result.total_configs)")
            println(io, "# Seeds per config: $(result.total_seeds)")
            println(io, "# Total runs: $total_runs")
            println(io, "# Configs with valid solutions: $(result.goal_configs)")
            println(io, "# Total valid solutions: $nsol")
            println(io, "# Success rate: $(round(100*nsol/total_runs, digits=1))%")
        end
        println("\nSaved to: $filename")
    end
    
    sol
end

# ============================================================================
# Benchmark hook (activated by N3L_BENCH=1 environment variable)
# ============================================================================
function run_benchmark(n::Int; k::Int=2n, seed::Int=42)
    p = N3LProblem(n; k)
    
    # Warmup
    solve_n3l(p; seed=seed, tspan=0.1)
    
    # Timed run
    GC.gc()
    t0 = time_ns()
    allocs_before = Base.gc_live_bytes()
    
    out = solve_n3l(p; seed=seed, tspan=3.0)
    
    allocs_after = Base.gc_live_bytes()
    t1 = time_ns()
    
    elapsed_ms = (t1 - t0) / 1e6
    alloc_bytes = allocs_after - allocs_before
    
    @printf("[N3L_BENCH] n=%d solve_n3l: %.2f ms, ~%.1f KB alloc, E=%.2e, t=%.3f\n",
            n, elapsed_ms, alloc_bytes/1024, out.E, out.t)
    
    out
end

end # module

using .N3L

function main()
    n = 5
    n_seeds = 50
    master_seed = 42
    tspan = 3.0
    tol = 1e-6
    save_file = false
    verbose = false
    parallel = false
    nworkers = Threads.nthreads()
    early_stop = true
    
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "-n" && i < length(ARGS)
            n = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--seeds" && i < length(ARGS)
            n_seeds = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--master-seed" && i < length(ARGS)
            master_seed = parse(Int, ARGS[i+1])
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
        elseif arg == "-p" || arg == "--parallel"
            parallel = true
            i += 1
        elseif arg == "-j" && i < length(ARGS)
            nworkers = parse(Int, ARGS[i+1])
            i += 2
        elseif arg == "--early-stop"
            early_stop = true
            i += 1
        elseif arg == "-h" || arg == "--help"
            println("Usage: julia --project=. stats_sciml.jl [options]")
            println("  -n N          Grid size (default: 5)")
            println("  --seeds N     Number of seeds (default: 50)")
            println("  --master-seed N  Master seed for reproducibility (default: 42)")
            println("  --tspan T     Integration time (default: 3.0)")
            println("  --tol T       Energy tolerance (default: 1e-6)")
            println("  --file        Save best solution to n_N_HHMMSS.txt")
            println("  -v, --verbose Print per-seed progress")
            println("  -p, --parallel Use parallel execution with thread pool")
            println("  -j N          Number of parallel workers (default: Threads.nthreads()=$nworkers)")
            println("  --early-stop  Stop when first valid solution is found")
            return
        else
            i += 1
        end
    end
    
    # Benchmark hook (always runs with n=5 for consistent comparison)
    if get(ENV, "N3L_BENCH", "") == "1"
        N3L.run_benchmark(5)
        return
    end
    
    start_time = time()
    
    if parallel
        verbose && println("Running in PARALLEL mode with $nworkers workers")
        result = race_search_parallel(n; k=2n, n_seeds=n_seeds, master_seed=master_seed, tspan=tspan, tol=tol,
                                       verbose=verbose, nworkers=nworkers, early_stop=early_stop)
    else
        verbose && println("Running in SEQUENTIAL mode")
        result = race_search(n; k=2n, n_seeds=n_seeds, master_seed=master_seed, tspan=tspan, tol=tol, verbose=verbose)
    end
    
    elapsed = time() - start_time
    
    analyze_solution(result; file = save_file ? "" : nothing, elapsed=elapsed)
end

main()
