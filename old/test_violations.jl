using Random

push!(LOAD_PATH, ".")
include("stats_sciml.jl")
using .N3L

board = [
    0 0 1 0 0 0 0 0 0 0 0 1 0 0;
    0 0 0 0 0 0 0 0 0 0 1 0 1 0;
    0 0 0 0 0 0 0 0 1 0 0 0 1 0;
    0 1 0 0 0 0 0 0 0 0 1 0 0 0;
    1 0 0 0 0 0 0 0 0 1 0 0 0 0;
    0 0 0 0 0 0 0 0 1 1 0 0 0 0;
    0 0 0 0 0 0 1 0 0 0 0 1 0 0;
    0 1 0 0 0 1 0 0 0 0 0 0 0 0;
    0 0 0 1 0 1 0 0 0 0 0 0 0 0;
    0 0 0 0 0 0 1 0 0 0 0 0 0 1;
    1 0 0 1 0 0 0 0 0 0 0 0 0 0;
    0 0 0 0 1 0 0 0 0 0 0 0 0 1;
    0 0 1 0 0 0 0 1 0 0 0 0 0 0;
    0 0 0 0 1 0 0 1 0 0 0 0 0 0
]

n = 14

line_ptr, line_idx = N3L._generate_lines_csr(n)
println("Total lines: ", length(line_ptr) - 1)

nviol = N3L.count_violations_csr(board, line_ptr, line_idx)
println("Total violations: ", nviol)
