using IntervalArithmetic
using JSON

include(joinpath(@__DIR__, "util", "mat4.jl"))
include(joinpath(@__DIR__, "assemble_matrices.jl"))

"""
Load a checked lattice triangulation at inset `delta`.

`path` is a MAT-v4 file containing two-row integer `node_keys`, three-row
one-based `triangles`, and scalar `lattice_size`. Return physical interval
`nodes` together with the integer connectivity data.
"""
function load_triangulation(path, delta)
    data = read_mat4_variables(path)
    node_keys = [
        (Int(column[1]), Int(column[2]))
        for column in eachcol(data["node_keys"])
    ]
    triangles = [
        (Int(column[1]), Int(column[2]), Int(column[3]))
        for column in eachcol(data["triangles"])
    ]
    lattice_size = Int(only(data["lattice_size"]))
    physical_scale = (1 - delta) / lattice_size
    nodes = Matrix{Interval{BigFloat}}(undef, length(node_keys), 2)
    for (index, (x, y)) in pairs(node_keys)
        nodes[index, 1] = interval(BigFloat, x * physical_scale)
        nodes[index, 2] = interval(BigFloat, y * physical_scale)
    end
    return (; nodes, triangles, node_keys)
end

"""Assemble the hardcoded FEM problem."""
function run_orchestration()
    delta = 1 // 5_120_000
    metric_coefficient_cutoff = 20
    local_polynomial_degree = 8
    neumann_order = 8
    lower_matrix_search_steps = 40
    liu_constant_target = 21 // 2_000
    progress_interval = 10_000

    triangulation_path =
        joinpath(@__DIR__, "triangulation", "triangulation.mat")
    mesh = load_triangulation(triangulation_path, delta)
    oracle = InverseMetricOracle(metric_coefficient_cutoff)
    certificates = certify_elements(
        oracle,
        mesh,
        local_polynomial_degree,
        neumann_order,
        lower_matrix_search_steps,
        liu_constant_target,
        progress_interval,
    )
    assembly = assemble_matrices(mesh, certificates)
    write_matlab_matrices(assembly, @__DIR__)

    bounds_path = normpath(
        joinpath(@__DIR__, "..", "data", "verified_bounds.json"),
    )
    bounds = read_bounds_json(bounds_path)
    bounds["delta_inset"] = "$(numerator(delta))/$(denominator(delta))"
    bounds["Liu_FEM_comparison_constant"] =
        "$(numerator(liu_constant_target))/$(denominator(liu_constant_target))"
    open(bounds_path, "w") do io
        JSON.print(io, bounds, 4)
        println(io)
    end
    println("maximum certified Liu constant: $(assembly.liu_constant)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    setprecision(BigFloat, 256)
    run_orchestration()
end
