using IntervalArithmetic
using JSON
using Printf

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

"""
Serialize `value` with `digits` places after the decimal point.

Return a decimal string whose exact value is at least `value`.
"""
function directed_upper_decimal(value, digits)
    target = BigFloat(value)
    isfinite(target) || error("Cannot serialize a non-finite upper bound")
    candidate = nextfloat(target)
    while true
        text = @sprintf("%.*f", digits, candidate)
        parsed_lower = setrounding(BigFloat, RoundDown) do
            parse(BigFloat, text)
        end
        parsed_lower >= target && return text
        candidate = nextfloat(candidate)
    end
end

"""
Write `delta` and an upper bound for `comparison_constant` to JSON `path`.

The inset is stored exactly as a rational string and the comparison constant
as an upward-rounded decimal with `digits` places. Any eigenvalue bounds from
an earlier matrix assembly are removed. Return nothing.
"""
function write_comparison_constants(path, delta, comparison_constant, digits)
    bounds = JSON.parsefile(path)
    bounds["delta_inset"] = "$(numerator(delta))/$(denominator(delta))"
    bounds["Liu_FEM_comparison_constant"] =
        directed_upper_decimal(comparison_constant, digits)
    delete!(bounds, "lambda_1_delta_lower_bound")
    delete!(bounds, "lambda_1_lower_bound")
    open(path, "w") do io
        JSON.print(io, bounds, 4)
        println(io)
    end
end

"""Invalidate the old certificate and assemble the hardcoded FEM problem."""
function run_orchestration()
    setprecision(BigFloat, 256) do
        rm(joinpath(@__DIR__, "stiff_matrix.mat"); force = true)
        rm(joinpath(@__DIR__, "mass_matrix.mat"); force = true)
        rm(joinpath(@__DIR__, "verified_eigenvalue.mat"); force = true)
        delta = 1 // 5_120_000
        metric_coefficient_cutoff = 20
        local_polynomial_degree = 8
        neumann_order = 8
        lower_matrix_search_steps = 40
        quality_target = 21 // 2_000
        progress_interval = 10_000
        decimal_digits = 77

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
            quality_target,
            progress_interval,
        )
        assembly = assemble_matrices(mesh, certificates)
        write_matlab_matrices(assembly, @__DIR__)

        bounds_path = normpath(
            joinpath(@__DIR__, "..", "data", "verified_bounds.json"),
        )
        write_comparison_constants(
            bounds_path,
            delta,
            assembly.liu_constant,
            decimal_digits,
        )
        println("Liu FEM comparison constant: $(assembly.liu_constant)")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_orchestration()
end
