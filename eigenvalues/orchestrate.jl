using IntervalArithmetic
using JSON

include(joinpath(@__DIR__, "assemble_matrices.jl"))
module BoundIO
using IntervalArithmetic
include(joinpath(@__DIR__, "..", "bound_residual", "util", "io.jl"))
end

const VERIFIED_EIGENVALUE_PATH = joinpath(@__DIR__, "verified_eigenvalue.mat")
const VERIFIED_BOUNDS_PATH = normpath(joinpath(@__DIR__, "..", "data", "verified_bounds.json"))
const MATLAB_OPENBLAS_PATH = "/Applications/MATLAB_R2026a.app/bin/maca64/libmwopenblas.dylib"

function read_mat4_variables(path)
    vars = Dict{String, Matrix{Float64}}()

    open(path, "r") do io
        while !eof(io)
            read(io, Int32)
            rows = read(io, Int32)
            cols = read(io, Int32)
            read(io, Int32)
            name_len = read(io, Int32)
            name = String(read(io, name_len - 1))
            read(io, UInt8)
            data = Vector{Float64}(undef, rows * cols)
            read!(io, data)
            vars[name] = reshape(data, rows, cols)
        end
    end

    return vars
end

function parse_decimal_interval(s)
    lower = setrounding(BigFloat, RoundDown) do
        parse(BigFloat, s)
    end
    upper = setrounding(BigFloat, RoundUp) do
        parse(BigFloat, s)
    end
    return interval(lower, upper)
end

rational_interval(p, q) = interval(BigFloat(p)) / interval(BigFloat(q))

function assemble_eigenvalue_matrices()
    nodes, triangles = get_pdelta_triangulation()
    metric_integrals = metric_integrals_on_pdelta_triangulation(nodes, triangles)
    assembly = assemble_d6_invariant_cr_matrices(nodes, triangles, metric_integrals)
    write_matlab_matrices(assembly)
    return assembly
end

function run_matlab_verified_eigenvalue()
    env = copy(ENV)
    get!(env, "BLAS_VERSION", MATLAB_OPENBLAS_PATH)
    script = joinpath(@__DIR__, "compute_verified_eigenvalue.m")
    run(setenv(`matlab -nodesktop -nosplash -nodisplay -batch $("run('$script')")`, env))
end

function read_lambda_1_delta_lower_bound()
    vars = read_mat4_variables(VERIFIED_EIGENVALUE_PATH)
    return vars["lambda_lb"][1]
end

function theorem_rhs_upper_bound(bounds)
    Kminus = parse_decimal_interval(bounds["curvature_bounds"]["ricci_lower_bound"])
    delta = rational_interval(numerator(DEFAULT_DELTA), denominator(DEFAULT_DELTA))
    lambda = interval(BigFloat(5))

    inf(Kminus) > sup(rational_interval(1, 3) - rational_interval(1, 10)) || error("Ricci lower bound assumption failed")
    inf(delta) > 0 || error("Delta positivity assumption failed")
    sup(delta) < inf(rational_interval(1, 10000)) || error("Delta smallness assumption failed")

    rhs = lambda * (
        1 +
        sqrt(rational_interval(175, 48)) *
        (lambda / (interval(BigFloat(2)) * Kminus) + interval(BigFloat(2))) *
        sqrt(delta)
    )

    return sup(rhs)
end

function update_verified_bounds(lambda_1_delta_lower_bound::Float64)
    bounds = JSON.parsefile(VERIFIED_BOUNDS_PATH)
    lhs = interval(BigFloat(lambda_1_delta_lower_bound))
    rhs_upper = theorem_rhs_upper_bound(bounds)

    inf(lhs) > rhs_upper || error("Theorem check failed")

    bounds["lambda_1_delta_lower_bound"] = BoundIO.serialize_bound_value(lhs)
    bounds["lambda_1_lower_bound"] = "5"

    open(VERIFIED_BOUNDS_PATH, "w") do io
        JSON.print(io, bounds, 4)
        println(io)
    end
end

function run_orchestration()
    assemble_eigenvalue_matrices()
    run_matlab_verified_eigenvalue()
    update_verified_bounds(read_lambda_1_delta_lower_bound())
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_orchestration()
end
