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

rational_interval(p, q) = interval(BigFloat(p)) / interval(BigFloat(q))

function run_matlab_verified_eigenvalue()
    env = copy(ENV)
    get!(env, "BLAS_VERSION", MATLAB_OPENBLAS_PATH)
    script = joinpath(@__DIR__, "compute_verified_eigenvalue.m")
    run(setenv(`matlab -nodesktop -nosplash -nodisplay -batch $("run('$script')")`, env))
end

function read_lambda_delta()
    vars = read_mat4_variables(VERIFIED_EIGENVALUE_PATH)
    return vars["lambda_lb"][1]
end

function compute_lambda1_lower_bound(lambda_delta, bounds)
    mu = BoundIO.parse_bound_value(bounds["curvature_bounds"]["ricci_lower_bound"])
    delta = rational_interval(numerator(DEFAULT_DELTA), denominator(DEFAULT_DELTA))
    one = interval(BigFloat(1))
    two = interval(BigFloat(2))
    three = interval(BigFloat(3))
    seven = interval(BigFloat(7))

    inf(lambda_delta) > 0 || error("Inset eigenvalue lower bound must be positive")
    inf(mu) > 0 || error("Ricci lower bound must be positive")
    inf(delta) > 0 || error("Delta positivity assumption failed")
    inf(three - seven * delta) > 0 || error("Inset-comparison denominator failed")

    sqrt_7delta = sqrt(seven * delta)
    sqrt_7delta_over_3 = sqrt(seven * delta / three)
    c = one - sqrt_7delta_over_3 - sqrt_7delta / (two * interval(BigFloat, pi) * (three - seven * delta))
    b = sqrt_7delta_over_3 * three / (two * mu)

    inf(c) > 0 || error("Inset-comparison denominator constant failed")
    return lambda_delta * c / (one + lambda_delta * b)
end

function update_verified_bounds(lambda_delta::Float64)
    bounds = JSON.parsefile(VERIFIED_BOUNDS_PATH)
    # lambda_delta is the left endpoint of the INTLAB interval, which is the
    # conservative lower bound for the inset eigenvalue.
    lambda_delta = interval(BigFloat(lambda_delta))
    lambda1_lower_bound = compute_lambda1_lower_bound(lambda_delta, bounds)

    bounds["lambda_1_delta_lower_bound"] = BoundIO.serialize_bound_value(lambda_delta)
    bounds["lambda_1_lower_bound"] = BoundIO.serialize_bound_value(BigFloat(inf(lambda1_lower_bound)))

    open(VERIFIED_BOUNDS_PATH, "w") do io
        JSON.print(io, bounds, 4)
        println(io)
    end
end

function run_orchestration()
    pmesh = pdelta_mesh_data()
    metric_integrals = metric_integrals_on_pdelta_triangulation(pmesh.nodes, pmesh.triangles)
    assembly = assemble_d6_invariant_cr_matrices(pmesh.nodes, pmesh.triangles, metric_integrals)
    write_matlab_matrices(assembly)
    run_matlab_verified_eigenvalue()
    update_verified_bounds(read_lambda_delta())
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_orchestration()
end
