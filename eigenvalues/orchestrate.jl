using IntervalArithmetic
using JSON
using Printf

include(joinpath(@__DIR__, "assemble_matrices.jl"))

"""Read all numeric variables from a MATLAB version-4 binary file."""
function read_mat4_variables(path)
    variables = Dict{String,Matrix{Float64}}()
    open(path, "r") do io
        while !eof(io)
            read(io, Int32)
            rows = read(io, Int32)
            columns = read(io, Int32)
            read(io, Int32)
            name_length = read(io, Int32)
            name = String(read(io, name_length - 1))
            read(io, UInt8)
            data = Vector{Float64}(undef, rows * columns)
            read!(io, data)
            variables[name] = reshape(data, rows, columns)
        end
    end
    return variables
end

"""Run the fixed MATLAB/INTLAB verifier in the eigenvalue directory."""
function run_matlab_verifier()
    script = replace(joinpath(@__DIR__, "verify_eigenvalue.m"), "'" => "''")
    run(`matlab -nodesktop -nosplash -nodisplay -batch $("run('$script')")`)
end

"""Validate and return the certified FEM eigenvalue endpoints from MATLAB."""
function verified_fem_endpoints(path)
    variables = read_mat4_variables(path)
    required = ("lambda_fem_lb", "lambda_fem_ub", "lambda_fem_ind")
    all(name -> haskey(variables, name), required) ||
        error("The MATLAB result is missing a verified eigenvalue variable")
    all(name -> size(variables[name]) == (1, 1), required) ||
        error("The verified eigenvalue variables must be scalars")
    lower = variables["lambda_fem_lb"][1]
    upper = variables["lambda_fem_ub"][1]
    index = variables["lambda_fem_ind"][1]
    all(isfinite, (lower, upper, index)) || error("The verified eigenvalue data are not finite")
    index == 2 || error("INTLAB did not certify generalized eigenvalue 2")
    lower <= upper || error("The verified eigenvalue endpoints are reversed")
    return lower, upper
end

"""Embed an exact rational number in an outward-rounded BigFloat interval."""
rational_interval(value::Rational) =
    interval(BigFloat(numerator(value))) / interval(BigFloat(denominator(value)))

"""Apply the inset-to-compact comparison to a smooth inset lower bound."""
function compact_lower_bound(lambda_inset_lower, delta, ricci_lower_bound)
    lambda = interval(BigFloat(lambda_inset_lower))
    delta_interval = rational_interval(delta)
    mu = parse_bound_value(ricci_lower_bound)
    one, two = interval(BigFloat(1)), interval(BigFloat(2))
    three, seven = interval(BigFloat(3)), interval(BigFloat(7))
    inf(mu) > 0 || error("The Ricci lower bound must be positive")
    inf(delta_interval) > 0 || error("The inset parameter must be positive")
    inf(three - seven * delta_interval) > 0 ||
        error("The inset parameter must be smaller than 3/7")
    square_root = sqrt(seven * delta_interval)
    boundary_fraction = sqrt(seven * delta_interval / three)
    c = one - boundary_fraction -
        square_root / (two * interval(BigFloat, pi) * (three - seven * delta_interval))
    b = boundary_fraction * three / (two * mu)
    inf(c) > 0 || error("The compact-comparison numerator is not positive")
    denominator = one + lambda * b
    inf(denominator) > 0 || error("The compact-comparison denominator is not positive")
    return inf(lambda * c / denominator)
end

"""Serialize a certified lower bound as a decimal which rounds downward."""
function directed_lower_decimal(value, digits)
    target = BigFloat(value)
    isfinite(target) || error("Cannot serialize a non-finite lower bound")
    candidate = prevfloat(target)
    while true
        text = @sprintf("%.*f", digits, candidate)
        parsed_upper = setrounding(BigFloat, RoundUp) do
            parse(BigFloat, text)
        end
        parsed_upper <= target && return text
        candidate = prevfloat(candidate)
    end
end

"""Replace only the two canonical eigenvalue bounds in the verified JSON data."""
function write_eigenvalue_bounds(path, bounds, inset_lower, compact_lower, decimal_digits)
    bounds["lambda_1_delta_lower_bound"] =
        directed_lower_decimal(inset_lower, decimal_digits)
    bounds["lambda_1_lower_bound"] =
        directed_lower_decimal(compact_lower, decimal_digits)
    open(path, "w") do io
        JSON.print(io, bounds, 4)
        println(io)
    end
end

"""Run the fixed rigorous FEM, Liu, and compact-manifold comparison pipeline."""
function run_orchestration()
    setprecision(BigFloat, 256) do
        delta = 1 // 5000
        base_resolution = 20
        boundary_refinements = 3
        boundary_layers = 1
        corner_refinements = 2
        corner_layers = 1
        metric_coefficient_degree = 30
        polynomial_degree = 8
        neumann_terms = 8
        bisection_steps = 70
        quality_target = 7 // 50
        progress_interval = 100
        decimal_digits = 77

        mesh = sector_mesh(
            delta,
            base_resolution,
            boundary_refinements,
            boundary_layers,
            corner_refinements,
            corner_layers,
        )
        oracle = InverseMetricOracle(metric_coefficient_degree)
        certificates = certify_elements(
            oracle,
            mesh,
            polynomial_degree,
            neumann_terms,
            bisection_steps,
            quality_target,
            progress_interval,
        )
        assembly = assemble_matrices(mesh, certificates)
        write_matlab_matrices(assembly, @__DIR__)

        run_matlab_verifier()
        result_path = joinpath(@__DIR__, "verified_eigenvalue.mat")
        lambda_fem_lower, lambda_fem_upper = verified_fem_endpoints(result_path)
        lambda_inset_lower = liu_lower_bound(lambda_fem_lower, assembly.liu_constant)

        bounds_path = normpath(joinpath(@__DIR__, "..", "data", "verified_bounds.json"))
        bounds = JSON.parsefile(bounds_path)
        lambda_compact_lower = compact_lower_bound(
            lambda_inset_lower,
            delta,
            bounds["curvature_bounds"]["ricci_lower_bound"],
        )
        write_eigenvalue_bounds(
            bounds_path,
            bounds,
            lambda_inset_lower,
            lambda_compact_lower,
            decimal_digits,
        )
        maximum_cover_depth = maximum(
            certificate.forced_depth for certificate in certificates
        )
        println("FEM eigenvalue enclosure: [$lambda_fem_lower, $lambda_fem_upper]")
        println("Liu comparison constant: $(assembly.liu_constant)")
        println("maximum forced metric-cover depth: $maximum_cover_depth")
        println("smooth inset lower bound: $lambda_inset_lower")
        println("compact-manifold lower bound: $lambda_compact_lower")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_orchestration()
end
