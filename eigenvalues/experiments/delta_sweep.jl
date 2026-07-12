using Dates
using IntervalArithmetic
using JSON
using LinearAlgebra
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "assemble_matrices.jl"))

const EXPERIMENT_ROOT = @__DIR__
const RUNS_ROOT = joinpath(EXPERIMENT_ROOT, "runs")
const BOUNDS_PATH = normpath(joinpath(EXPERIMENT_ROOT, "..", "..", "data", "verified_bounds.json"))

const DEFAULT_DELTA_LADDER = [
    1 // 20000,
    1 // 100000,
    1 // 500000,
    1 // 1000000,
    1 // 5000000,
    1 // 10000000,
    1 // 50000000,
    1 // 100000000,
    1 // 1000000000,
    1 // 10000000000,
    1 // 1000000000000,
]

interval_midpoint(x) = (inf(x) + sup(x)) / 2
interval_radius(x) = (sup(x) - inf(x)) / 2
rational_interval(x::Rational) = interval(BigFloat(numerator(x))) / interval(BigFloat(denominator(x)))

function exact_decimal(x)
    return @sprintf("%.77f", BigFloat(x))
end

function write_json(path, value)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, value, 4)
        println(io)
    end
end

function parse_rational(text::AbstractString)
    fields = split(text, '/')
    length(fields) == 2 || error("Expected an exact rational p/q, got '$text'")
    return parse(BigInt, fields[1]) // parse(BigInt, fields[2])
end

function parse_options(args)
    options = Dict{String,String}()
    flags = Set{String}()

    for arg in args
        startswith(arg, "--") || error("Unknown argument '$arg'")
        if occursin('=', arg)
            key, value = split(arg[3:end], '='; limit = 2)
            options[key] = value
        else
            push!(flags, arg[3:end])
        end
    end

    deltas = haskey(options, "deltas") ? parse_rational.(split(options["deltas"], ',')) : DEFAULT_DELTA_LADDER
    N = parse(Int, get(options, "N", "14"))
    deg = parse(Int, get(options, "deg", "8"))
    terms = parse(Int, get(options, "terms", "8"))
    skip_matlab = "skip-matlab" in flags

    isempty(deltas) && error("The delta ladder must not be empty")
    all(delta -> 0 < delta < 1, deltas) || error("Every delta must lie in (0,1)")
    N > 0 || error("N must be positive")
    deg >= 0 || error("deg must be nonnegative")
    terms >= 0 || error("terms must be nonnegative")

    return (; deltas, N, deg, terms, skip_matlab)
end

function run_slug(delta::Rational, N::Integer, deg::Integer, terms::Integer)
    return "delta_$(numerator(delta))_$(denominator(delta))_N$(N)_deg$(deg)_terms$(terms)"
end

function reciprocal_diagnostics(D, terms::Integer)
    d0 = (inf(D[1, 1]) + sup(D[1, 1])) / 2
    R = copy(D)
    R[1, 1] -= interval(d0)
    ratio = sup(poly_abs_bound(R)) / abs(d0)
    remainder = ratio < 1 ? ratio^(terms + 1) / (abs(d0) * (1 - ratio)) : BigFloat(Inf)
    return (; ratio, remainder)
end

function metric_integral_with_diagnostics(oracle, nodes, tri; deg::Integer, terms::Integer)
    xs = nodes[collect(tri), 1]
    ys = nodes[collect(tri), 2]
    xbox = hull(hull(xs[1], xs[2]), xs[3])
    ybox = hull(hull(ys[1], ys[2]), ys[3])
    xcheb = exact(2) * xbox - exact(1)
    ycheb = exact(2) * ybox + exact(1)

    C = local_metric_component_polynomials(oracle, xcheb, ycheb; deg)
    reciprocal = reciprocal_diagnostics(C.D, terms)
    reciprocal.ratio < 1 || error("Neumann series does not contract: ratio = $(reciprocal.ratio)")
    invD = reciprocal_polynomial_neumann(C.D; deg, terms)

    U = physical_metric_components((
        xx = integrate_local_polynomial_on_triangle(poly_mul(C.A11, invD), nodes, tri, xbox, ybox),
        xy = integrate_local_polynomial_on_triangle(poly_mul(C.A12, invD), nodes, tri, xbox, ybox),
        yx = integrate_local_polynomial_on_triangle(poly_mul(C.A12, invD), nodes, tri, xbox, ybox),
        yy = integrate_local_polynomial_on_triangle(poly_mul(C.A22, invD), nodes, tri, xbox, ybox),
    ))

    return U, reciprocal
end

function metric_integrals_with_diagnostics(pmesh; deg::Integer, terms::Integer)
    oracle = InverseMetricBoxOracleV2()
    outer_index = maximum(first(key) for key in pmesh.node_keys)
    integrals = Vector{Any}(undef, length(pmesh.triangles))
    ratios = Vector{BigFloat}(undef, length(pmesh.triangles))
    remainders = Vector{BigFloat}(undef, length(pmesh.triangles))
    layers = Vector{Int}(undef, length(pmesh.triangles))

    for (index, tri) in pairs(pmesh.triangles)
        integrals[index], reciprocal = metric_integral_with_diagnostics(
            oracle,
            pmesh.nodes,
            tri;
            deg,
            terms,
        )
        ratios[index] = reciprocal.ratio
        remainders[index] = reciprocal.remainder
        layers[index] = outer_index - maximum(first(pmesh.node_keys[node]) for node in tri)
    end

    return (; integrals, ratios, remainders, layers)
end

function component_width_diagnostics(integrals)
    absolute_radii = Float64[]
    scaled_radii = Float64[]
    worst_score = -Inf
    worst_triangle = 0
    worst_component = ""

    for (triangle, U) in pairs(integrals)
        mids = Dict(
            "xx" => abs(interval_midpoint(U.xx)),
            "xy" => abs(interval_midpoint(U.xy)),
            "yx" => abs(interval_midpoint(U.yx)),
            "yy" => abs(interval_midpoint(U.yy)),
        )
        energy_scale = sqrt(max(mids["xx"] * mids["yy"], eps(BigFloat)))

        for component in ("xx", "xy", "yx", "yy")
            value = getproperty(U, Symbol(component))
            radius = interval_radius(value)
            scale = component in ("xx", "yy") ? max(mids[component], eps(BigFloat)) : energy_scale
            score = Float64(radius / scale)
            push!(absolute_radii, Float64(radius))
            push!(scaled_radii, score)
            if score > worst_score
                worst_score = score
                worst_triangle = triangle
                worst_component = component
            end
        end
    end

    return Dict(
        "max_absolute_radius" => maximum(absolute_radii),
        "median_scaled_radius" => median(scaled_radii),
        "q95_scaled_radius" => quantile(scaled_radii, 0.95),
        "max_scaled_radius" => maximum(scaled_radii),
        "worst_triangle" => worst_triangle,
        "worst_component" => worst_component,
    )
end

function matrix_width_diagnostics(A, n::Integer)
    S = symmetrize_entries(A)
    diagonal = [abs(interval_midpoint(get(S, (i, i), interval(BigFloat(0))))) for i in 1:n]
    absolute_radii = Float64[]
    scaled_radii = Float64[]
    worst_entry = (0, 0)
    worst_score = -Inf

    for ((i, j), value) in S
        radius = interval_radius(value)
        scale = sqrt(max(diagonal[i] * diagonal[j], eps(BigFloat)))
        score = Float64(radius / scale)
        push!(absolute_radii, Float64(radius))
        push!(scaled_radii, score)
        if score > worst_score
            worst_score = score
            worst_entry = (i, j)
        end
    end

    return Dict(
        "stored_entries" => length(S),
        "max_absolute_radius" => maximum(absolute_radii),
        "median_scaled_radius" => median(scaled_radii),
        "q95_scaled_radius" => quantile(scaled_radii, 0.95),
        "max_scaled_radius" => maximum(scaled_radii),
        "worst_entry" => collect(worst_entry),
    )
end

function layer_diagnostics(ratios, remainders, layers)
    result = Dict{String,Any}()
    for layer in sort(unique(layers))
        indices = findall(==(layer), layers)
        layer_ratios = Float64.(ratios[indices])
        layer_remainders = Float64.(remainders[indices])
        result[string(layer)] = Dict(
            "triangle_count" => length(indices),
            "max_neumann_ratio" => maximum(layer_ratios),
            "median_neumann_ratio" => median(layer_ratios),
            "max_neumann_remainder" => maximum(layer_remainders),
        )
    end
    return result
end

function dense_midpoint(A, n::Integer)
    B = zeros(Float64, n, n)
    for ((i, j), value) in symmetrize_entries(A)
        B[i, j] = Float64(interval_midpoint(value))
    end
    return B
end

function midpoint_eigenvalue(assembly)
    K = Symmetric(dense_midpoint(assembly.stiffness, assembly.matrix_size))
    M = Symmetric(dense_midpoint(assembly.mass, assembly.matrix_size))
    values = eigen(K, M).values
    return values[2]
end

function read_mat4_variables(path)
    variables = Dict{String,Matrix{Float64}}()
    open(path, "r") do io
        while !eof(io)
            read(io, Int32)
            rows = read(io, Int32)
            cols = read(io, Int32)
            read(io, Int32)
            name_length = read(io, Int32)
            name = String(read(io, name_length - 1))
            read(io, UInt8)
            data = Vector{Float64}(undef, rows * cols)
            read!(io, data)
            variables[name] = reshape(data, rows, cols)
        end
    end
    return variables
end

function theorem_lower_bound(lambda_delta::Real, delta::Rational)
    bounds = JSON.parsefile(BOUNDS_PATH)
    mu = parse_bound_value(bounds["curvature_bounds"]["ricci_lower_bound"])
    lambda = interval(BigFloat(lambda_delta))
    delta_interval = rational_interval(delta)
    one = interval(BigFloat(1))
    two = interval(BigFloat(2))
    three = interval(BigFloat(3))
    seven = interval(BigFloat(7))

    sqrt_7delta = sqrt(seven * delta_interval)
    sqrt_7delta_over_3 = sqrt(seven * delta_interval / three)
    c = one - sqrt_7delta_over_3 - sqrt_7delta / (two * interval(BigFloat, pi) * (three - seven * delta_interval))
    b = sqrt_7delta_over_3 * three / (two * mu)
    return inf(lambda * c / (one + lambda * b))
end

function matlab_command(run_dir)
    expression = "addpath('$(replace(EXPERIMENT_ROOT, "'" => "''"))'); verify_eigenvalue('$(replace(run_dir, "'" => "''"))')"
    return `matlab -nodesktop -nosplash -nodisplay -batch $expression`
end

function run_matlab(run_dir)
    log_path = joinpath(run_dir, "matlab.log")
    open(log_path, "w") do io
        run(pipeline(matlab_command(run_dir); stdout = io, stderr = io))
    end
    return read_mat4_variables(joinpath(run_dir, "verified_eigenvalue.mat"))
end

function config_dictionary(delta, N, deg, terms, skip_matlab)
    return Dict(
        "delta" => "$(numerator(delta))/$(denominator(delta))",
        "delta_float" => Float64(delta),
        "N" => N,
        "polynomial_degree" => deg,
        "neumann_terms" => terms,
        "skip_matlab" => skip_matlab,
        "started_at" => string(now()),
    )
end

function run_one_experiment(delta; N::Integer, deg::Integer, terms::Integer, skip_matlab::Bool)
    slug = run_slug(delta, N, deg, terms)
    run_dir = joinpath(RUNS_ROOT, slug)
    mkpath(run_dir)
    config = config_dictionary(delta, N, deg, terms, skip_matlab)
    write_json(joinpath(run_dir, "config.json"), config)

    result = Dict{String,Any}(
        "status" => "assembling",
        "run" => slug,
        "config" => config,
    )
    write_json(joinpath(run_dir, "result.json"), result)

    println("[$(now())] assembling $slug")
    flush(stdout)
    assembly_start = time()
    pmesh = pdelta_mesh_data(; delta, N)
    metric = metric_integrals_with_diagnostics(pmesh; deg, terms)
    assembly = assemble_d6_invariant_cr_matrices(
        pmesh.nodes,
        pmesh.triangles,
        metric.integrals;
        delta,
        N,
    )
    assembly_seconds = time() - assembly_start

    write_matlab_matrices(assembly; dir = run_dir)
    midpoint_lambda = midpoint_eigenvalue(assembly)
    max_ratio, max_ratio_triangle = findmax(metric.ratios)

    result["status"] = skip_matlab ? "assembled" : "verifying"
    result["assembly_seconds"] = assembly_seconds
    result["triangle_count"] = length(pmesh.triangles)
    result["matrix_size"] = assembly.matrix_size
    result["midpoint_lambda_1_delta"] = midpoint_lambda
    result["metric_integrals"] = component_width_diagnostics(metric.integrals)
    result["stiffness_matrix"] = matrix_width_diagnostics(assembly.stiffness, assembly.matrix_size)
    result["mass_matrix"] = matrix_width_diagnostics(assembly.mass, assembly.matrix_size)
    result["neumann"] = Dict(
        "max_ratio" => Float64(max_ratio),
        "max_ratio_triangle" => max_ratio_triangle,
        "max_ratio_boundary_layer" => metric.layers[max_ratio_triangle],
        "max_remainder" => Float64(maximum(metric.remainders)),
        "by_boundary_layer" => layer_diagnostics(metric.ratios, metric.remainders, metric.layers),
    )
    write_json(joinpath(run_dir, "result.json"), result)

    if !skip_matlab
        println("[$(now())] verifying $slug with INTLAB")
        flush(stdout)
        matlab_start = time()
        variables = run_matlab(run_dir)
        matlab_wall_seconds = time() - matlab_start
        lambda_lb = variables["lambda_lb"][1]
        lambda_ub = variables["lambda_ub"][1]
        theorem_lb = theorem_lower_bound(lambda_lb, delta)

        result["status"] = "complete"
        result["matlab_wall_seconds"] = matlab_wall_seconds
        result["veigs_seconds"] = variables["elapsed"][1]
        result["lambda_1_delta_lower"] = exact_decimal(lambda_lb)
        result["lambda_1_delta_upper"] = exact_decimal(lambda_ub)
        result["lambda_1_delta_width"] = lambda_ub - lambda_lb
        result["midpoint_minus_lower"] = midpoint_lambda - lambda_lb
        result["lambda_1_lower_bound"] = exact_decimal(theorem_lb)
        println("[$(now())] complete $slug: inset lb = $(lambda_lb), final lb = $(Float64(theorem_lb))")
        flush(stdout)
    end

    result["finished_at"] = string(now())
    write_json(joinpath(run_dir, "result.json"), result)
    return result
end

function collected_results()
    results = Dict{String,Any}[]
    isdir(RUNS_ROOT) || return results

    for entry in readdir(RUNS_ROOT; join = true)
        result_path = joinpath(entry, "result.json")
        isfile(result_path) || continue
        push!(results, JSON.parsefile(result_path))
    end

    sort!(results; by = result -> (
        get(get(result, "config", Dict()), "N", 0),
        get(get(result, "config", Dict()), "polynomial_degree", 0),
        get(get(result, "config", Dict()), "neumann_terms", 0),
        -get(get(result, "config", Dict()), "delta_float", 0.0),
    ))
    return results
end

function write_summary(_results)
    results = collected_results()
    write_json(joinpath(RUNS_ROOT, "summary.json"), results)
    open(joinpath(RUNS_ROOT, "summary.csv"), "w") do io
        println(io, "delta,status,N,degree,terms,matrix_size,max_neumann_ratio,max_stiffness_scaled_radius,midpoint_lambda,inset_lower,inset_width,final_lower,assembly_seconds,veigs_seconds")
        for result in results
            config = result["config"]
            fields = [
                config["delta"],
                result["status"],
                string(config["N"]),
                string(config["polynomial_degree"]),
                string(config["neumann_terms"]),
                string(get(result, "matrix_size", "")),
                string(get(get(result, "neumann", Dict()), "max_ratio", "")),
                string(get(get(result, "stiffness_matrix", Dict()), "max_scaled_radius", "")),
                string(get(result, "midpoint_lambda_1_delta", "")),
                string(get(result, "lambda_1_delta_lower", "")),
                string(get(result, "lambda_1_delta_width", "")),
                string(get(result, "lambda_1_lower_bound", "")),
                string(get(result, "assembly_seconds", "")),
                string(get(result, "veigs_seconds", "")),
            ]
            println(io, join(fields, ','))
        end
    end
end

function run_sweep(options)
    mkpath(RUNS_ROOT)
    results = Dict{String,Any}[]

    for delta in options.deltas
        result = try
            run_one_experiment(
                delta;
                N = options.N,
                deg = options.deg,
                terms = options.terms,
                skip_matlab = options.skip_matlab,
            )
        catch error
            failed = Dict{String,Any}(
                "status" => "failed",
                "config" => config_dictionary(delta, options.N, options.deg, options.terms, options.skip_matlab),
                "error" => sprint(showerror, error, catch_backtrace()),
                "finished_at" => string(now()),
            )
            run_dir = joinpath(RUNS_ROOT, run_slug(delta, options.N, options.deg, options.terms))
            write_json(joinpath(run_dir, "result.json"), failed)
            println(stderr, "[$(now())] failed $(failed["config"]["delta"]): $(sprint(showerror, error))")
            flush(stderr)
            failed
        end
        push!(results, result)
        write_summary(results)
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sweep(parse_options(ARGS))
end
