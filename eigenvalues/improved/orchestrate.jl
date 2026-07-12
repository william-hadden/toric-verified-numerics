using Dates
using IntervalArithmetic
using JSON
using Printf
using Statistics

if !isdefined(@__MODULE__, :assemble_improved_problem)
    include(joinpath(@__DIR__, "assemble_matrices.jl"))
end

module ImprovedBoundIO
using IntervalArithmetic
include(joinpath(@__DIR__, "..", "..", "bound_residual", "util", "io.jl"))
end

const IMPROVED_RUNS_ROOT = joinpath(@__DIR__, "runs")
const IMPROVED_BOUNDS_PATH = normpath(joinpath(@__DIR__, "..", "..", "data", "verified_bounds.json"))
const IMPROVED_VALUE_OPTIONS = Set([
    "delta",
    "N",
    "refinement-level",
    "boundary-layers",
    "corner-refinement-level",
    "corner-layers",
    "deg",
    "terms",
    "bisection-iterations",
    "box-subdivision-depth",
    "quality-liu-target",
    "quality-subdivision-depth",
    "tag",
])
const IMPROVED_FLAG_OPTIONS = Set(["parallel", "skip-matlab"])

function improved_parse_rational(text::AbstractString)
    fields = split(text, '/')
    length(fields) == 2 || error("Expected an exact rational p/q, got '$text'")
    return parse(BigInt, fields[1]) // parse(BigInt, fields[2])
end

function improved_parse_options(args)
    values = Dict{String,String}()
    flags = Set{String}()
    for argument in args
        startswith(argument, "--") || error("Unknown argument '$argument'")
        if occursin('=', argument)
            key, value = split(argument[3:end], '='; limit = 2)
            key in IMPROVED_VALUE_OPTIONS || error("Unknown option '--$key'")
            haskey(values, key) && error("Option '--$key' was supplied more than once")
            values[key] = value
        else
            key = argument[3:end]
            key in IMPROVED_FLAG_OPTIONS || error("Unknown flag '--$key'")
            key in flags && error("Flag '--$key' was supplied more than once")
            push!(flags, key)
        end
    end

    options = (;
        delta = improved_parse_rational(get(values, "delta", "1/10")),
        N = parse(Int, get(values, "N", "14")),
        refinement_level = parse(Int, get(values, "refinement-level", "0")),
        boundary_layers = parse(Int, get(values, "boundary-layers", "1")),
        corner_refinement_level = parse(Int, get(values, "corner-refinement-level", "0")),
        corner_layers = parse(Int, get(values, "corner-layers", "1")),
        deg = parse(Int, get(values, "deg", "8")),
        terms = parse(Int, get(values, "terms", "8")),
        bisection_iterations = parse(Int, get(values, "bisection-iterations", "80")),
        max_subdivision_depth = parse(Int, get(values, "box-subdivision-depth", "0")),
        quality_liu_target = haskey(values, "quality-liu-target") ?
            improved_parse_rational(values["quality-liu-target"]) : nothing,
        max_quality_subdivision_depth = parse(
            Int,
            get(values, "quality-subdivision-depth", "0"),
        ),
        parallel = "parallel" in flags,
        skip_matlab = "skip-matlab" in flags,
        tag = get(values, "tag", ""),
    )

    0 < options.delta < 3 // 7 || error("delta must lie in (0,3/7)")
    options.N > 0 || error("N must be positive")
    options.refinement_level >= 0 || error("refinement-level must be nonnegative")
    1 <= options.boundary_layers <= options.N || error("boundary-layers must lie in 1:N")
    options.corner_refinement_level >= 0 || error("corner-refinement-level must be nonnegative")
    1 <= options.corner_layers <= options.N || error("corner-layers must lie in 1:N")
    options.deg >= 0 || error("deg must be nonnegative")
    options.terms >= 0 || error("terms must be nonnegative")
    options.bisection_iterations > 0 || error("bisection-iterations must be positive")
    options.max_subdivision_depth >= 0 || error("box-subdivision-depth must be nonnegative")
    options.max_quality_subdivision_depth >= 0 ||
        error("quality-subdivision-depth must be nonnegative")
    if !isnothing(options.quality_liu_target)
        options.quality_liu_target > 0 || error("quality-liu-target must be positive")
        options.max_quality_subdivision_depth > 0 ||
            error("quality-subdivision-depth must be positive when quality-liu-target is set")
    end
    (isempty(options.tag) || occursin(r"^[A-Za-z0-9_-]+$", options.tag)) ||
        error("tag may contain only ASCII letters, digits, underscores, and hyphens")
    return options
end

function improved_run_slug(options)
    delta = options.delta
    stem = "delta_$(numerator(delta))_$(denominator(delta))" *
        "_N$(options.N)_r$(options.refinement_level)_layers$(options.boundary_layers)" *
        "_corner$(options.corner_refinement_level)_clayers$(options.corner_layers)" *
        "_deg$(options.deg)_terms$(options.terms)_bisect$(options.bisection_iterations)" *
        "_boxdepth$(options.max_subdivision_depth)"
    if !isnothing(options.quality_liu_target)
        target = options.quality_liu_target
        stem *= "_qtarget$(numerator(target))_$(denominator(target))" *
            "_qdepth$(options.max_quality_subdivision_depth)"
    end
    return isempty(options.tag) ? stem : "$(stem)_$(options.tag)"
end

function improved_write_json(path, value)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, value, 4)
        println(io)
    end
end

function improved_directed_decimal(value, direction::RoundingMode)
    target = BigFloat(value)
    candidate = direction == RoundDown ? prevfloat(target) : nextfloat(target)
    for _ in 1:8
        text = @sprintf("%.77f", candidate)
        decimal_lower = setrounding(BigFloat, RoundDown) do
            parse(BigFloat, text)
        end
        decimal_upper = setrounding(BigFloat, RoundUp) do
            parse(BigFloat, text)
        end
        direction == RoundDown && decimal_upper <= target && return text
        direction == RoundUp && decimal_lower >= target && return text
        candidate = direction == RoundDown ? prevfloat(candidate) : nextfloat(candidate)
    end
    error("Could not serialize a directed decimal endpoint")
end

improved_lower_decimal(value) = improved_directed_decimal(value, RoundDown)
improved_upper_decimal(value) = improved_directed_decimal(value, RoundUp)

function improved_read_mat4_variables(path)
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

function improved_matlab_command(run_dir)
    script_dir = replace(@__DIR__, "'" => "''")
    escaped_run_dir = replace(run_dir, "'" => "''")
    expression = "addpath('$script_dir'); verify_eigenvalue('$escaped_run_dir')"
    return `matlab -nodesktop -nosplash -nodisplay -batch $expression`
end

function improved_run_matlab(run_dir)
    log_path = joinpath(run_dir, "matlab.log")
    open(log_path, "w") do io
        run(pipeline(improved_matlab_command(run_dir); stdout = io, stderr = io))
    end
    return improved_read_mat4_variables(joinpath(run_dir, "verified_eigenvalue.mat"))
end

function improved_matrix_width_diagnostics(matrix, size_)
    symmetric = symmetrize_entries(matrix)
    diagonal = [
        abs((inf(get(symmetric, (i, i), interval(BigFloat(0)))) +
             sup(get(symmetric, (i, i), interval(BigFloat(0))))) / BigFloat(2))
        for i in 1:size_
    ]
    scaled_radii = Float64[]
    absolute_radii = Float64[]
    for ((i, j), value) in symmetric
        radius = (sup(value) - inf(value)) / BigFloat(2)
        scale = sqrt(max(diagonal[i] * diagonal[j], eps(BigFloat)))
        push!(absolute_radii, Float64(radius))
        push!(scaled_radii, Float64(radius / scale))
    end
    return Dict(
        "stored_entries" => length(symmetric),
        "max_absolute_radius" => maximum(absolute_radii),
        "median_scaled_radius" => median(scaled_radii),
        "max_scaled_radius" => maximum(scaled_radii),
    )
end

function improved_certificate_diagnostics(assembly)
    certificates = assembly.certificates
    theta = Float64[certificate.theta for certificate in certificates]
    alpha = Float64[certificate.alpha for certificate in certificates]
    margins = Float64[certificate.spd_margin for certificate in certificates]
    local_liu = Float64[certificate.liu_constant_upper for certificate in certificates]
    quality_depths = Int[
        certificate.quality_minimum_subdivision_depth for certificate in certificates
    ]
    neumann_ratios = Float64[certificate.diagnostics.neumann.ratio for certificate in certificates]
    leaf_counts = Int[certificate.diagnostics.leaf_count for certificate in certificates]
    leaf_depths = Int[certificate.diagnostics.max_leaf_depth for certificate in certificates]
    return Dict(
        "min_theta" => minimum(theta),
        "median_theta" => median(theta),
        "min_alpha" => minimum(alpha),
        "min_residual_spd_margin" => minimum(margins),
        "max_neumann_ratio" => maximum(neumann_ratios),
        "subcovered_triangle_count" => count(>(1), leaf_counts),
        "max_metric_leaf_count" => maximum(leaf_counts),
        "max_metric_leaf_depth" => maximum(leaf_depths),
        "liu_constant_upper" => Float64(assembly.liu_constant_upper),
        "worst_liu_triangle" => assembly.worst_liu_triangle,
        "worst_local_liu_constant" => maximum(local_liu),
        "quality_refined_triangle_count" => count(>(0), quality_depths),
        "max_quality_minimum_subdivision_depth" => maximum(quality_depths),
    )
end

function improved_rational_interval(value::Rational)
    return interval(BigFloat(numerator(value))) / interval(BigFloat(denominator(value)))
end

"""Apply the existing inset-to-compact comparison to a smooth inset bound."""
function improved_compact_lower_bound(lambda_smooth_inset_lower, delta::Rational)
    bounds = JSON.parsefile(IMPROVED_BOUNDS_PATH)
    mu = ImprovedBoundIO.parse_bound_value(bounds["curvature_bounds"]["ricci_lower_bound"])
    lambda = interval(BigFloat(lambda_smooth_inset_lower))
    delta_interval = improved_rational_interval(delta)
    one = interval(BigFloat(1))
    two = interval(BigFloat(2))
    three = interval(BigFloat(3))
    seven = interval(BigFloat(7))

    inf(lambda) > 0 || error("The smooth inset lower bound must be positive")
    inf(mu) > 0 || error("The Ricci lower bound must be positive")
    inf(three - seven * delta_interval) > 0 ||
        error("The inset-comparison denominator requires delta < 3/7")
    sqrt_7delta = sqrt(seven * delta_interval)
    sqrt_7delta_over_3 = sqrt(seven * delta_interval / three)
    c = one - sqrt_7delta_over_3 -
        sqrt_7delta / (two * interval(BigFloat, pi) * (three - seven * delta_interval))
    b = sqrt_7delta_over_3 * three / (two * mu)
    inf(c) > 0 || error("The inset-comparison numerator constant is not positive")
    return inf(lambda * c / (one + lambda * b))
end

function improved_config_dictionary(options)
    return Dict(
        "delta" => "$(numerator(options.delta))/$(denominator(options.delta))",
        "delta_float" => Float64(options.delta),
        "N" => options.N,
        "refinement_level" => options.refinement_level,
        "boundary_layers" => options.boundary_layers,
        "corner_refinement_level" => options.corner_refinement_level,
        "corner_layers" => options.corner_layers,
        "polynomial_degree" => options.deg,
        "neumann_terms" => options.terms,
        "bisection_iterations" => options.bisection_iterations,
        "box_subdivision_depth" => options.max_subdivision_depth,
        "quality_liu_target" => isnothing(options.quality_liu_target) ? nothing :
            "$(numerator(options.quality_liu_target))/$(denominator(options.quality_liu_target))",
        "quality_subdivision_depth" => options.max_quality_subdivision_depth,
        "parallel" => options.parallel,
        "julia_threads" => Threads.nthreads(),
        "skip_matlab" => options.skip_matlab,
        "tag" => options.tag,
        "started_at" => string(now()),
    )
end

"""Keyword arguments forwarded from a run configuration to the assembler."""
function improved_assembly_options(options)
    return (;
        delta = options.delta,
        N = options.N,
        refinement_level = options.refinement_level,
        boundary_layers = options.boundary_layers,
        corner_refinement_level = options.corner_refinement_level,
        corner_layers = options.corner_layers,
        deg = options.deg,
        terms = options.terms,
        bisection_iterations = options.bisection_iterations,
        max_subdivision_depth = options.max_subdivision_depth,
        quality_liu_target = options.quality_liu_target,
        max_quality_subdivision_depth = options.max_quality_subdivision_depth,
        parallel = options.parallel,
    )
end

function improved_run_configuration(options)
    slug = improved_run_slug(options)
    run_dir = joinpath(IMPROVED_RUNS_ROOT, slug)
    mkpath(run_dir)
    result = Dict{String,Any}(
        "status" => "assembling",
        "run" => slug,
        "config" => improved_config_dictionary(options),
    )
    improved_write_json(joinpath(run_dir, "result.json"), result)

    try
        println("[$(now())] assembling $slug")
        flush(stdout)
        assembly_started = time()
        problem = assemble_improved_problem(; improved_assembly_options(options)...)
        assembly_seconds = time() - assembly_started
        assembly = problem.assembly
        write_matlab_matrices(assembly; dir = run_dir)

        result["status"] = options.skip_matlab ? "assembled" : "verifying"
        result["assembly_seconds"] = assembly_seconds
        result["triangle_count"] = length(problem.mesh.triangles)
        result["node_count"] = size(problem.mesh.nodes, 1)
        result["matrix_size"] = assembly.matrix_size
        result["midpoint_lambda_fem"] = assembly.matrix_size <= 2000 ?
            improved_midpoint_eigenvalue(assembly) : nothing
        result["certificates"] = improved_certificate_diagnostics(assembly)
        result["stiffness_matrix"] = improved_matrix_width_diagnostics(
            assembly.stiffness,
            assembly.matrix_size,
        )
        result["mass_matrix"] = improved_matrix_width_diagnostics(
            assembly.mass,
            assembly.matrix_size,
        )
        improved_write_json(joinpath(run_dir, "result.json"), result)

        if !options.skip_matlab
            println("[$(now())] verifying the FEM eigenvalue with INTLAB")
            flush(stdout)
            matlab_started = time()
            variables = improved_run_matlab(run_dir)
            result["matlab_wall_seconds"] = time() - matlab_started
            result["veigs_seconds"] = variables["elapsed"][1]
            variables["lambda_fem_ind"][1] == 2 ||
                error("INTLAB did not certify the requested second generalized eigenvalue")

            lambda_fem_lower = variables["lambda_fem_lb"][1]
            lambda_fem_upper = variables["lambda_fem_ub"][1]
            lambda_smooth_inset_lower = improved_liu_lower_bound(
                lambda_fem_lower,
                assembly.liu_constant_upper,
            )
            lambda_compact_lower = improved_compact_lower_bound(
                lambda_smooth_inset_lower,
                options.delta,
            )

            result["lambda_fem_lower"] = improved_lower_decimal(lambda_fem_lower)
            result["lambda_fem_upper"] = improved_upper_decimal(lambda_fem_upper)
            result["lambda_fem_width"] = lambda_fem_upper - lambda_fem_lower
            result["lambda_smooth_inset_lower"] = improved_lower_decimal(lambda_smooth_inset_lower)
            result["lambda_compact_lower"] = improved_lower_decimal(lambda_compact_lower)
            result["status"] = "complete"
            println(
                "[$(now())] complete: FEM lb = $lambda_fem_lower, " *
                "smooth inset lb = $(Float64(lambda_smooth_inset_lower)), " *
                "compact lb = $(Float64(lambda_compact_lower))",
            )
        end

        result["finished_at"] = string(now())
        improved_write_json(joinpath(run_dir, "result.json"), result)
        return result
    catch exception
        result["status"] = "failed"
        result["error"] = sprint(showerror, exception, catch_backtrace())
        result["finished_at"] = string(now())
        improved_write_json(joinpath(run_dir, "result.json"), result)
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    improved_run_configuration(improved_parse_options(ARGS))
end
