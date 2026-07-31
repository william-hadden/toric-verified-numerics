using IntervalArithmetic
using Logging

global_logger(NullLogger())

include(joinpath(@__DIR__, "bound_residual", "util", "io.jl"))
include(joinpath(@__DIR__, "bound_residual", "util", "chebyshev_algebra.jl"))
include(joinpath(@__DIR__, "bound_residual", "util", "problem.jl"))
include(joinpath(@__DIR__, "bound_metric", "util", "progress.jl"))
include(joinpath(@__DIR__, "bound_metric", "util", "interval_helpers.jl"))
include(joinpath(@__DIR__, "bound_metric", "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "bound_metric", "util", "inverse_bounds.jl"))
include(joinpath(@__DIR__, "bound_metric", "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "bound_metric", "util", "chebyshev_interval_evaluation.jl"))
include(joinpath(@__DIR__, "bound_metric", "util", "inverse_subdivision_bounds.jl"))
include(joinpath(@__DIR__, "bound_metric", "util", "curvature_bounds.jl"))

const DEFAULT_PRECISION = 256
const INVERSE_PDEG = 20
const INVERSE_SUBDIVISIONS = 8
const RICCI_PDEG = 10
const RICCI_SUBDIVISIONS = 4
const RICCI_LOWER_PDEG = 20
const RICCI_LOWER_SUBDIVISIONS = 30
const RIEMANN_PDEG = 10
const RIEMANN_SUBDIVISIONS = 4
const NABLA_RIEMANN_PDEG = 10
const NABLA_RIEMANN_SUBDIVISIONS = 4
const NABLA2_RIEMANN_PDEG = 10
const NABLA2_RIEMANN_SUBDIVISIONS = 4

const INVERSE_DERIVATIVE_PRODUCT_STEPS = 144
const RICCI_NUMERATOR_PRODUCT_STEPS = 65
const RICCI_LOWER_NUMERATOR_PRODUCT_STEPS = 152
const RIEMANN_NUMERATOR_PRODUCT_STEPS = 136
const NABLA_RIEMANN_NUMERATOR_PRODUCT_STEPS = 936
const NABLA_NABLA_RIEMANN_NUMERATOR_PRODUCT_STEPS = 8132

const METRIC_COMPONENTS = Dict(
    "xx" => (1, 1),
    "xy" => (1, 2),
    "yx" => (2, 1),
    "yy" => (2, 2),
)

const FIRST_DERIVATIVES = Dict(
    "x" => (1, 0),
    "y" => (0, 1),
)

const SECOND_DERIVATIVES = Dict(
    "xx" => (2, 0),
    "xy" => (1, 1),
    "yx" => (1, 1),
    "yy" => (0, 2),
)

function bound_inverse_derivatives_by_local_subdivision(
    inverse_derivative_coeffs;
    pdeg::Integer,
    nx::Integer,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = prepare_inverse_deriv_coeffs_for_subdivision(inverse_derivative_coeffs, pdeg)
    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    derivative_bounds = Dict{Tuple{Int, Int, Tuple{Int, Int}}, BigFloat}()
    for key in keys(trunc.deriv_num)
        derivative_bounds[key] = big"0"
    end

    global_D_lower = big"Inf"

    for xbox in xboxes, ybox in yboxes
        D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)
        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error("Could not certify positivity of D for inverse derivative bounds: D_box = $D_box")
        end

        for ((i, j, a), trunc_coeffs) in trunc.deriv_num
            num_box = local_coeff_sum_centered_enclosure_with_tail(trunc_coeffs, xbox, ybox)
            denom_power = a[1] + a[2] + 1
            derivative_box = num_box / (D_box^denom_power)
            derivative_bounds[(i, j, a)] =
                max(derivative_bounds[(i, j, a)], sup(abs(derivative_box)))
        end

        global_D_lower = min(global_D_lower, D_lower)
        advance_progress!(progress)
    end

    return (; derivative_bounds, D_lower = global_D_lower)
end

function metric_inverse_json(derivative_bounds)
    value = Dict{String,Any}()
    for (label, (i, j)) in METRIC_COMPONENTS
        value[label] = serialize_bound_value(derivative_bounds[(i, j, (0, 0))])
    end

    d1 = Dict{String,Any}()
    for (derivative_label, a) in FIRST_DERIVATIVES
        entries = Dict{String,Any}()
        for (component_label, (i, j)) in METRIC_COMPONENTS
            entries[component_label] = serialize_bound_value(derivative_bounds[(i, j, a)])
        end
        d1[derivative_label] = entries
    end

    d2 = Dict{String,Any}()
    for (derivative_label, a) in SECOND_DERIVATIVES
        entries = Dict{String,Any}()
        for (component_label, (i, j)) in METRIC_COMPONENTS
            entries[component_label] = serialize_bound_value(derivative_bounds[(i, j, a)])
        end
        d2[derivative_label] = entries
    end

    return Dict{String,Any}(
        "value" => value,
        "d1" => d1,
        "d2" => d2,
    )
end

function update_verified_bounds_json(metric_inverse, curvature_bounds; path::AbstractString = VERIFIED_BOUNDS_PATH)
    bounds = isfile(path) ? JSON.parsefile(path) : Dict{String,Any}()
    bounds["metric_inverse"] = metric_inverse
    bounds["curvature_bounds"] = curvature_bounds

    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, bounds, 4)
        println(io)
    end

    return nothing
end

function run_verified_bounds_pipeline(;
    coeffs_path::AbstractString = U0_PATH,
    inverse_pdeg::Integer = INVERSE_PDEG,
    inverse_subdivisions::Integer = INVERSE_SUBDIVISIONS,
    ricci_pdeg::Integer = RICCI_PDEG,
    ricci_subdivisions::Integer = RICCI_SUBDIVISIONS,
    ricci_lower_pdeg::Integer = RICCI_LOWER_PDEG,
    ricci_lower_subdivisions::Integer = RICCI_LOWER_SUBDIVISIONS,
    riemann_pdeg::Integer = RIEMANN_PDEG,
    riemann_subdivisions::Integer = RIEMANN_SUBDIVISIONS,
    nabla_riemann_pdeg::Integer = NABLA_RIEMANN_PDEG,
    nabla_riemann_subdivisions::Integer = NABLA_RIEMANN_SUBDIVISIONS,
    nabla2_riemann_pdeg::Integer = NABLA2_RIEMANN_PDEG,
    nabla2_riemann_subdivisions::Integer = NABLA2_RIEMANN_SUBDIVISIONS,
    output_path::AbstractString = VERIFIED_BOUNDS_PATH,
)
    load_progress = start_progress("Loading metric coefficients", filesize(coeffs_path))
    step1 = load_rational_coeffs_csv(coeffs_path; progress = load_progress)
    finish_progress!(load_progress)

    println("Computing second derivatives ...")
    step2 = compute_second_derivatives(step1)

    step3_progress = start_progress("Build inverse metric coefficients", 74)
    step3 = build_inverse_metric_coeffs(step2; progress = step3_progress)
    finish_progress!(step3_progress)

    println()
    println("Computing inverse metric value, first derivative, and second derivative bounds ...")
    inverse_num_progress =
        start_progress("Inverse derivative numerator products", INVERSE_DERIVATIVE_PRODUCT_STEPS)
    inverse_derivative_coeffs =
        compute_inverse_derivative_numerator_components_truncated_coeff_space(
            step3;
            k = 2,
            pdeg = inverse_pdeg,
            progress = inverse_num_progress,
        )
    finish_progress!(inverse_num_progress)

    inverse_box_progress =
        start_progress("Inverse derivative subdivision boxes", inverse_subdivisions * inverse_subdivisions)
    inverse_bounds = bound_inverse_derivatives_by_local_subdivision(
        inverse_derivative_coeffs;
        pdeg = inverse_pdeg,
        nx = inverse_subdivisions,
        ny = inverse_subdivisions,
        progress = inverse_box_progress,
    )
    finish_progress!(inverse_box_progress)

    println()
    println("Computing Ricci bound ...")
    ricci_progress = start_progress("Ricci numerator products", RICCI_NUMERATOR_PRODUCT_STEPS)
    ricci_coeffs =
        compute_ricci_numerators_truncated_coefficient_space(
            step3;
            pdeg = ricci_pdeg,
            progress = ricci_progress,
        )
    finish_progress!(ricci_progress)

    ricci_box_progress =
        start_progress("Ricci subdivision boxes", ricci_subdivisions * ricci_subdivisions)
    ricci_bounds = compute_ricci_bound_by_local_subdivision_truncated(
        ricci_coeffs;
        pdeg = ricci_pdeg,
        nx = ricci_subdivisions,
        ny = ricci_subdivisions,
        progress = ricci_box_progress,
    )
    finish_progress!(ricci_box_progress)

    println()
    println("Computing Ricci lower bound ...")
    ricci_lower_progress =
        start_progress("Ricci-minus-identity numerator products", RICCI_LOWER_NUMERATOR_PRODUCT_STEPS)
    ricci_lower_coeffs =
        compute_ricci_minus_identity_numerator_truncated_coefficient_space(
            step3;
            pdeg = ricci_lower_pdeg,
            progress = ricci_lower_progress,
        )
    finish_progress!(ricci_lower_progress)

    ricci_lower_box_progress =
        start_progress(
            "Ricci lower-bound subdivision boxes",
            ricci_lower_subdivisions * ricci_lower_subdivisions,
        )
    ricci_lower_bounds = compute_ricci_minus_identity_bound_by_local_subdivision_truncated(
        ricci_lower_coeffs;
        pdeg = ricci_lower_pdeg,
        nx = ricci_lower_subdivisions,
        ny = ricci_lower_subdivisions,
        progress = ricci_lower_box_progress,
    )
    finish_progress!(ricci_lower_box_progress)

    println()
    println("Computing Riemann bound ...")
    riem_progress = start_progress("Riemann numerator products", RIEMANN_NUMERATOR_PRODUCT_STEPS)
    riem_coeffs =
        compute_riem_numerators_truncated_coeff_space(
            step3;
            pdeg = riemann_pdeg,
            progress = riem_progress,
        )
    finish_progress!(riem_progress)

    riem_box_progress =
        start_progress("Riemann subdivision boxes", riemann_subdivisions * riemann_subdivisions)
    riem_bounds = compute_riem_by_local_subdivision_truncated(
        riem_coeffs;
        pdeg = riemann_pdeg,
        nx = riemann_subdivisions,
        ny = riemann_subdivisions,
        progress = riem_box_progress,
    )
    finish_progress!(riem_box_progress)

    println()
    println("Computing covariant Riemann bound ...")
    cov_riem_progress =
        start_progress("Covariant Riemann numerator products", NABLA_RIEMANN_NUMERATOR_PRODUCT_STEPS)
    cov_riem_coeffs =
        compute_inverse_derivative_numerator_components_truncated_coeff_space(
            step3;
            k = 3,
            pdeg = nabla_riemann_pdeg,
            progress = cov_riem_progress,
        )
    finish_progress!(cov_riem_progress)

    cov_riem_box_progress =
        start_progress("Covariant Riemann subdivision boxes", nabla_riemann_subdivisions * nabla_riemann_subdivisions)
    cov_riem_bounds = compute_cov_riem_by_local_subdivision_truncated(
        cov_riem_coeffs;
        pdeg = nabla_riemann_pdeg,
        nx = nabla_riemann_subdivisions,
        ny = nabla_riemann_subdivisions,
        progress = cov_riem_box_progress,
    )
    finish_progress!(cov_riem_box_progress)

    println()
    println("Computing second covariant Riemann bound ...")
    cov_cov_riem_progress =
        start_progress("Second covariant Riemann numerator products", NABLA_NABLA_RIEMANN_NUMERATOR_PRODUCT_STEPS)
    cov_cov_riem_coeffs =
        compute_inverse_derivative_numerator_components_truncated_coeff_space(
            step3;
            k = 4,
            pdeg = nabla2_riemann_pdeg,
            progress = cov_cov_riem_progress,
        )
    finish_progress!(cov_cov_riem_progress)

    cov_cov_riem_box_progress =
        start_progress("Second covariant Riemann subdivision boxes", nabla2_riemann_subdivisions * nabla2_riemann_subdivisions)
    cov_cov_riem_bounds = compute_cov_cov_riem_by_local_subdivision_truncated(
        cov_cov_riem_coeffs;
        pdeg = nabla2_riemann_pdeg,
        nx = nabla2_riemann_subdivisions,
        ny = nabla2_riemann_subdivisions,
        progress = cov_cov_riem_box_progress,
    )
    finish_progress!(cov_cov_riem_box_progress)

    metric_inverse = metric_inverse_json(inverse_bounds.derivative_bounds)
    curvature_bounds = Dict{String,Any}(
        "ricci_C0" => serialize_bound_value(ricci_bounds.ricci_norm_bound),
        "ricci_lower_bound" => serialize_bound_value(ricci_lower_bounds.ricci_lower_bound),
        "riemann_C0" => serialize_bound_value(riem_bounds.riem_norm_bound),
        "nabla_riemann_C0" => serialize_bound_value(cov_riem_bounds.cov_riem_norm_bound),
        "nabla2_riemann_C0" => serialize_bound_value(cov_cov_riem_bounds.cov_cov_riem_norm_bound),
    )

    update_verified_bounds_json(metric_inverse, curvature_bounds; path = output_path)
    println()
    println("Updated verified bounds: $output_path")

    return (;
        inverse_bounds,
        ricci_bounds,
        ricci_lower_bounds,
        riem_bounds,
        cov_riem_bounds,
        cov_cov_riem_bounds,
        metric_inverse,
        curvature_bounds,
        parameters = (;
            inverse_pdeg,
            inverse_subdivisions,
            ricci_pdeg,
            ricci_subdivisions,
            ricci_lower_pdeg,
            ricci_lower_subdivisions,
            riemann_pdeg,
            riemann_subdivisions,
            nabla_riemann_pdeg,
            nabla_riemann_subdivisions,
            nabla2_riemann_pdeg,
            nabla2_riemann_subdivisions,
        ),
    )
end

function main()
    setprecision(BigFloat, DEFAULT_PRECISION)
    run_verified_bounds_pipeline()
#     run_verified_bounds_pipeline(;
#     inverse_pdeg = 20,
#     inverse_subdivisions = 8,
#     ricci_pdeg = 12,
#     ricci_subdivisions = 6,
#     ricci_lower_pdeg = 12,
#     ricci_lower_subdivisions = 6,
#     riemann_pdeg = 10,
#     riemann_subdivisions = 4,
#     nabla_riemann_pdeg = 8,
#     nabla_riemann_subdivisions = 4,
#     nabla2_riemann_pdeg = 6,
#     nabla2_riemann_subdivisions = 3,
# )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
