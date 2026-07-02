using IntervalArithmetic
using Logging

Logging.disable_logging(Logging.Info)

if !isdefined(@__MODULE__, :inverse_metric_box_bounds)
    include(joinpath(@__DIR__, "..", "bound_residual", "util", "io.jl"))
    include(joinpath(@__DIR__, "..", "bound_residual", "util", "chebyshev_algebra.jl"))
    include(joinpath(@__DIR__, "..", "bound_residual", "util", "problem.jl"))
    include(joinpath(@__DIR__, "..", "bound_metric", "util", "io.jl"))
    include(joinpath(@__DIR__, "..", "bound_metric", "util", "progress.jl"))
    include(joinpath(@__DIR__, "..", "bound_metric", "util", "interval_helpers.jl"))
    include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_coefficients.jl"))
    include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_truncation.jl"))
    include(joinpath(@__DIR__, "..", "bound_metric", "util", "chebyshev_interval_evaluation.jl"))
    include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_subdivision_bounds.jl"))
end

struct InverseMetricBoxOracleV2
    h11
    h12
    h22
    canonical
end

function InverseMetricBoxOracleV2(coeffs_path::AbstractString = METRIC_U0_PATH; pdeg::Integer = 30)
    coeffs = load_metric_coeffs_csv(coeffs_path)
    derivatives = compute_second_derivatives(coeffs)
    return InverseMetricBoxOracleV2(
        truncate_coeffs_with_tail(derivatives.uxx, pdeg),
        truncate_coeffs_with_tail(derivatives.uxy, pdeg),
        truncate_coeffs_with_tail(derivatives.uyy, pdeg),
        factored_canonical_metric_coeffs(),
    )
end

function inverse_metric_bounds_on_lower_square_box_v2(
    oracle::InverseMetricBoxOracleV2,
    xbox::Interval,
    ybox::Interval,
)
    xcheb = exact(2) * xbox - exact(1)
    ycheb = exact(2) * ybox + exact(1)
    v = oracle.canonical

    h11 = local_coeff_sum_centered_enclosure_with_tail(oracle.h11, xcheb, ycheb)
    h12 = local_coeff_sum_centered_enclosure_with_tail(oracle.h12, xcheb, ycheb)
    h22 = local_coeff_sum_centered_enclosure_with_tail(oracle.h22, xcheb, ycheb)

    lprod = local_coeff_sum_centered_enclosure_cheb_2d(v.lprod, xcheb, ycheb)
    v11 = local_coeff_sum_centered_enclosure_cheb_2d(v.lprod_v11, xcheb, ycheb)
    v12 = local_coeff_sum_centered_enclosure_cheb_2d(v.lprod_v12, xcheb, ycheb)
    v22 = local_coeff_sum_centered_enclosure_cheb_2d(v.lprod_v22, xcheb, ycheb)
    B = local_coeff_sum_centered_enclosure_cheb_2d(v.B, xcheb, ycheb)

    A11 = v22 + lprod * h22
    A12 = -(v12 + lprod * h12)
    A22 = v11 + lprod * h11
    D = B + v22 * h11 + v11 * h22 - exact(2) * v12 * h12 +
        lprod * (h11 * h22 - h12^2)
    box_bounds = inverse_metric_box_from_component_enclosures(A11, A12, A22, D)

    return (;
        xx = box_bounds.u11_box,
        xy = box_bounds.u12_box,
        yx = box_bounds.u21_box,
        yy = box_bounds.u22_box,
    )
end
