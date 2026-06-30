using IntervalArithmetic

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

struct InverseMetricBoxOracle
    inverse_coeffs
end

function InverseMetricBoxOracle(coeffs_path::AbstractString = METRIC_U0_PATH)
    coeffs = load_metric_coeffs_csv(coeffs_path)
    derivatives = compute_second_derivatives(coeffs)
    inverse_coeffs = build_inverse_metric_coeffs(derivatives)
    return InverseMetricBoxOracle(inverse_coeffs)
end

function _box_interval(box::Interval)
    return interval(BigFloat(inf(box)), BigFloat(sup(box)))
end

function _box_interval(box::Tuple{<:Real, <:Real})
    lo, hi = box
    lo <= hi || error("Expected box endpoints in increasing order, got $box")
    return interval(BigFloat(lo), BigFloat(hi))
end

function _box_interval(box::AbstractVector{<:Real})
    length(box) == 2 || error("Expected two box endpoints, got length $(length(box))")
    return _box_interval((box[1], box[2]))
end

function _check_box_range(box::Interval, lo, hi, name::AbstractString)
    blo = BigFloat(lo)
    bhi = BigFloat(hi)
    inf(box) >= blo && sup(box) <= bhi ||
        error("$name must be contained in [$blo, $bhi], got $box")
    return nothing
end

function _package_inverse_metric_box_bounds(box_bounds, xcheb, ycheb)
    matrix = [
        box_bounds.u11_box box_bounds.u12_box
        box_bounds.u21_box box_bounds.u22_box
    ]

    lower = (;
        xx = inf(box_bounds.u11_box),
        xy = inf(box_bounds.u12_box),
        yx = inf(box_bounds.u21_box),
        yy = inf(box_bounds.u22_box),
    )

    upper = (;
        xx = sup(box_bounds.u11_box),
        xy = sup(box_bounds.u12_box),
        yx = sup(box_bounds.u21_box),
        yy = sup(box_bounds.u22_box),
    )

    return (;
        xx = box_bounds.u11_box,
        xy = box_bounds.u12_box,
        yx = box_bounds.u21_box,
        yy = box_bounds.u22_box,
        matrix,
        lower,
        upper,
        D = box_bounds.D_box,
        D_lower = box_bounds.D_lower,
        A11 = box_bounds.A11_box,
        A12 = box_bounds.A12_box,
        A22 = box_bounds.A22_box,
        chebyshev_box = (; x = xcheb, y = ycheb),
    )
end

"""
Return rigorous lower and upper bounds for all four entries of `u^{ij}`.

The input box is in the Chebyshev variables used by the inverse metric
coefficient arrays, so both coordinates must be subintervals of `[-1, 1]`.
"""
function inverse_metric_bounds_on_chebyshev_box(
    oracle::InverseMetricBoxOracle,
    xbox,
    ybox,
)
    xcheb = _box_interval(xbox)
    ycheb = _box_interval(ybox)
    _check_box_range(xcheb, -1, 1, "xbox")
    _check_box_range(ycheb, -1, 1, "ybox")

    box_bounds = inverse_metric_box_bounds(oracle.inverse_coeffs, xcheb, ycheb)
    return _package_inverse_metric_box_bounds(box_bounds, xcheb, ycheb)
end

"""
Bound `u^{ij}` on a box in the `[0, 1]^2` coefficient coordinates.

The coordinate conversion is `x_cheb = 2x - 1`, `y_cheb = 1 - 2y`, matching
the exported basis `T_m(2x - 1) T_n(1 - 2y)`.
"""
function inverse_metric_bounds_on_unit_box(
    oracle::InverseMetricBoxOracle,
    xbox,
    ybox,
)
    xunit = _box_interval(xbox)
    yunit = _box_interval(ybox)
    _check_box_range(xunit, 0, 1, "xbox")
    _check_box_range(yunit, 0, 1, "ybox")

    xcheb = interval_constant(2) * xunit - interval_constant(1)
    ycheb = interval_constant(1) - interval_constant(2) * yunit
    return inverse_metric_bounds_on_chebyshev_box(oracle, xcheb, ycheb)
end

"""
Bound `u^{ij}` on a box in the square `[0, 1] x [-1, 0]`.

For this lower square inside the hexagon, the exported coefficients use
`x_cheb = 2x - 1` and `y_cheb = 2y + 1`.
"""
function inverse_metric_bounds_on_lower_square_box(
    oracle::InverseMetricBoxOracle,
    xbox,
    ybox,
)
    xsquare = _box_interval(xbox)
    ysquare = _box_interval(ybox)
    _check_box_range(xsquare, 0, 1, "xbox")
    _check_box_range(ysquare, -1, 0, "ybox")

    xcheb = interval_constant(2) * xsquare - interval_constant(1)
    ycheb = interval_constant(2) * ysquare + interval_constant(1)
    return inverse_metric_bounds_on_chebyshev_box(oracle, xcheb, ycheb)
end

function main()
    setprecision(BigFloat, 100)

    oracle = InverseMetricBoxOracle()
    U = inverse_metric_bounds_on_lower_square_box(
        oracle,
        (0 // 1, 1 // 1000),
        (-1 // 1000, 0 // 1),
    )

    println("xx = ", U.xx)
    println("xy = ", U.xy)
    println("yx = ", U.yx)
    println("yy = ", U.yy)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
