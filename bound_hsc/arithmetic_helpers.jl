
function exact_big_interval(x::Integer)
    value = BigFloat(x)
    isinteger(value) && BigInt(value) == x || error(
        "Integer $x is not exactly representable at the current BigFloat precision",
    )
    return interval(value)
end

exact_big_interval(x::Rational) =
    exact_big_interval(numerator(x)) / exact_big_interval(denominator(x))
exact_big_interval(x::BigFloat) = interval(x)
exact_big_interval(x::AbstractFloat) = interval(BigFloat, x)

function exact_big_hull(lo::Rational, hi::Rational)
    ilo = exact_big_interval(lo)
    ihi = exact_big_interval(hi)
    return interval(BigFloat, inf(ilo), sup(ihi))
end

function box_intervals(B::PolytopeBox)
    X = exact_big_hull(B.xlo, B.xhi)
    Y = exact_big_hull(B.ylo, B.yhi)
    return X, Y
end

all_guaranteed(x::Number) = isguaranteed(x)
all_guaranteed(xs) = all(all_guaranteed, xs)

function require_guaranteed(x, label::AbstractString)
    all_guaranteed(x) || error("Non-guaranteed interval in $label")
    return x
end


eval_component(c::Interval, X, Y) =
    require_guaranteed(c, "interval direction component")

eval_component(c::Real, X, Y) =
    require_guaranteed(exact_big_interval(c), "direction component")

eval_component(c::AbstractMatrix, X, Y) =
    evaluate_coeffs_at_point(c, X, Y)

function as_big_interval(x::Interval)
    require_guaranteed(x, "input interval parameter")
    return require_guaranteed(
        interval(BigFloat, inf(x), sup(x)),
        "converted interval parameter",
    )
end

as_big_interval(x::Real) =
    require_guaranteed(exact_big_interval(x), "real parameter")