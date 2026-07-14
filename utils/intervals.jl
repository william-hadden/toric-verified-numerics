# Generic helpers for constructing and inspecting rigorous intervals.
intervalize_coefficients(coefficients::AbstractMatrix{<:Interval}) = Matrix(coefficients)
intervalize_coefficients(coefficients::AbstractMatrix{<:Number}) = interval.(coefficients)

"""
Create a point interval with `BigFloat` endpoints.

This helper is used for constants in rigorous interval computations so integer,
rational, and floating inputs are promoted through `BigFloat` before being
wrapped as an `Interval`.
"""
interval_constant(x) = interval(BigFloat(x))
interval_half() = interval_constant(1) / exact(2)
interval_quarter() = interval_constant(1) / exact(4)

function symmetric_interval(eps)
    e = sup(abs(eps))
    return interval(-e, e)
end

bound_upper(x) = x
bound_upper(x::Interval) = sup(x)
bound_lower(x) = x
bound_lower(x::Interval) = inf(x)
