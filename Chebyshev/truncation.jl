"""
Chebyshev tail bound.
"""
function chebyshev_tensor_tail_bound(coeffs::AbstractMatrix{<:Number}, N::Integer)
    interval_coeffs = intervalize_coefficients(coeffs)
    tail_bound = zero(abs(interval_coeffs[1, 1]))

    for j in axes(interval_coeffs, 2), i in axes(interval_coeffs, 1)
        m = i - 1
        n = j - 1

        if m >= N || n >= N
            tail_bound += abs(interval_coeffs[i, j])
        end
    end

    return tail_bound
end

"""Bound all nonconstant modes of a tensor-product Chebyshev series."""
function chebyshev_coeff_tail_bound(coeffs::AbstractMatrix{<:Number})
    interval_coeffs = intervalize_coefficients(coeffs)
    tail_bound = zero(abs(interval_coeffs[1, 1]))

    for j in axes(interval_coeffs, 2), i in axes(interval_coeffs, 1)
        i == 1 && j == 1 && continue
        tail_bound += abs(interval_coeffs[i, j])
    end
    return tail_bound
end

function absorb_tail_into_constant(coeffs, pdeg)
    coeffs = intervalize_coefficients(coeffs)

    tail_bound = chebyshev_tensor_tail_bound(coeffs, pdeg)

    reduced = copy(coeffs[1:pdeg, 1:pdeg])

    a00 = reduced[1,1]

    reduced[1,1] =
        interval(
            inf(a00) - sup(tail_bound),
            sup(a00) + sup(tail_bound)
        )

    return reduced
end

function inflate_constant_mode!(coeffs, eps)
    coeffs = copy(coeffs)
    a00 = coeffs[1, 1]

    r = sup(eps)
    symmetric_eps = interval(-r,r)

    coeffs[1, 1] = a00 + symmetric_eps

    return coeffs
end

"""
Truncate a Chebyshev coefficient array and record a rigorous tail bound.

The input coefficients are first intervalized. If `pdeg <= 0` or `pdeg` is at
least the smaller array dimension, no truncation is performed and the returned
tail is zero. Otherwise, the returned `coeffs` are the leading
`pdeg x pdeg` block, and `tail` is the interval bound on all coefficients with
at least one index outside that block.

Returns `(; coeffs, tail, pdeg)`.
"""
function truncate_coeffs_with_tail(coeffs::AbstractMatrix, pdeg::Integer)
    coeffs = intervalize_coefficients(coeffs)

    if pdeg <= 0 || pdeg >= minimum(size(coeffs))
        return (; coeffs = coeffs, tail = zero(abs(coeffs[1, 1])), pdeg = size(coeffs, 1))
    end

    tail = chebyshev_tensor_tail_bound(coeffs, pdeg)
    truncated = copy(coeffs[1:pdeg, 1:pdeg])

    return (; coeffs = truncated, tail, pdeg)
end
