using IntervalArithmetic

"""
Local bivariate polynomials are dense coefficient matrices.

Entry `P[p + 1, q + 1]` is the coefficient of `x^p y^q`. Coefficients are
assumed to be intervals; these helpers deliberately do no type promotion.
"""

"""
Return a degree-`deg` enclosure of the local polynomial `A`.

The coefficient `A[p + 1, q + 1]` represents the term `x^p y^q`. Terms with
`p + q <= deg` are copied into the output. All other terms are discarded as
polynomial terms, but their absolute coefficient intervals are summed and added
to the constant coefficient as a symmetric interval. This is rigorous for local
coordinates satisfying `|x| <= 1` and `|y| <= 1`, since every discarded monomial
is then bounded in absolute value by the absolute value of its coefficient.
"""
function poly_trim_with_tail(A, deg::Integer)
    B = zeros(eltype(A), min(size(A, 1), deg + 1), min(size(A, 2), deg + 1))
    tail = zero(abs(A[1, 1]))

    for j in axes(A, 2), i in axes(A, 1)
        if i <= size(B, 1) && j <= size(B, 2) && (i - 1) + (j - 1) <= deg
            B[i, j] += A[i, j]
        else
            tail += abs(A[i, j])
        end
    end

    B[1, 1] += symmetric_interval(tail)
    return B
end

"""Add two dense local polynomials after padding them to a common size."""
function poly_add(A, B)
    C = zeros(eltype(A), max(size(A, 1), size(B, 1)), max(size(A, 2), size(B, 2)))
    C[1:size(A, 1), 1:size(A, 2)] .+= A
    C[1:size(B, 1), 1:size(B, 2)] .+= B
    return C
end

"""Subtract two dense local polynomials."""
poly_sub(A, B) = poly_add(A, -B)

"""Scale every coefficient of a dense local polynomial."""
poly_scale(A, c) = c .* A

"""Test whether an interval coefficient is exactly the zero interval."""
poly_exact_zero(x) = isequal_interval(x, zero(x))

"""Multiply two dense local polynomials without truncation."""
function poly_mul(A, B)
    C = zeros(eltype(A), size(A, 1) + size(B, 1) - 1, size(A, 2) + size(B, 2) - 1)

    for j in axes(A, 2), i in axes(A, 1)
        poly_exact_zero(A[i, j]) && continue
        for l in axes(B, 2), k in axes(B, 1)
            poly_exact_zero(B[k, l]) && continue
            C[i + k - 1, j + l - 1] += A[i, j] * B[k, l]
        end
    end

    return C
end

"""Multiply two local polynomials and enclose all terms above `deg` in the tail."""
poly_mul_trunc(A, B, deg::Integer) = poly_trim_with_tail(poly_mul(A, B), deg)

"""Convert a truncated Chebyshev series to a local power polynomial with tail."""
function local_polynomial_with_tail(trunc, xcheb, ycheb, deg::Integer)
    P = local_power_coeffs_cheb_2d(trunc.coeffs, xcheb, ycheb)
    P[1, 1] += symmetric_interval(trunc.tail)
    return poly_trim_with_tail(P, deg)
end

"""Convert an exact Chebyshev series to a trimmed local power polynomial."""
function local_polynomial_trimmed(coeffs, xcheb, ycheb, deg::Integer)
    return poly_trim_with_tail(local_power_coeffs_cheb_2d(coeffs, xcheb, ycheb), deg)
end

"""Return local numerator and denominator polynomials for the inverse metric."""
function local_metric_component_polynomials(oracle, xcheb, ycheb; deg::Integer)
    v = oracle.canonical

    h11 = local_polynomial_with_tail(oracle.h11, xcheb, ycheb, deg)
    h12 = local_polynomial_with_tail(oracle.h12, xcheb, ycheb, deg)
    h22 = local_polynomial_with_tail(oracle.h22, xcheb, ycheb, deg)

    lprod = local_polynomial_trimmed(v.lprod, xcheb, ycheb, deg)
    v11 = local_polynomial_trimmed(v.lprod_v11, xcheb, ycheb, deg)
    v12 = local_polynomial_trimmed(v.lprod_v12, xcheb, ycheb, deg)
    v22 = local_polynomial_trimmed(v.lprod_v22, xcheb, ycheb, deg)
    B = local_polynomial_trimmed(v.B, xcheb, ycheb, deg)

    A11 = poly_trim_with_tail(poly_add(v22, poly_mul_trunc(lprod, h22, deg)), deg)
    A12 = poly_scale(
        poly_trim_with_tail(poly_add(v12, poly_mul_trunc(lprod, h12, deg)), deg),
        -interval(BigFloat(1)),
    )
    A22 = poly_trim_with_tail(poly_add(v11, poly_mul_trunc(lprod, h11, deg)), deg)

    D = poly_add(B, poly_mul_trunc(v22, h11, deg))
    D = poly_add(D, poly_mul_trunc(v11, h22, deg))
    D = poly_sub(D, poly_scale(poly_mul_trunc(v12, h12, deg), interval(BigFloat(2))))
    determinant = poly_sub(
        poly_mul_trunc(h11, h22, deg),
        poly_mul_trunc(h12, h12, deg),
    )
    D = poly_add(D, poly_mul_trunc(lprod, determinant, deg))
    D = poly_trim_with_tail(D, deg)

    return (; A11, A12, A22, D)
end

"""Bound the absolute value of a local polynomial on `[-1,1]^2`."""
poly_abs_bound(A) = sum(abs(A[i, j]) for j in axes(A, 2), i in axes(A, 1))

"""
Enclose `1 / D` on `[-1,1]^2` using a finite Neumann expansion.

Write `D = d0 + R`, where `d0` is the midpoint of the interval constant
coefficient. Since every local monomial has absolute value at most one,
`poly_abs_bound(R)` bounds `|R|` throughout the box. The returned `ratio` is a
rigorous upper bound for `|R| / |d0|`.

If `ratio < 1`, the Neumann series converges uniformly. Return its truncated
polynomial, including a rigorous geometric-series tail, together with `ratio`.
Return `nothing` if the chosen centre is zero or contraction cannot be proved.
"""
function reciprocal_polynomial_neumann(D; deg::Integer, terms::Integer)
    d0 = (inf(D[1, 1]) + sup(D[1, 1])) / 2
    if iszero(d0)
        return nothing
    end
    R = copy(D)
    R[1, 1] -= interval(d0)

    ratio = sup(poly_abs_bound(R) / interval(abs(d0)))
    if ratio >= 1
        return nothing
    end

    P = zeros(eltype(D), 1, 1)
    Rpow = zeros(eltype(D), 1, 1)
    Rpow[1, 1] = interval(BigFloat(1))

    for k in 0:terms
        P = poly_add(P, poly_scale(Rpow, interval((-1)^k) / interval(d0)^(k + 1)))
        Rpow = poly_mul_trunc(Rpow, R, deg)
    end

    ratio_interval = interval(ratio)
    remainder = sup(
        ratio_interval^(terms + 1) /
        (interval(abs(d0)) * (interval(BigFloat(1)) - ratio_interval)),
    )
    P[1, 1] += symmetric_interval(interval(remainder))
    return (; polynomial = poly_trim_with_tail(P, deg), ratio)
end
