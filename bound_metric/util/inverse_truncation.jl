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

    coeffs[1, 1] = interval(
        inf(a00) - sup(eps),
        sup(a00) + sup(eps),
    )

    return coeffs
end

function derivative_tail_bounds(full_coeffs, truncated_coeffs, pdeg::Integer)
    full_step2 = compute_second_derivatives(full_coeffs)
    trunc_step2 = compute_second_derivatives(truncated_coeffs)

    uxx_tail = chebyshev_tensor_tail_bound(full_step2.uxx, pdeg)
    uyy_tail = chebyshev_tensor_tail_bound(full_step2.uyy, pdeg)
    uxy_tail = chebyshev_tensor_tail_bound(full_step2.uxy, pdeg)

    return (; full_step2, trunc_step2, uxx_tail, uyy_tail, uxy_tail)
end

function truncate_with_derivative_tails(coeffs, pdeg::Integer)
    full_coeffs = intervalize_coefficients(coeffs)

    u_tail = chebyshev_tensor_tail_bound(full_coeffs, pdeg)
    u_trunc = copy(full_coeffs[1:pdeg, 1:pdeg])
    u_trunc = inflate_constant_mode!(u_trunc, u_tail)

    tails = derivative_tail_bounds(full_coeffs, u_trunc, pdeg)

    uxx = inflate_constant_mode!(tails.trunc_step2.uxx, tails.uxx_tail)
    uyy = inflate_constant_mode!(tails.trunc_step2.uyy, tails.uyy_tail)
    uxy = inflate_constant_mode!(tails.trunc_step2.uxy, tails.uxy_tail)

    pack = build_derivative_pack(u_trunc)

    step2 = (;
        uxx,
        uyy,
        uxy,
        pack,
        u_tail,
        uxx_tail = tails.uxx_tail,
        uyy_tail = tails.uyy_tail,
        uxy_tail = tails.uxy_tail,
    )

    return (; u_trunc, step2)
end

function truncate_coeffs_with_tail(coeffs::AbstractMatrix, pdeg::Integer)
    coeffs = intervalize_coefficients(coeffs)

    if pdeg <= 0 || pdeg >= minimum(size(coeffs))
        return (; coeffs = coeffs, tail = zero(abs(coeffs[1, 1])), pdeg = size(coeffs, 1))
    end

    tail = chebyshev_tensor_tail_bound(coeffs, pdeg)
    truncated = copy(coeffs[1:pdeg, 1:pdeg])

    return (; coeffs = truncated, tail, pdeg)
end

function symmetric_interval(eps)
    e = sup(abs(eps))
    return interval(-e, e)
end

function prepare_inverse_coeffs_for_subdivision(inverse_coeffs, pdeg::Integer)
    return (;
        A11 = truncate_coeffs_with_tail(inverse_coeffs.A11, pdeg),
        A12 = truncate_coeffs_with_tail(inverse_coeffs.A12, pdeg),
        A22 = truncate_coeffs_with_tail(inverse_coeffs.A22, pdeg),
        D   = truncate_coeffs_with_tail(inverse_coeffs.D,   pdeg),
    )
end
