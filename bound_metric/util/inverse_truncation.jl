"""
Compute second-derivative tail bounds used by inverse-metric certificates.
"""
function derivative_tail_bounds(full_coeffs, truncated_coeffs, pdeg::Integer)
    # full_step2 = compute_second_derivatives(full_coeffs)
    # trunc_step2 = compute_second_derivatives(truncated_coeffs)
    full_step2 = build_derivative_pack(full_coeffs)
    trunc_step2 = build_derivative_pack(truncated_coeffs)

    uxx_tail = chebyshev_tensor_tail_bound(full_step2.uxx, pdeg)
    uyy_tail = chebyshev_tensor_tail_bound(full_step2.uyy, pdeg)
    uxy_tail = chebyshev_tensor_tail_bound(full_step2.uxy, pdeg)

    return (; full_step2, trunc_step2, uxx_tail, uyy_tail, uxy_tail)
end

function truncate_with_derivative_tails(coeffs, pdeg::Integer)
    full_coeffs = intervalize_coefficients(coeffs)
    u_tail = chebyshev_tensor_tail_bound(full_coeffs, pdeg)
    u_trunc = inflate_constant_mode!(copy(full_coeffs[1:pdeg, 1:pdeg]), u_tail)
    tails = derivative_tail_bounds(full_coeffs, u_trunc, pdeg)

    uxx = inflate_constant_mode!(tails.trunc_step2.uxx, tails.uxx_tail)
    uyy = inflate_constant_mode!(tails.trunc_step2.uyy, tails.uyy_tail)
    uxy = inflate_constant_mode!(tails.trunc_step2.uxy, tails.uxy_tail)
    pack = build_derivative_pack(u_trunc)

    step2 = (;
        uxx, uyy, uxy, pack, u_tail,
        uxx_tail = tails.uxx_tail,
        uyy_tail = tails.uyy_tail,
        uxy_tail = tails.uxy_tail,
    )
    return (; u_trunc, step2)
end

function prepare_inverse_coeffs_for_subdivision(inverse_coeffs, pdeg::Integer)
    return (;
        A11 = truncate_coeffs_with_tail(inverse_coeffs.A11, pdeg),
        A12 = truncate_coeffs_with_tail(inverse_coeffs.A12, pdeg),
        A22 = truncate_coeffs_with_tail(inverse_coeffs.A22, pdeg),
        D = truncate_coeffs_with_tail(inverse_coeffs.D, pdeg),
    )
end
