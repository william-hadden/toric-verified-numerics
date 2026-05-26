function chebyshev_coeff_tail_bound(coeffs::AbstractMatrix{<:Number})
    interval_coeffs = intervalize_coefficients(coeffs)
    tail_bound = zero(abs(interval_coeffs[1, 1]))

    for j in axes(interval_coeffs, 2), i in axes(interval_coeffs, 1)
        i == 1 && j == 1 && continue
        tail_bound += abs(interval_coeffs[i, j])
    end

    return tail_bound
end

function compute_inverse_bound_by_coefficient_bounds(inverse_coeffs)
    A11_bound = chebyshev_coeff_sup_bound(inverse_coeffs.A11)
    A12_bound = chebyshev_coeff_sup_bound(inverse_coeffs.A12)
    A22_bound = chebyshev_coeff_sup_bound(inverse_coeffs.A22)

    D_constant = inverse_coeffs.D[1, 1]
    D_tail_bound = chebyshev_coeff_tail_bound(inverse_coeffs.D)
    D_lower = interval(inf(D_constant)) - interval(sup(D_tail_bound))

    if inf(D_lower) <= 0
        error(
            "Coefficient-bound method could not certify positivity of the denominator: " *
            "D_lower = $D_lower",
        )
    end

    u11_bound = A11_bound / D_lower
    u12_bound = A12_bound / D_lower
    u22_bound = A22_bound / D_lower

    return (;
        A11_bound,
        A12_bound,
        A22_bound,
        D_constant,
        D_tail_bound,
        D_lower,
        u11_bound,
        u12_bound,
        u22_bound,
    )
end

bound_upper(x) = x
bound_upper(x::Interval) = sup(x)
bound_lower(x) = x
bound_lower(x::Interval) = inf(x)

function print_inverse_bound_summary(step4)
    println()
    println("Certified C^0 bounds for inverse metric entries:")
    println("    ||u^{11}||_∞ <= $(bound_upper(step4.u11_bound))")
    println("    ||u^{12}||_∞ <= $(bound_upper(step4.u12_bound))")
    println("    ||u^{22}||_∞ <= $(bound_upper(step4.u22_bound))")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(bound_lower(step4.D_lower))")
end
