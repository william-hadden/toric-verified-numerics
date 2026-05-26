function derivative_coeff_pack(coeffs::AbstractMatrix{<:Number})
    x = differentiate_coeffs_x(coeffs)
    y = differentiate_coeffs_y(coeffs)
    xx = differentiate_coeffs_x(x)
    xy = differentiate_coeffs_y(x)
    yy = differentiate_coeffs_y(y)
    return (; f = coeffs, x, y, xx, xy, yy)
end

first_derivative_coeffs(pack, dim::Integer) =
    dim == 1 ? pack.x :
    dim == 2 ? pack.y :
    error("Expected derivative dimension 1 or 2")

second_derivative_coeffs(pack, a::Integer, b::Integer) =
    a == 1 && b == 1 ? pack.xx :
    a == 2 && b == 2 ? pack.yy :
    a != b ? pack.xy :
    error("Expected derivative dimensions 1 or 2")

function coeff_sup_bound_number(coeffs::AbstractMatrix{<:Number})
    return sup(chebyshev_coeff_sup_bound(intervalize_coefficients(coeffs)))
end

function truncated_coeff_enclosure(coeffs::AbstractMatrix{<:Number}, pdeg::Integer)
    trunc = truncate_coeffs_with_tail(coeffs, pdeg)
    tail = trunc.tail isa Interval ? sup(abs(trunc.tail)) : abs(trunc.tail)
    return (; coeffs = trunc.coeffs, tail, pdeg = trunc.pdeg)
end

function truncate_enclosure_coeffs(coeffs::AbstractMatrix{<:Number}, tail, pdeg::Integer)
    trunc = truncate_coeffs_with_tail(coeffs, pdeg)

    old_tail = tail isa Interval ? sup(abs(tail)) : abs(tail)
    new_tail = trunc.tail isa Interval ? sup(abs(trunc.tail)) : abs(trunc.tail)

    return (; coeffs = trunc.coeffs, tail = old_tail + new_tail, pdeg = trunc.pdeg)
end

function enclosure_add(f, g, pdeg::Integer)
    return truncate_enclosure_coeffs(cheb_add(f.coeffs, g.coeffs), f.tail + g.tail, pdeg)
end

function enclosure_sub(f, g, pdeg::Integer)
    return truncate_enclosure_coeffs(cheb_sub(f.coeffs, g.coeffs), f.tail + g.tail, pdeg)
end

function enclosure_scale(f, c, pdeg::Integer)
    c_abs = c isa Interval ? sup(abs(c)) : abs(c)
    return truncate_enclosure_coeffs(cheb_scale(f.coeffs, c), c_abs * f.tail, pdeg)
end

function enclosure_mul(f, g, pdeg::Integer; progress = nothing)
    product_coeffs = cheb_mul_fast(f.coeffs, g.coeffs)
    advance_progress!(progress)

    product_tail =
        f.tail * coeff_sup_bound_number(g.coeffs) +
        g.tail * coeff_sup_bound_number(f.coeffs) +
        f.tail * g.tail

    return truncate_enclosure_coeffs(product_coeffs, product_tail, pdeg)
end

function enclosure_to_coeffs(f)
    return inflate_constant_mode!(f.coeffs, f.tail)
end

# worker
function quotient_second_derivative_numerator_enclosure(
    A_derivs,
    D_derivs,
    D2,
    a::Integer,
    b::Integer,
    pdeg::Integer;
    progress = nothing,
)
    A = truncated_coeff_enclosure(A_derivs.f, pdeg)
    A_a = truncated_coeff_enclosure(first_derivative_coeffs(A_derivs, a), pdeg)
    A_b = truncated_coeff_enclosure(first_derivative_coeffs(A_derivs, b), pdeg)
    A_ab = truncated_coeff_enclosure(second_derivative_coeffs(A_derivs, a, b), pdeg)

    D = truncated_coeff_enclosure(D_derivs.f, pdeg)
    D_a = truncated_coeff_enclosure(first_derivative_coeffs(D_derivs, a), pdeg)
    D_b = truncated_coeff_enclosure(first_derivative_coeffs(D_derivs, b), pdeg)
    D_ab = truncated_coeff_enclosure(second_derivative_coeffs(D_derivs, a, b), pdeg)

    term1 = enclosure_mul(A_ab, D2, pdeg; progress)

    A_a_D_b = enclosure_mul(A_a, D_b, pdeg; progress)
    A_b_D_a = enclosure_mul(A_b, D_a, pdeg; progress)
    A_D_ab = enclosure_mul(A, D_ab, pdeg; progress)

    term2_inner = enclosure_add(enclosure_add(A_a_D_b, A_b_D_a, pdeg), A_D_ab, pdeg)
    term2 = enclosure_mul(term2_inner, D, pdeg; progress)

    A_D_a = enclosure_mul(A, D_a, pdeg; progress)
    term3_inner = enclosure_mul(A_D_a, D_b, pdeg; progress)
    term3 = enclosure_scale(term3_inner, interval_constant(2), pdeg)

    return enclosure_add(enclosure_sub(term1, term2, pdeg), term3, pdeg)
end

# worker
function compute_ricci_numerators_truncated_coefficient_space(
    inverse_coeffs;
    pdeg::Integer,
    progress = nothing,
)
    pdeg > 0 || error("Truncated Ricci numerator construction requires pdeg > 0")

    A11 = inverse_coeffs.A11
    A12 = inverse_coeffs.A12
    A22 = inverse_coeffs.A22
    D = inverse_coeffs.D

    D_enclosure = truncated_coeff_enclosure(D, pdeg)
    D2 = enclosure_mul(D_enclosure, D_enclosure, pdeg; progress)

    D_derivs = derivative_coeff_pack(D)
    A11_derivs = derivative_coeff_pack(A11)
    A12_derivs = derivative_coeff_pack(A12)
    A22_derivs = derivative_coeff_pack(A22)

    u11_xx_num = quotient_second_derivative_numerator_enclosure(A11_derivs, D_derivs, D2, 1, 1, pdeg; progress)
    u11_xy_num = quotient_second_derivative_numerator_enclosure(A11_derivs, D_derivs, D2, 1, 2, pdeg; progress)

    u12_xx_num = quotient_second_derivative_numerator_enclosure(A12_derivs, D_derivs, D2, 1, 1, pdeg; progress)
    u12_xy_num = quotient_second_derivative_numerator_enclosure(A12_derivs, D_derivs, D2, 1, 2, pdeg; progress)
    u12_yy_num = quotient_second_derivative_numerator_enclosure(A12_derivs, D_derivs, D2, 2, 2, pdeg; progress)

    u22_xy_num = quotient_second_derivative_numerator_enclosure(A22_derivs, D_derivs, D2, 1, 2, pdeg; progress)
    u22_yy_num = quotient_second_derivative_numerator_enclosure(A22_derivs, D_derivs, D2, 2, 2, pdeg; progress)

    half = interval_constant(-1) / exact(2)

    R11_num = enclosure_to_coeffs(enclosure_scale(enclosure_add(u11_xx_num, u12_xy_num, pdeg), half, pdeg))
    R12_num = enclosure_to_coeffs(enclosure_scale(enclosure_add(u11_xy_num, u12_yy_num, pdeg), half, pdeg))
    R21_num = enclosure_to_coeffs(enclosure_scale(enclosure_add(u12_xx_num, u22_xy_num, pdeg), half, pdeg))
    R22_num = enclosure_to_coeffs(enclosure_scale(enclosure_add(u12_xy_num, u22_yy_num, pdeg), half, pdeg))

    return (; R11_num, R12_num, R21_num, R22_num, D, pdeg)
end

function prepare_ricci_coeffs_for_subdivision(ricci_coeffs, pdeg::Integer)
    return (;
        R11_num = truncate_coeffs_with_tail(ricci_coeffs.R11_num, pdeg),
        R12_num = truncate_coeffs_with_tail(ricci_coeffs.R12_num, pdeg),
        R21_num = truncate_coeffs_with_tail(ricci_coeffs.R21_num, pdeg),
        R22_num = truncate_coeffs_with_tail(ricci_coeffs.R22_num, pdeg),
        D       = truncate_coeffs_with_tail(ricci_coeffs.D, pdeg),
    )
end

# worker
function compute_ricci_bound_by_local_subdivision_truncated(
    ricci_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = prepare_ricci_coeffs_for_subdivision(ricci_coeffs, pdeg)

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_R11 = big"0"
    global_R12 = big"0"
    global_R21 = big"0"
    global_R22 = big"0"
    global_ricci_norm_squared = big"0"
    global_ricci_norm = big"0"
    global_D_lower = big"Inf"

    box_results = []

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        R11_num_box = local_coeff_sum_centered_enclosure_with_tail(trunc.R11_num, xbox, ybox)
        R12_num_box = local_coeff_sum_centered_enclosure_with_tail(trunc.R12_num, xbox, ybox)
        R21_num_box = local_coeff_sum_centered_enclosure_with_tail(trunc.R21_num, xbox, ybox)
        R22_num_box = local_coeff_sum_centered_enclosure_with_tail(trunc.R22_num, xbox, ybox)
        D_box       = local_coeff_sum_centered_enclosure_with_tail(trunc.D,       xbox, ybox)

        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on Ricci box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        R11_box = R11_num_box / (D_box^3)
        R12_box = R12_num_box / (D_box^3)
        R21_box = R21_num_box / (D_box^3)
        R22_box = R22_num_box / (D_box^3)

        R11_bound = sup(abs(R11_box))
        R12_bound = sup(abs(R12_box))
        R21_bound = sup(abs(R21_box))
        R22_bound = sup(abs(R22_box))

        ricci_norm_squared_box =
            interval_constant(2) * (
                R11_box^2 +
                interval_constant(2) * R12_box * R21_box +
                R22_box^2
            )

        ricci_norm_squared_bound = sup(ricci_norm_squared_box)

        ricci_norm_squared_bound < 0 && error(
            "Ricci norm-squared enclosure has negative upper bound: $ricci_norm_squared_box"
        )

        ricci_norm_bound = sqrt(ricci_norm_squared_bound)

        global_R11 = max(global_R11, R11_bound)
        global_R12 = max(global_R12, R12_bound)
        global_R21 = max(global_R21, R21_bound)
        global_R22 = max(global_R22, R22_bound)
        global_ricci_norm_squared = max(global_ricci_norm_squared, ricci_norm_squared_bound)
        global_ricci_norm = max(global_ricci_norm, ricci_norm_bound)
        global_D_lower = min(global_D_lower, D_lower)

        push!(box_results, (;
            ix,
            iy,
            xbox,
            ybox,
            R11_num_box,
            R12_num_box,
            R21_num_box,
            R22_num_box,
            D_box,
            D_lower,
            R11_box,
            R12_box,
            R21_box,
            R22_box,
            R11_bound,
            R12_bound,
            R21_bound,
            R22_bound,
            ricci_norm_squared_box,
            ricci_norm_squared_bound,
            ricci_norm_bound,
        ))

        advance_progress!(progress)
    end

    return (;
        nx,
        ny,
        pdeg,
        trunc,
        box_results,
        D_lower = global_D_lower,
        R11_bound = global_R11,
        R12_bound = global_R12,
        R21_bound = global_R21,
        R22_bound = global_R22,
        ricci_norm_squared_bound = global_ricci_norm_squared,
        ricci_norm_bound = global_ricci_norm,
    )
end

function print_truncated_subdivision_ricci_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bound for Ricci:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||Ric||_inf <= $(step.ricci_norm_bound)")
    println("    ||Ric||_inf^2 <= $(step.ricci_norm_squared_bound)")
    println()
    println("Ricci entry bounds:")
    println("    ||R^1_1||_inf <= $(step.R11_bound)")
    println("    ||R^1_2||_inf <= $(step.R12_bound)")
    println("    ||R^2_1||_inf <= $(step.R21_bound)")
    println("    ||R^2_2||_inf <= $(step.R22_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Denominator coefficient tail bound:")
    println("    D tail <= $(step.trunc.D.tail)")
end