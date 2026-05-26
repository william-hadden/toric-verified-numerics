partial_coeffs(coeffs::AbstractMatrix{<:Number}, dim::Integer) =
    dim == 1 ? differentiate_coeffs_x(coeffs) :
    dim == 2 ? differentiate_coeffs_y(coeffs) :
    error("Expected derivative dimension 1 or 2")

function derivative_coeff_pack(coeffs::AbstractMatrix{<:Number})
    x = partial_coeffs(coeffs, 1)
    y = partial_coeffs(coeffs, 2)
    xx = partial_coeffs(x, 1)
    xy = partial_coeffs(x, 2)
    yy = partial_coeffs(y, 2)
    return (; f = coeffs, x, y, xx, xy, yy)
end

second_derivative_coeffs(pack, a::Integer, b::Integer) =
    a == 1 && b == 1 ? pack.xx :
    a == 2 && b == 2 ? pack.yy :
    a != b ? pack.xy :
    error("Expected derivative dimensions 1 or 2")

first_derivative_coeffs(pack, dim::Integer) =
    dim == 1 ? pack.x :
    dim == 2 ? pack.y :
    error("Expected derivative dimension 1 or 2")

function quotient_second_derivative_numerator(A::AbstractMatrix{<:Number}, D::AbstractMatrix{<:Number}, D2::AbstractMatrix{<:Number}, a::Integer, b::Integer; progress = nothing)
    A_derivs = derivative_coeff_pack(A)
    D_derivs = derivative_coeff_pack(D)
    return quotient_second_derivative_numerator(A_derivs, D_derivs, D2, a, b; progress)
end

function quotient_second_derivative_numerator(A_derivs, D_derivs, D2::AbstractMatrix{<:Number}, a::Integer, b::Integer; progress = nothing)
    A = A_derivs.f
    D = D_derivs.f
    A_a = first_derivative_coeffs(A_derivs, a)
    A_b = first_derivative_coeffs(A_derivs, b)
    A_ab = second_derivative_coeffs(A_derivs, a, b)
    D_a = first_derivative_coeffs(D_derivs, a)
    D_b = first_derivative_coeffs(D_derivs, b)
    D_ab = second_derivative_coeffs(D_derivs, a, b)

    term1 = cheb_mul_fast(A_ab, D2)
    advance_progress!(progress)

    A_a_D_b = cheb_mul_fast(A_a, D_b)
    advance_progress!(progress)
    A_b_D_a = cheb_mul_fast(A_b, D_a)
    advance_progress!(progress)
    A_D_ab = cheb_mul_fast(A, D_ab)
    advance_progress!(progress)

    term2_inner = cheb_add(cheb_add(A_a_D_b, A_b_D_a), A_D_ab)
    term2 = cheb_mul_fast(term2_inner, D)
    advance_progress!(progress)

    A_D_a = cheb_mul_fast(A, D_a)
    advance_progress!(progress)
    term3 = cheb_scale(cheb_mul_fast(A_D_a, D_b), interval_constant(2))
    advance_progress!(progress)

    return cheb_add(cheb_sub(term1, term2), term3)
end

function coeff_sup_bound_number(coeffs::AbstractMatrix{<:Number})
    return sup(chebyshev_coeff_sup_bound(intervalize_coefficients(coeffs)))
end

function tail_bound_number(tail)
    tail isa Interval && return sup(abs(tail))
    return abs(tail)
end

function scalar_abs_bound_number(c)
    c isa Interval && return sup(abs(c))
    return abs(c)
end

function truncated_coeff_enclosure(coeffs::AbstractMatrix{<:Number}, pdeg::Integer)
    trunc = truncate_coeffs_with_tail(coeffs, pdeg)
    return (; coeffs = trunc.coeffs, tail = tail_bound_number(trunc.tail), pdeg = trunc.pdeg)
end

function truncate_enclosure_coeffs(coeffs::AbstractMatrix{<:Number}, tail, pdeg::Integer)
    trunc = truncate_coeffs_with_tail(coeffs, pdeg)
    return (; coeffs = trunc.coeffs, tail = tail_bound_number(tail) + tail_bound_number(trunc.tail), pdeg = trunc.pdeg)
end

function enclosure_add(f, g, pdeg::Integer)
    return truncate_enclosure_coeffs(cheb_add(f.coeffs, g.coeffs), f.tail + g.tail, pdeg)
end

function enclosure_sub(f, g, pdeg::Integer)
    return truncate_enclosure_coeffs(cheb_sub(f.coeffs, g.coeffs), f.tail + g.tail, pdeg)
end

function enclosure_scale(f, c, pdeg::Integer)
    c_abs = scalar_abs_bound_number(c)
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

function quotient_second_derivative_numerator_enclosure(A_derivs, D_derivs, D2, a::Integer, b::Integer, pdeg::Integer; progress = nothing)
    A = A_derivs.f
    A_a = first_derivative_coeffs(A_derivs, a)
    A_b = first_derivative_coeffs(A_derivs, b)
    A_ab = second_derivative_coeffs(A_derivs, a, b)
    D = D_derivs.f
    D_a = first_derivative_coeffs(D_derivs, a)
    D_b = first_derivative_coeffs(D_derivs, b)
    D_ab = second_derivative_coeffs(D_derivs, a, b)

    A = truncated_coeff_enclosure(A, pdeg)
    A_a = truncated_coeff_enclosure(A_a, pdeg)
    A_b = truncated_coeff_enclosure(A_b, pdeg)
    A_ab = truncated_coeff_enclosure(A_ab, pdeg)
    D = truncated_coeff_enclosure(D, pdeg)
    D_a = truncated_coeff_enclosure(D_a, pdeg)
    D_b = truncated_coeff_enclosure(D_b, pdeg)
    D_ab = truncated_coeff_enclosure(D_ab, pdeg)

    term1 = enclosure_mul(A_ab, D2, pdeg; progress)

    A_a_D_b = enclosure_mul(A_a, D_b, pdeg; progress)
    A_b_D_a = enclosure_mul(A_b, D_a, pdeg; progress)
    A_D_ab = enclosure_mul(A, D_ab, pdeg; progress)

    term2_inner = enclosure_add(enclosure_add(A_a_D_b, A_b_D_a, pdeg), A_D_ab, pdeg)
    term2 = enclosure_mul(term2_inner, D, pdeg; progress)

    A_D_a = enclosure_mul(A, D_a, pdeg; progress)
    term3 = enclosure_scale(enclosure_mul(A_D_a, D_b, pdeg; progress), interval_constant(2), pdeg)

    return enclosure_add(enclosure_sub(term1, term2, pdeg), term3, pdeg)
end

function bound_rational_second_derivative(num::AbstractMatrix{<:Number}, D_lower)
    return chebyshev_coeff_sup_bound(num) / (D_lower^3)
end

function ricci_bounds_from_numerators(ricci_coeffs, D_lower)
    R11_num = ricci_coeffs.R11_num
    R12_num = ricci_coeffs.R12_num
    R21_num = ricci_coeffs.R21_num
    R22_num = ricci_coeffs.R22_num

    R11_bound = bound_rational_second_derivative(R11_num, D_lower)
    R12_bound = bound_rational_second_derivative(R12_num, D_lower)
    R21_bound = bound_rational_second_derivative(R21_num, D_lower)
    R22_bound = bound_rational_second_derivative(R22_num, D_lower)

    ricci_norm_squared_bound =
        interval_constant(2) * (
            R11_bound^2 +
            interval_constant(2) * R12_bound * R21_bound +
            R22_bound^2
        )
    ricci_norm_bound = sqrt(ricci_norm_squared_bound)

    return (;
        R11_num,
        R12_num,
        R21_num,
        R22_num,
        R11_bound,
        R12_bound,
        R21_bound,
        R22_bound,
        ricci_norm_squared_bound,
        ricci_norm_bound,
        D_lower,
    )
end

function compute_ricci_bound_from_inverse_coeffs(inverse_coeffs, inverse_bounds; progress = nothing, numerator_method::Symbol = :coefficient_space, pdeg::Integer = 0)
    D_lower = inverse_bounds.D_lower
    ricci_coeffs = compute_ricci_numerators_from_inverse_coeffs(inverse_coeffs; progress, method = numerator_method, pdeg)
    return ricci_bounds_from_numerators(ricci_coeffs, D_lower)
end

function compute_ricci_numerators_coefficient_space(inverse_coeffs; progress = nothing)
    A11 = inverse_coeffs.A11
    A12 = inverse_coeffs.A12
    A22 = inverse_coeffs.A22
    D = inverse_coeffs.D

    D2 = cheb_mul_fast(D, D)
    advance_progress!(progress)

    D_derivs = derivative_coeff_pack(D)
    A11_derivs = derivative_coeff_pack(A11)
    A12_derivs = derivative_coeff_pack(A12)
    A22_derivs = derivative_coeff_pack(A22)

    u11_xx_num = quotient_second_derivative_numerator(A11_derivs, D_derivs, D2, 1, 1; progress)
    u11_xy_num = quotient_second_derivative_numerator(A11_derivs, D_derivs, D2, 1, 2; progress)
    u12_xx_num = quotient_second_derivative_numerator(A12_derivs, D_derivs, D2, 1, 1; progress)
    u12_xy_num = quotient_second_derivative_numerator(A12_derivs, D_derivs, D2, 1, 2; progress)
    u12_yy_num = quotient_second_derivative_numerator(A12_derivs, D_derivs, D2, 2, 2; progress)
    u22_xy_num = quotient_second_derivative_numerator(A22_derivs, D_derivs, D2, 1, 2; progress)
    u22_yy_num = quotient_second_derivative_numerator(A22_derivs, D_derivs, D2, 2, 2; progress)

    half = interval_constant(-1) / exact(2)
    R11_num = cheb_scale(cheb_add(u11_xx_num, u12_xy_num), half)
    R12_num = cheb_scale(cheb_add(u11_xy_num, u12_yy_num), half)
    R21_num = cheb_scale(cheb_add(u12_xx_num, u22_xy_num), half)
    R22_num = cheb_scale(cheb_add(u12_xy_num, u22_yy_num), half)

    return (; R11_num, R12_num, R21_num, R22_num, D)
end

function compute_ricci_numerators_truncated_coefficient_space(inverse_coeffs; pdeg::Integer, progress = nothing)
    A11 = inverse_coeffs.A11
    A12 = inverse_coeffs.A12
    A22 = inverse_coeffs.A22
    D = inverse_coeffs.D

    pdeg > 0 || error("Truncated Ricci numerator construction requires pdeg > 0")

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

function common_ricci_numerator_degree(inverse_coeffs)
    D = inverse_coeffs.D
    A_degx = maximum(size(A, 1) for A in (inverse_coeffs.A11, inverse_coeffs.A12, inverse_coeffs.A22)) - 1
    A_degy = maximum(size(A, 2) for A in (inverse_coeffs.A11, inverse_coeffs.A12, inverse_coeffs.A22)) - 1
    D_degx = size(D, 1) - 1
    D_degy = size(D, 2) - 1
    return A_degx + 2 * D_degx, A_degy + 2 * D_degy
end

function lobatto_derivative_value_pack(coeffs::AbstractMatrix{<:Number}, degx::Integer, degy::Integer)
    coeffs_pad = cheb_pad(intervalize_coefficients(coeffs), degx, degy)
    values = cheb_coeffs_to_lobatto_values_2d(coeffs_pad)

    T = eltype(coeffs_pad)
    Dx = exact(2) .* cheb_diff_matrix(degx, T)
    Dy = -exact(2) .* cheb_diff_matrix(degy, T)

    x = Dx * values
    y = values * transpose(Dy)
    xx = Dx * x
    xy = Dx * y
    yy = y * transpose(Dy)

    return (; f = values, x, y, xx, xy, yy)
end

first_derivative_values(pack, dim::Integer) =
    dim == 1 ? pack.x :
    dim == 2 ? pack.y :
    error("Expected derivative dimension 1 or 2")

second_derivative_values(pack, a::Integer, b::Integer) =
    a == 1 && b == 1 ? pack.xx :
    a == 2 && b == 2 ? pack.yy :
    a != b ? pack.xy :
    error("Expected derivative dimensions 1 or 2")

function quotient_second_derivative_numerator_values(A_derivs, D_derivs, a::Integer, b::Integer)
    A = A_derivs.f
    D = D_derivs.f
    A_a = first_derivative_values(A_derivs, a)
    A_b = first_derivative_values(A_derivs, b)
    A_ab = second_derivative_values(A_derivs, a, b)
    D_a = first_derivative_values(D_derivs, a)
    D_b = first_derivative_values(D_derivs, b)
    D_ab = second_derivative_values(D_derivs, a, b)
    D2 = D .* D

    return A_ab .* D2 .-
        (A_a .* D_b .+ A_b .* D_a .+ A .* D_ab) .* D .+
        interval_constant(2) .* (A .* D_a) .* D_b
end

function compute_ricci_numerators_real_space(inverse_coeffs; progress = nothing)
    A11 = inverse_coeffs.A11
    A12 = inverse_coeffs.A12
    A22 = inverse_coeffs.A22
    D = inverse_coeffs.D

    degx, degy = common_ricci_numerator_degree(inverse_coeffs)

    D_derivs = lobatto_derivative_value_pack(D, degx, degy)
    advance_progress!(progress)
    A11_derivs = lobatto_derivative_value_pack(A11, degx, degy)
    advance_progress!(progress)
    A12_derivs = lobatto_derivative_value_pack(A12, degx, degy)
    advance_progress!(progress)
    A22_derivs = lobatto_derivative_value_pack(A22, degx, degy)
    advance_progress!(progress)

    u11_xx_num = quotient_second_derivative_numerator_values(A11_derivs, D_derivs, 1, 1)
    advance_progress!(progress)
    u11_xy_num = quotient_second_derivative_numerator_values(A11_derivs, D_derivs, 1, 2)
    advance_progress!(progress)
    u12_xx_num = quotient_second_derivative_numerator_values(A12_derivs, D_derivs, 1, 1)
    advance_progress!(progress)
    u12_xy_num = quotient_second_derivative_numerator_values(A12_derivs, D_derivs, 1, 2)
    advance_progress!(progress)
    u12_yy_num = quotient_second_derivative_numerator_values(A12_derivs, D_derivs, 2, 2)
    advance_progress!(progress)
    u22_xy_num = quotient_second_derivative_numerator_values(A22_derivs, D_derivs, 1, 2)
    advance_progress!(progress)
    u22_yy_num = quotient_second_derivative_numerator_values(A22_derivs, D_derivs, 2, 2)
    advance_progress!(progress)

    half = interval_constant(-1) / exact(2)
    R11_num_values = half .* (u11_xx_num .+ u12_xy_num)
    R12_num_values = half .* (u11_xy_num .+ u12_yy_num)
    R21_num_values = half .* (u12_xx_num .+ u22_xy_num)
    R22_num_values = half .* (u12_xy_num .+ u22_yy_num)

    R11_num = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(R11_num_values))
    advance_progress!(progress)
    R12_num = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(R12_num_values))
    advance_progress!(progress)
    R21_num = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(R21_num_values))
    advance_progress!(progress)
    R22_num = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(R22_num_values))
    advance_progress!(progress)

    return (; R11_num, R12_num, R21_num, R22_num, D)
end

function compute_ricci_numerators_from_inverse_coeffs(inverse_coeffs; progress = nothing, method::Symbol = :real_space, pdeg::Integer = 0)
    if method === :coefficient_space
        return compute_ricci_numerators_coefficient_space(inverse_coeffs; progress)
    elseif method === :truncated_coefficient_space
        return compute_ricci_numerators_truncated_coefficient_space(inverse_coeffs; pdeg, progress)
    elseif method === :real_space
        return compute_ricci_numerators_real_space(inverse_coeffs; progress)
    else
        error("Unknown Ricci numerator method: $method")
    end
end

function prepare_ricci_coeffs_for_subdivision(ricci_coeffs, pdeg::Integer)
    return (;
        R11_num = truncate_coeffs_with_tail(ricci_coeffs.R11_num, pdeg),
        R12_num = truncate_coeffs_with_tail(ricci_coeffs.R12_num, pdeg),
        R21_num = truncate_coeffs_with_tail(ricci_coeffs.R21_num, pdeg),
        R22_num = truncate_coeffs_with_tail(ricci_coeffs.R22_num, pdeg),
        D       = truncate_coeffs_with_tail(ricci_coeffs.D,       pdeg),
    )
end

function ricci_entry_interval_from_box(num_box, D_box)
    D_lower = inf(D_box)

    if !isfinite(D_lower) || D_lower <= 0
        error("Could not certify positivity of D on Ricci subdivision box: D_box = $D_box, inf(D_box) = $D_lower")
    end

    return num_box / (D_box^3)
end

function ricci_norm_squared_interval(R11_box, R12_box, R21_box, R22_box)
    return interval_constant(2) * (
        R11_box^2 +
        interval_constant(2) * R12_box * R21_box +
        R22_box^2
    )
end

function ricci_norm_upper_from_squared(norm_squared_box)
    upper = sup(norm_squared_box)
    upper < 0 && error("Ricci norm-squared enclosure has negative upper bound: $norm_squared_box")
    return sqrt(upper)
end

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

        R11_box = ricci_entry_interval_from_box(R11_num_box, D_box)
        R12_box = ricci_entry_interval_from_box(R12_num_box, D_box)
        R21_box = ricci_entry_interval_from_box(R21_num_box, D_box)
        R22_box = ricci_entry_interval_from_box(R22_num_box, D_box)

        R11_bound = sup(abs(R11_box))
        R12_bound = sup(abs(R12_box))
        R21_bound = sup(abs(R21_box))
        R22_bound = sup(abs(R22_box))

        ricci_norm_squared_box = ricci_norm_squared_interval(R11_box, R12_box, R21_box, R22_box)
        ricci_norm_squared_bound = sup(ricci_norm_squared_box)
        ricci_norm_bound = ricci_norm_upper_from_squared(ricci_norm_squared_box)

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

function print_ricci_bound_summary(ricci_bound)
    println()
    println("Certified C^0 bound for Ricci:")
    println("    ||Ric||_∞ <= $(bound_upper(ricci_bound.ricci_norm_bound))")
    println("    ||Ric||_∞^2 <= $(bound_upper(ricci_bound.ricci_norm_squared_bound))")
    println()
    println("Ricci entry bounds:")
    println("    ||R^1_1||_∞ <= $(bound_upper(ricci_bound.R11_bound))")
    println("    ||R^1_2||_∞ <= $(bound_upper(ricci_bound.R12_bound))")
    println("    ||R^2_1||_∞ <= $(bound_upper(ricci_bound.R21_bound))")
    println("    ||R^2_2||_∞ <= $(bound_upper(ricci_bound.R22_bound))")
end
