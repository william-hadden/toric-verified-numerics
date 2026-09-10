"""
Evaluate a truncated Chebyshev enclosure on a subdivision box.

`trunc` is expected to be a `(; coeffs, tail, pdeg)` object, such as the output
of `truncate_coeffs_with_tail`. The retained coefficients are enclosed on the
local interval box `xbox x ybox`, then the result is widened by the symmetric
interval `[-tail, tail]` to account rigorously for discarded coefficients.
"""
function local_coeff_sum_centered_enclosure_with_tail(trunc, xbox, ybox)
    val = local_coeff_sum_centered_enclosure_cheb_2d(trunc.coeffs, xbox, ybox)
    return val + symmetric_interval(trunc.tail)
end

"""Bound inverse metric entries on one truncated-coefficient box."""
function inverse_metric_box_bounds_with_tail(trunc, xbox, ybox; box_label::AbstractString = "")
    A11_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A11, xbox, ybox)
    A12_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A12, xbox, ybox)
    A22_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A22, xbox, ybox)
    D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)
    D_lower = inf(D_box)

    if !isfinite(D_lower) || D_lower <= 0
        label = isempty(box_label) ? "" : " on $box_label"
        error(
            "Could not certify positivity of D$label: " *
            "D_box = $D_box, inf(D_box) = $D_lower",
        )
    end

    u11_box = A11_box / D_box
    u12_box = A12_box / D_box
    u22_box = A22_box / D_box
    return (;
        A11_box,
        A12_box,
        A22_box,
        D_box,
        D_lower,
        u11_box,
        u12_box,
        u21_box = u12_box,
        u22_box,
    )
end

function compute_inverse_bound_by_local_subdivision_truncated(
    inverse_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = prepare_inverse_coeffs_for_subdivision(inverse_coeffs, pdeg)

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_u11 = big"0"
    global_u12 = big"0"
    global_u22 = big"0"
    global_D_lower = big"Inf"

    box_results = []

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        A11_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A11, xbox, ybox)
        A12_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A12, xbox, ybox)
        A22_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A22, xbox, ybox)
        D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)
        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on box (ix=$ix, iy=$iy): " *
                "D_box = $D_box, inf(D_box) = $D_lower",
            )
        end

        u11_box = A11_box / D_box
        u12_box = A12_box / D_box
        u22_box = A22_box / D_box

        u11_box_bound = sup(abs(u11_box))
        u12_box_bound = sup(abs(u12_box))
        u22_box_bound = sup(abs(u22_box))

        global_u11 = max(global_u11, u11_box_bound)
        global_u12 = max(global_u12, u12_box_bound)
        global_u22 = max(global_u22, u22_box_bound)
        global_D_lower = min(global_D_lower, D_lower)

        push!(box_results, (;
            ix,
            iy,
            xbox,
            ybox,
            A11_box,
            A12_box,
            A22_box,
            D_box,
            D_lower,
            u11_box,
            u12_box,
            u21_box = u12_box,
            u22_box,
            u11_box_bound,
            u12_box_bound,
            u22_box_bound,
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
        u11_bound = global_u11,
        u12_bound = global_u12,
        u22_bound = global_u22,
    )
end

"""Bound inverse metric derivatives using their existing numerator enclosures."""
function bound_inverse_derivatives_by_local_subdivision(
    inverse_derivative_coeffs;
    pdeg::Integer,
    nx::Integer,
    ny::Integer = nx,
    progress = nothing,
)
    deriv_num = Dict(key => truncate_coeffs_with_tail(coeffs, pdeg)
        for (key, coeffs) in inverse_derivative_coeffs.deriv_num)
    D = truncate_coeffs_with_tail(inverse_derivative_coeffs.D, pdeg)
    derivative_bounds = Dict(key => BigFloat(0) for key in keys(deriv_num))
    global_D_lower = big"Inf"

    for xbox in subdivide_minus_one_one(nx), ybox in subdivide_minus_one_one(ny)
        D_box = local_coeff_sum_centered_enclosure_with_tail(D, xbox, ybox)
        D_lower = inf(D_box)
        isfinite(D_lower) && D_lower > 0 || error(
            "Could not certify positivity of D for inverse derivative bounds: D_box = $D_box",
        )
        for (key, numerator) in deriv_num
            a = key[3]
            num_box = local_coeff_sum_centered_enclosure_with_tail(numerator, xbox, ybox)
            derivative_box = num_box / D_box^(sum(a) + 1)
            derivative_bounds[key] = max(derivative_bounds[key], sup(abs(derivative_box)))
        end
        global_D_lower = min(global_D_lower, D_lower)
        advance_progress!(progress)
    end
    return (; derivative_bounds, D_lower = global_D_lower)
end

"""Arrange inverse derivative bounds in the schema consumed by the residual stage."""
function metric_inverse_bounds(derivative_bounds)
    components = ("xx" => (1, 1), "xy" => (1, 2), "yx" => (2, 1), "yy" => (2, 2))
    entries(a) = Dict(label => derivative_bounds[(i, j, a)] for (label, (i, j)) in components)
    return (;
        value = entries((0, 0)),
        d1 = Dict("x" => entries((1, 0)), "y" => entries((0, 1))),
        d2 = Dict("xx" => entries((2, 0)), "xy" => entries((1, 1)),
            "yx" => entries((1, 1)), "yy" => entries((0, 2))),
    )
end

function print_truncated_subdivision_inverse_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bounds for inverse metric entries:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||u^{11}||_inf <= $(step.u11_bound)")
    println("    ||u^{12}||_inf <= $(step.u12_bound)")
    println("    ||u^{22}||_inf <= $(step.u22_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Coefficient tail bounds:")
    println("    A11 tail <= $(step.trunc.A11.tail)")
    println("    A12 tail <= $(step.trunc.A12.tail)")
    println("    A22 tail <= $(step.trunc.A22.tail)")
    println("    D tail   <= $(step.trunc.D.tail)")
end
