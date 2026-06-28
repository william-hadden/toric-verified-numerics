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

"""
Assemble inverse metric entry enclosures from enclosed numerator components.

The inverse metric entries are represented as

    u^{11} = A11 / D,  u^{12} = u^{21} = A12 / D,  u^{22} = A22 / D.

`A11_box`, `A12_box`, `A22_box`, and `D_box` must be interval enclosures of
the corresponding quantities on the same physical box. The denominator is
first certified positive by checking `inf(D_box) > 0`; if this fails, an error
is thrown with `box_label` included in the diagnostic message when supplied.

Returns a named tuple containing the original component enclosures, `D_lower`,
and the interval quotients `u11_box`, `u12_box`, `u21_box`, and `u22_box`.
The lower and upper bounds for an entry are the endpoints of its interval
quotient, for example `inf(result.u11_box)` and `sup(result.u11_box)`.
"""
function inverse_metric_box_from_component_enclosures(
    A11_box,
    A12_box,
    A22_box,
    D_box;
    box_label::AbstractString = "",
)
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

"""
Bound all inverse metric entries on one Chebyshev-coordinate box.

`inverse_coeffs` must provide full Chebyshev coefficient arrays named `A11`,
`A12`, `A22`, and `D`. The input intervals `xbox` and `ybox` are boxes in the
Chebyshev variables of these arrays.

This function does not truncate the coefficient arrays. Each polynomial is
rewritten on the local box in coordinates ranging over `[-1, 1]^2`; the
constant term gives the centered value and the sum of absolute nonconstant
coefficients gives a rigorous local tail. The resulting numerator and
denominator enclosures are passed to
`inverse_metric_box_from_component_enclosures`, which certifies `D > 0` and
forms the interval quotients.
"""
function inverse_metric_box_bounds(inverse_coeffs, xbox, ybox; box_label::AbstractString = "")
    A11_box = local_coeff_sum_centered_enclosure_cheb_2d(inverse_coeffs.A11, xbox, ybox)
    A12_box = local_coeff_sum_centered_enclosure_cheb_2d(inverse_coeffs.A12, xbox, ybox)
    A22_box = local_coeff_sum_centered_enclosure_cheb_2d(inverse_coeffs.A22, xbox, ybox)
    D_box   = local_coeff_sum_centered_enclosure_cheb_2d(inverse_coeffs.D,   xbox, ybox)

    return inverse_metric_box_from_component_enclosures(
        A11_box,
        A12_box,
        A22_box,
        D_box;
        box_label,
    )
end

"""
Bound all inverse metric entries on one box using truncated coefficients.

`trunc` must contain entries `A11`, `A12`, `A22`, and `D`, each of the form
returned by `truncate_coeffs_with_tail`: a retained coefficient array plus a
global discarded-coefficient tail. The retained coefficients are enclosed on
`xbox x ybox` using the same local centered enclosure as
`inverse_metric_box_bounds`, then widened by the stored tail before forming
the quotient intervals.

This helper preserves the behavior of the existing subdivision-wide upper
bound routine.
"""
function inverse_metric_box_bounds_with_tail(trunc, xbox, ybox; box_label::AbstractString = "")
    A11_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A11, xbox, ybox)
    A12_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A12, xbox, ybox)
    A22_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A22, xbox, ybox)
    D_box   = local_coeff_sum_centered_enclosure_with_tail(trunc.D,   xbox, ybox)

    return inverse_metric_box_from_component_enclosures(
        A11_box,
        A12_box,
        A22_box,
        D_box;
        box_label,
    )
end

# worker
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
        box = inverse_metric_box_bounds_with_tail(
            trunc,
            xbox,
            ybox;
            box_label = "box (ix=$ix, iy=$iy)",
        )

        u11_box_bound = sup(abs(box.u11_box))
        u12_box_bound = sup(abs(box.u12_box))
        u22_box_bound = sup(abs(box.u22_box))

        global_u11 = max(global_u11, u11_box_bound)
        global_u12 = max(global_u12, u12_box_bound)
        global_u22 = max(global_u22, u22_box_bound)
        global_D_lower = min(global_D_lower, box.D_lower)

        push!(box_results, (;
            ix,
            iy,
            xbox,
            ybox,
            A11_box = box.A11_box,
            A12_box = box.A12_box,
            A22_box = box.A22_box,
            D_box = box.D_box,
            D_lower = box.D_lower,
            u11_box = box.u11_box,
            u12_box = box.u12_box,
            u21_box = box.u21_box,
            u22_box = box.u22_box,
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

function print_subdivision_inverse_bound_summary(step)
    println()
    println("Subdivision-certified C^0 bounds for inverse metric entries:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    ||u^{11}||_inf <= $(step.u11_bound)")
    println("    ||u^{12}||_inf <= $(step.u12_bound)")
    println("    ||u^{22}||_inf <= $(step.u22_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
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
