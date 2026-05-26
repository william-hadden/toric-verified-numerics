function local_coeff_sum_centered_enclosure_with_tail(trunc, xbox, ybox)
    val = local_coeff_sum_centered_enclosure_cheb_2d(trunc.coeffs, xbox, ybox)
    return val + symmetric_interval(trunc.tail)
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
        A11_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A11, xbox, ybox)
        A12_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A12, xbox, ybox)
        A22_box = local_coeff_sum_centered_enclosure_with_tail(trunc.A22, xbox, ybox)
        D_box   = local_coeff_sum_centered_enclosure_with_tail(trunc.D,   xbox, ybox)

        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        u11_box_bound = sup(abs(A11_box)) / D_lower
        u12_box_bound = sup(abs(A12_box)) / D_lower
        u22_box_bound = sup(abs(A22_box)) / D_lower

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
    println("    inf D >= $(inf(step.D_lower))")
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
