using IntervalArithmetic
using Logging

global_logger(NullLogger())

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "util", "inverse_bounds.jl"))
include(joinpath(@__DIR__, "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "util", "inverse_subdivision_bounds.jl"))
include(joinpath(@__DIR__, "util", "curvature_bounds.jl"))

"""
Bound the `C^0` norm of the Ricci tensor using truncated local subdivision.
"""
function bound_ricci_subdivision_local(
    coeffs_path::AbstractString,
    pdeg::Integer,
    num_subdivisions::Integer;
)
    println("Step 1: Load u0")
    step1 = load_rational_coeffs_csv(coeffs_path)
    println("Step 1: Load u0 ... ok")
    println("Step 1 guaranteed: $(all(isguaranteed, step1))")

    step2_progress = start_progress("Step 2: Compute second derivatives", 1)
    # step2 = compute_second_derivatives(step1)
    step2 = build_derivative_pack(step1)
    finish_progress!(step2_progress)
    println("Step 2: Compute second derivatives ... ok")
    step2_guaranteed =
        all(isguaranteed, step2.uxx) &&
        all(isguaranteed, step2.uxy) &&
        all(isguaranteed, step2.uyy)
    println("Step 2 guaranteed: $step2_guaranteed")

    step3_progress = start_progress("Step 3: Build inverse metric coefficients", 74)
    step3 = build_inverse_metric_coeffs(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build inverse metric coefficient arrays ... ok")
    step3_guaranteed =
        all(isguaranteed, step3.A11) &&
        all(isguaranteed, step3.A12) &&
        all(isguaranteed, step3.A22) &&
        all(isguaranteed, step3.D)
    println("Step 3 guaranteed: $step3_guaranteed")

    step4_progress = start_progress("Step 4: Ricci numerator products", 65)
    step4 = compute_ricci_numerators_truncated_coefficient_space(step3; progress = step4_progress, pdeg)
    finish_progress!(step4_progress)
    println("Step 4: Build Ricci numerator coefficient arrays ... ok")
    step4_pack_guaranteed = all(e -> all(isguaranteed, e.coeffs), values(step4.pack.data))
    step4_guaranteed =
        all(isguaranteed, step4.R11_num) &&
        all(isguaranteed, step4.R12_num) &&
        all(isguaranteed, step4.R21_num) &&
        all(isguaranteed, step4.R22_num) &&
        all(isguaranteed, step4.ricci_norm_squared_num) &&
        all(isguaranteed, step4.D) &&
        step4_pack_guaranteed
    println("Step 4 guaranteed: $step4_guaranteed")

    println("Step 5: Starting local Ricci subdivision bounds ...")
    step5_progress = start_progress("Step 5: Ricci subdivision boxes", num_subdivisions * num_subdivisions)
    subdivision_time = @elapsed begin
        step5 = compute_ricci_bound_by_local_subdivision_truncated(
            step4;
            pdeg = pdeg,
            nx = num_subdivisions,
            ny = num_subdivisions,
            progress = step5_progress,
        )
        global_step5 = step5
    end
    finish_progress!(step5_progress)

    println("Step 5: Local Ricci subdivision bounds ... ok")
    print_truncated_subdivision_ricci_bound_summary(global_step5)
    println()
    println("Ricci subdivision step time: $(round(subdivision_time; digits = 3)) seconds")

    trunc_guaranteed = all(
        t -> all(isguaranteed, t.coeffs) && isguaranteed(t.tail),
        values(global_step5.trunc),
    )
    boxes_guaranteed = all(global_step5.box_results) do box
        all(isguaranteed, (
            box.xbox,
            box.ybox,
            box.R11_num_box,
            box.R12_num_box,
            box.R21_num_box,
            box.R22_num_box,
            box.D_box,
            box.R11_box,
            box.R12_box,
            box.R21_box,
            box.R22_box,
            box.ricci_norm_squared_num_box,
            box.ricci_norm_squared_box,
        ))
    end
    println("Step 5 interval calculations guaranteed: " * "$(trunc_guaranteed && boxes_guaranteed)")

    return (;
        step1,
        step2,
        step3,
        step4,
        step5 = global_step5,
        subdivision_time,
        pdeg,
        num_subdivisions,
    )
end

function bound_ricci_from_below(coeffs_path::AbstractString,
    pdeg::Integer,
    num_subdivisions::Integer;
)
    println("Step 1: Load u0")
    step1 = load_rational_coeffs_csv(coeffs_path)
    println("Step 1: Load u0 ... ok")
    println("Step 1 guaranteed: $(all(isguaranteed, step1))")

    step2_progress = start_progress("Step 2: Compute second derivatives", 1)
    # step2 = compute_second_derivatives(step1)
    step2 = build_derivative_pack(step1)
    finish_progress!(step2_progress)
    println("Step 2: Compute second derivatives ... ok")
    step2_guaranteed =
        all(isguaranteed, step2.uxx) &&
        all(isguaranteed, step2.uxy) &&
        all(isguaranteed, step2.uyy)
    println("Step 2 guaranteed: $step2_guaranteed")

    step3_progress = start_progress("Step 3: Build inverse metric coefficients", 74)
    step3 = build_inverse_metric_coeffs(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build inverse metric coefficient arrays ... ok")
    step3_guaranteed =
        all(isguaranteed, step3.A11) &&
        all(isguaranteed, step3.A12) &&
        all(isguaranteed, step3.A22) &&
        all(isguaranteed, step3.D)
    println("Step 3 guaranteed: $step3_guaranteed")

    step4_progress = start_progress("Step 4: Ricci-minus-identity numerator products", 152)
    step4 = compute_ricci_minus_identity_numerator_truncated_coefficient_space(
        step3;
        pdeg,
        progress = step4_progress,
    )
    finish_progress!(step4_progress)
    println("Step 4: Build Ricci-minus-identity numerator array ... ok")
    step4_pack_guaranteed = all(e -> all(isguaranteed, e.coeffs), values(step4.pack.data))
    step4_guaranteed =
        all(isguaranteed, step4.norm_minus_id_num) &&
        all(isguaranteed, step4.D) &&
        step4_pack_guaranteed
    println("Step 4 guaranteed: $step4_guaranteed")

    println("Step 5: Starting local Ricci lower-bound subdivision ...")
    step5_progress = start_progress("Step 5: Ricci lower-bound subdivision boxes", num_subdivisions * num_subdivisions)
    subdivision_time = @elapsed begin
        step5 = compute_ricci_minus_identity_bound_by_local_subdivision_truncated(
            step4;
            pdeg = pdeg,
            nx = num_subdivisions,
            ny = num_subdivisions,
            progress = step5_progress,
        )
        global_step5 = step5
    end
    finish_progress!(step5_progress)

    println("Step 5: Local Ricci lower-bound     subdivision ... ok")
    print_truncated_subdivision_ricci_lower_bound_summary(global_step5)
    println()
    println("Ricci lower-bound subdivision step time: $(round(subdivision_time; digits = 3)) seconds")

    trunc_guaranteed = all(
        t -> all(isguaranteed, t.coeffs) && isguaranteed(t.tail),
        values(global_step5.trunc),
    )
    boxes_guaranteed = all(global_step5.box_results) do box
        all(isguaranteed, (
            box.xbox,
            box.ybox,
            box.D_box,
            box.norm_minus_id_num_box,
            box.norm_squared_box,
        ))
    end
    println("Step 5 interval calculations guaranteed: " * "$(trunc_guaranteed && boxes_guaranteed)")

    return global_step5.ricci_lower_bound
end

function main()
    setprecision(BigFloat, 100)
    # bound_ricci_subdivision_local(U0_PATH, 10, 4)
    # bound_ricci_from_below(U0_PATH, 10, 4)
    bound_ricci_from_below(U0_PATH, 30, 20)
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
