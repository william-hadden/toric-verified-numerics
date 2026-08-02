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
    step1_progress = start_progress("Step 1: Load u0", filesize(coeffs_path))
    step1 = load_rational_coeffs_csv(coeffs_path; progress = step1_progress)
    finish_progress!(step1_progress)
    println("Step 1: Load u0 ... ok")

    step2_progress = start_progress("Step 2: Compute second derivatives", 1)
    step2 = compute_second_derivatives(step1)
    finish_progress!(step2_progress)
    println("Step 2: Compute second derivatives ... ok")

    step3_progress = start_progress("Step 3: Build inverse metric coefficients", 74)
    step3 = build_inverse_metric_coeffs(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build inverse metric coefficient arrays ... ok")

    step4_progress = start_progress("Step 4: Ricci numerator products", 65)
    step4 = compute_ricci_numerators_truncated_coefficient_space(step3; progress = step4_progress, pdeg)
    finish_progress!(step4_progress)
    println("Step 4: Build Ricci numerator coefficient arrays ... ok")

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
    num_subdivisions::Integer;)
    step1_progress = start_progress("Step 1: Load u0", filesize(coeffs_path))
    step1 = load_rational_coeffs_csv(coeffs_path; progress = step1_progress)
    finish_progress!(step1_progress)
    println("Step 1: Load u0 ... ok")

    step2_progress = start_progress("Step 2: Compute second derivatives", 1)
    step2 = compute_second_derivatives(step1)
    finish_progress!(step2_progress)
    println("Step 2: Compute second derivatives ... ok")

    step3_progress = start_progress("Step 3: Build inverse metric coefficients", 74)
    step3 = build_inverse_metric_coeffs(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build inverse metric coefficient arrays ... ok")

    step4_progress = start_progress("Step 4: Ricci-minus-identity numerator products", 152)
    step4 = compute_ricci_minus_identity_numerator_truncated_coefficient_space(
        step3;
        pdeg,
        progress = step4_progress,
    )
    finish_progress!(step4_progress)
    println("Step 4: Build Ricci-minus-identity numerator array ... ok")

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

    println("Step 5: Local Ricci lower-bound subdivision ... ok")
    print_truncated_subdivision_ricci_lower_bound_summary(global_step5)
    println()
    println("Ricci lower-bound subdivision step time: $(round(subdivision_time; digits = 3)) seconds")

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
