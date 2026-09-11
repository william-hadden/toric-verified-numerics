using IntervalArithmetic
using Logging

global_logger(NullLogger())

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "util", "inverse_subdivision_bounds.jl"))
include(joinpath(@__DIR__, "util", "curvature_bounds.jl"))

"""
Bound the `C^0` norm of the Riemann tensor using truncated local subdivision.
"""
function bound_riem_subdivision_local(
    coeffs_path::AbstractString,
    pdeg::Integer = 20,
    num_subdivisions::Integer = 8;
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

    step4_progress = start_progress("Step 4: Riemmanian numerator products", 136)
    step4 = compute_riem_numerators_truncated_coeff_space(step3; progress = step4_progress, pdeg)
    finish_progress!(step4_progress)
    println("Step 4: Build Riemmanian numerator coefficient arrays ... ok")
    
    println("Step 5: Starting local Riem subdivision bounds ...")
    step5_progress = start_progress("Step 5: Riem subdivision boxes", num_subdivisions * num_subdivisions)
    subdivision_time = @elapsed begin
        step5 = compute_riem_by_local_subdivision_truncated(
            step4;
            pdeg = pdeg,
            nx = num_subdivisions,
            ny = num_subdivisions,
            progress = step5_progress,
        )
        global_step5 = step5
    end
    finish_progress!(step5_progress)

    println("Step 5: Local Riem subdivision bounds ... ok")
    print_truncated_subdivision_riem_bound_summary(global_step5)
    println()
    println("Riemann subdivision step time: $(round(subdivision_time; digits = 3)) seconds")

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

"""
Bound the `C^0` norm of the cov Riemann tensor using truncated local subdivision.
"""
function bound_cov_riem_subdivision_local(
    coeffs_path::AbstractString,
    pdeg::Integer = 20,
    num_subdivisions::Integer = 8;
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

    step4_progress = start_progress("Step 4: Cov Riemmanian numerator products", 936)
    step4 = compute_inverse_derivative_numerator_components_truncated_coeff_space(step3;k = 3, pdeg, progress = step4_progress,)
    finish_progress!(step4_progress)
    println("Step 4: Build cov riem numerator coefficient arrays ... ok")
    
    println("Step 5: Starting local cov Riem subdivision bounds ...")
    step5_progress = start_progress("Step 5: cov Riem subdivision boxes", num_subdivisions * num_subdivisions)
    subdivision_time = @elapsed begin
        step5 = compute_cov_riem_by_local_subdivision_truncated(
            step4;
            pdeg = pdeg,
            nx = num_subdivisions,
            ny = num_subdivisions,
            progress = step5_progress,
        )
        global_step5 = step5
    end
    finish_progress!(step5_progress)

    println("Step 5: Local cov Riem subdivision bounds ... ok")
    print_truncated_subdivision_cov_riem_bound_summary(global_step5)
    println()
    println("Cov Riemann subdivision step time: $(round(subdivision_time; digits = 3)) seconds")

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

"""
Bound the `C^0` norm of the cov Riemann tensor using truncated local subdivision.
"""
function bound_cov_cov_riem_subdivision_local(
    coeffs_path::AbstractString,
    pdeg::Integer = 20,
    num_subdivisions::Integer = 8;
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

    step4_progress = start_progress("Step 4:Cov Cov Riemmanian numerator products", 8132)
    step4 = compute_inverse_derivative_numerator_components_truncated_coeff_space(step3;k = 4, pdeg, progress = step4_progress,)
    finish_progress!(step4_progress)
    println("Step 4: Build cov riem numerator coefficient arrays ... ok")
    
    println("Step 5: Starting local cov cov Riem subdivision bounds ...")
    step5_progress = start_progress("Step 5: cov cov Riem subdivision boxes", num_subdivisions * num_subdivisions)
    subdivision_time = @elapsed begin
        step5 = compute_cov_cov_riem_by_local_subdivision_truncated(
            step4;
            pdeg = pdeg,
            nx = num_subdivisions,
            ny = num_subdivisions,
            progress = step5_progress,
        )
        global_step5 = step5
    end
    finish_progress!(step5_progress)

    println("Step 5: Local cov cov Riem subdivision bounds ... ok")
    print_truncated_subdivision_cov_cov_riem_bound_summary(global_step5)
    println()
    println("Cov Riemann subdivision step time: $(round(subdivision_time; digits = 3)) seconds")

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

function main(; coeffs_path::AbstractString = U0_PATH, pdeg::Integer = 20,
    num_subdivisions::Integer = 8, output_path::AbstractString = VERIFIED_BOUNDS_PATH)
    setprecision(BigFloat, 256)
    riem = bound_riem_subdivision_local(coeffs_path, pdeg, num_subdivisions)
    write_bound_entry("curvature_bounds", "riemann_C0", riem.step5.riem_norm_bound; path = output_path, rounding = RoundUp)
    cov_riem = bound_cov_riem_subdivision_local(coeffs_path, pdeg, num_subdivisions)
    write_bound_entry("curvature_bounds", "nabla_riemann_C0", cov_riem.step5.cov_riem_norm_bound; path = output_path, rounding = RoundUp)
    cov_cov_riem = bound_cov_cov_riem_subdivision_local(coeffs_path, pdeg, num_subdivisions)
    write_bound_entry("curvature_bounds", "nabla2_riemann_C0", cov_cov_riem.step5.cov_cov_riem_norm_bound; path = output_path, rounding = RoundUp)
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
