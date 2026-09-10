using IntervalArithmetic
using Logging

global_logger(NullLogger())

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "util", "inverse_subdivision_bounds.jl"))


function subdivide_and_bound_rigorous_local(
    coeffs_path::AbstractString,
    pdeg::Integer = 20,
    num_subdivisions::Integer = 8,
)
    println("Step 1: Load u0")
    step1 = load_rational_coeffs_csv(coeffs_path)
    println("Step 1: Load u0 ... ok")
    println("Step 1 guaranteed: $(all(isguaranteed, step1))")

    step2_progress = start_progress("Step 2: Compute derivatives", 1)
    # step2 = compute_second_derivatives(step1)
    step2 = build_derivative_pack(step1)
    finish_progress!(step2_progress)
    println("Step 2: Compute derivatives ... ok")
    step2_guaranteed =
        all(isguaranteed, step2.uxx) &&
        all(isguaranteed, step2.uxy) &&
        all(isguaranteed, step2.uyy)
    println("Step 2 guaranteed: $step2_guaranteed")

    step3_progress = start_progress("Step 3: Build full inverse metric coefficients", 74)
    step3 = build_inverse_metric_coeffs(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build full inverse metric coefficient arrays ... ok")
    step3_guaranteed =
        all(isguaranteed, step3.A11) &&
        all(isguaranteed, step3.A12) &&
        all(isguaranteed, step3.A22) &&
        all(isguaranteed, step3.D)
    println("Step 3 guaranteed: $step3_guaranteed")

    println("Step 4: Starting local subdivision bounds ...")
    step4_progress = start_progress("Step 4: Local subdivision boxes", num_subdivisions * num_subdivisions)
    subdivision_time = @elapsed begin
        step4 = compute_inverse_bound_by_local_subdivision_truncated(
            step3;
            pdeg = pdeg,
            nx = num_subdivisions,
            ny = num_subdivisions,
            progress = step4_progress,
        )
        global_step4 = step4
    end
    finish_progress!(step4_progress)
    println("Step 4: Local subdivision bounds ... ok")
    print_truncated_subdivision_inverse_bound_summary(global_step4)
    println()
    println("Subdivision step time: $(round(subdivision_time; digits = 3)) seconds")
    boxes_guaranteed = all(global_step4.box_results) do box
        all(isguaranteed, (
            box.xbox,
            box.ybox,
            box.A11_box,
            box.A12_box,
            box.A22_box,
            box.D_box,
            box.u11_box,
            box.u12_box,
            box.u22_box,
        ))
    end
    println("Step 4 interval calculations guaranteed: $boxes_guaranteed")

    return (;
        step1,
        step2,
        step3,
        step4 = global_step4,
        subdivision_time,
        pdeg,
        num_subdivisions,
    )
end

function main()
    setprecision(BigFloat, 256)
    subdivide_and_bound_rigorous_local(U0_PATH);
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
