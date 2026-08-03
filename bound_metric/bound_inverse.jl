using IntervalArithmetic
using Logging

global_logger(NullLogger())

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "..", "bound_residual", "util", "problem.jl"))
include(joinpath(@__DIR__, "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "util", "inverse_bounds.jl"))
include(joinpath(@__DIR__, "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "util", "inverse_subdivision_bounds.jl"))


function subdivide_and_bound_rigorous_local(
    coeffs_path::AbstractString,
    pdeg::Integer,
    num_subdivisions::Integer,
)
    println("Step 1: Load u0")
    step1 = load_rational_coeffs_csv(coeffs_path; progress = step1_progress)
    println("Step 1: Load u0 ... ok")

    step2_progress = start_progress("Step 2: Compute full second derivatives", 1)
    step2 = compute_second_derivatives(step1)
    finish_progress!(step2_progress)
    println("Step 2: Compute full second derivatives ... ok")

    step3_progress = start_progress("Step 3: Build full inverse metric coefficients", 74)
    step3 = build_inverse_metric_coeffs(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build full inverse metric coefficient arrays ... ok")

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
    setprecision(BigFloat, 100)
    subdivide_and_bound_rigorous_local(U0_PATH, 20, 8);
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
