using IntervalArithmetic
using Logging

global_logger(NullLogger())

include(joinpath(@__DIR__, "..", "bound_residual", "util", "io.jl"))
include(joinpath(@__DIR__, "..", "bound_residual", "util", "chebyshev_algebra.jl"))
include(joinpath(@__DIR__, "..", "bound_residual", "util", "problem.jl"))
include(joinpath(@__DIR__, "util", "io.jl"))
include(joinpath(@__DIR__, "util", "progress.jl"))
include(joinpath(@__DIR__, "util", "interval_helpers.jl"))
include(joinpath(@__DIR__, "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "util", "inverse_bounds.jl"))
include(joinpath(@__DIR__, "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "util", "chebyshev_interval_evaluation.jl"))
include(joinpath(@__DIR__, "util", "inverse_subdivision_bounds.jl"))
include(joinpath(@__DIR__, "util", "inverse_workflows.jl"))

"""
Bound the inverse metric `u^{ij}` using interval arithmetic and coefficient
sup bounds.
"""
function compute_bound_inverse(coeffs_path::AbstractString)
    step1_progress = start_progress("Step 1: Load u0", 1)
    step1 = load_metric_coeffs_csv(coeffs_path)
    finish_progress!(step1_progress)
    println("Step 1: Load u0 ... ok")

    step2_progress = start_progress("Step 2: Compute second derivatives", 1)
    step2 = compute_second_derivatives(step1)
    finish_progress!(step2_progress)
    println("Step 2: Compute second derivatives ... ok")

    step3_progress = start_progress("Step 3: Chebyshev products", 74)
    step3 = build_inverse_metric(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build inverse metric coefficient arrays ... ok")

    step4_progress = start_progress("Step 4: Coefficient bounds", 1)
    step4 = compute_inverse_bound_by_coefficient_bounds(step3)
    finish_progress!(step4_progress)
    println("Step 4: Bound inverse metric entries by coefficient sums ... ok")
    print_inverse_bound_summary(step4)

    return (; step1, step2, step3, step4)
end

"""
Bound the inverse metric `u^{ij}` using the real-space inverse metric builder.
"""
function compute_bound_inverse_real(coeffs_path::AbstractString)
    step1_progress = start_progress("Step 1: Load u0", 1)
    step1 = load_metric_coeffs_csv(coeffs_path)
    finish_progress!(step1_progress)
    println("Step 1: Load u0 ... ok")

    step2_progress = start_progress("Step 2: Compute second derivatives", 1)
    step2 = compute_second_derivatives(step1)
    finish_progress!(step2_progress)
    println("Step 2: Compute second derivatives ... ok")

    step3_progress = start_progress("Step 3: Real-space products", 28)
    step3 = build_inverse_metric(step2; method = :real_space, progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build inverse metric coefficient arrays in real space ... ok")

    step4_progress = start_progress("Step 4: Coefficient bounds", 1)
    step4 = compute_inverse_bound_by_coefficient_bounds(step3)
    finish_progress!(step4_progress)
    println("Step 4: Bound inverse metric entries by coefficient sums ... ok")
    print_inverse_bound_summary(step4)

    return (; step1, step2, step3, step4)
end

function main()
    setprecision(BigFloat, 100)
    compute_bound_inverse(METRIC_U0_PATH)
    # compute_bound_inverse_real(METRIC_U0_PATH)
    lower_degree_approximation(METRIC_U0_PATH,10)

    # pdeg > 0 means truncate
    # subdivide_and_bound_rigorous2(METRIC_U0_PATH, 10, 8)

    # pdeg <= 0 means use full coefficients
    # subdivide_and_bound_rigorous(METRIC_U0_PATH, 0, 8)

    # try to avoid interval blow up
    subdivide_and_bound_rigorous_local(METRIC_U0_PATH, 20, 8);
    # subdivide_and_bound_rigorous_local(METRIC_U0_PATH, 60, 8)

end
main()
# if abspath(PROGRAM_FILE) == @__FILE__
#     main()
# end
