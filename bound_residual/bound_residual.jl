using IntervalArithmetic

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "util", "problem.jl"))

function compute_bound_residual(coeffs_path::AbstractString; pdeg::Integer=240)
    println("Step 1: Load u0 and normalize u0")
    step1 = step1_load_and_normalize_u0(coeffs_path)
    println("Step 1: Load and normalize u0 ... ok")
    println("Step 1 guaranteed: $(all(isguaranteed, step1.normalized_coeffs))")

    step2_progress = start_progress("Step 2: Compute G and H values on the grid", pdeg + 2)
    step2 = step2_compute_GH_values_grid(step1.normalized_coeffs; pdeg, progress = step2_progress,)
    finish_progress!(step2_progress)
    println("Step 2 guaranteed: $(all(isguaranteed, step2.G) && all(isguaranteed, step2.H))")

    step3_progress = start_progress("Step 3: Compute MA derivative bounds", 12)
    step3 = step3_compute_MA_derivatives(step2.H, step2.G; pdeg, progress = step3_progress,)
    finish_progress!(step3_progress)
    println(
        "Step 3: ||MAx||_∞ <= [$(inf(step3.MAx_bound)), $(sup(step3.MAx_bound))], " *
        "||MAy||_∞ <= [$(inf(step3.MAy_bound)), $(sup(step3.MAy_bound))]"
    )
    println("Step 3 guaranteed: $(isguaranteed(step3.MAx_bound))")

    step4 = step4_compute_MA_derivative_bound(step3.MAx_bound, step3.MAy_bound)
    println("Step 4: ||dMA||_∞ <= [$(inf(step4.dMA_bound)), $(sup(step4.dMA_bound))]")
    println("Step 4 guaranteed: $(isguaranteed(step4.dMA_bound))")

    step5 = step5_compute_residual_bound_by_MVT(step3.H_coeffs, step3.G_coeffs, step4.dMA_bound)
    println("Step 5: ||E||_∞ <= $(step5.residual_bound)")
    println("Step 5 guaranteed: $(isguaranteed(step5.residual_bound))")

    step_nabla()
    step_nabla2()
    step_nabla3()

    return (; step1, step2, step3, step4, step5)
end

"""
Run the default residual-bound computation from the command line.
"""
function main()
    setprecision(BigFloat, 200)
    IntervalArithmetic.configure(; matmul = :slow)
    compute_bound_residual(U0_PATH; pdeg = 240)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
