using IntervalArithmetic

include(joinpath(@__DIR__, "util", "io.jl"))
include(joinpath(@__DIR__, "util", "chebyshev_algebra.jl"))
include(joinpath(@__DIR__, "util", "problem.jl"))

function compute_bound_residual_v2(coeffs_path::AbstractString; pdeg::Integer=240)
    step1 = step1_load_and_normalize_u0(coeffs_path)
    println("Step 1: Load and normalize u0 ... ok")
    println("Step 1 guaranteed: $(all(isguaranteed, step1.normalized_coeffs))")

    step2 = step2_compute_GH_values_grid(step1.normalized_coeffs; pdeg)
    println("Step 2: Compute G and H values on the grid ... ok")
    println("Step 2 guaranteed: $(all(isguaranteed, step2.G) && all(isguaranteed, step2.H))")

    step3 = step3_compute_MA_derivative_bound_v2(step2.H, step2.G)
    println("Step 3: ||MAx||_∞ <= [$(inf(step3.MAx_bound)), $(sup(step3.MAx_bound))], " * "||MAy||_∞ <= [$(inf(step3.MAy_bound)), $(sup(step3.MAy_bound))]" * "||dMA||_∞ <= [$(inf(step3.dMA_bound)), $(sup(step3.dMA_bound))]")
    println("Step 3 guaranteed: $(isguaranteed(step3.dMA_bound))")

    step4 = step4_compute_residual_bound_by_MVT(step3.H_coeffs, step3.G_coeffs, step3.dMA_bound)
    println("Step 4: ||E||_∞ <= $(step4.residual_bound)")
    println("Step 4 guaranteed: $(isguaranteed(step4.residual_bound))")

    step_nabla()
    step_nabla2()
    step_nabla3()

    return (; step1, step2, step3, step4)
end

"""
Run the default residual-bound computation from the command line.
"""
function main()
    setprecision(BigFloat, 100)
    compute_bound_residual_v2(U0_PATH; pdeg = 240)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# exact(k) rather than true(k) 
