using IntervalArithmetic

include(joinpath(@__DIR__, "util", "io.jl"))
include(joinpath(@__DIR__, "util", "chebyshev_algebra.jl"))
include(joinpath(@__DIR__, "util", "problem.jl"))

"""
Run the coefficient-space five-step bound computation for `E = exp(F) - 1`.
"""
function compute_bound_residual(coeffs_path::AbstractString; ref_index::Tuple{Int, Int} = DEFAULT_REF_INDEX)
    step1 = step1_load_and_normalize_u0(coeffs_path; ref_index)
    println("Step 1: Load and normalize u0 ... ok")

    step2 = step2_coarse_sup_bounds(step1.pack)
    println("Step 2: ||H||_∞ <= [$(inf(step2.H_bound)), $(sup(step2.H_bound))], " * "||G||_∞ <= [$(inf(step2.G_bound)), $(sup(step2.G_bound))]")

    step3 = step3_compute_MA_derivative_bound(step2.H_coeffs, step2.G_coeffs, step2.H_bound, step1.pack)
    println("Step 3: ||MAx||_∞ <= [$(inf(step3.MAx_bound)), $(sup(step3.MAx_bound))], " * "||MAy||_∞ <= [$(inf(step3.MAy_bound)), $(sup(step3.MAy_bound))]" * "||dMA||_∞ <= [$(inf(step3.dMA_bound)), $(sup(step3.dMA_bound))]")

    step4 = step4_compute_residual_bound(step2.H_coeffs, step2.G_coeffs, step3.dMA_bound)
    println("Step 4: ||E||_∞ <= $(step4.residual_bound)")

    return (;
        step1,
        step2,
        step3,
        step4,
    )
end

"""
Run the default residual-bound computation from the command line.
"""
function main()
    setprecision(BigFloat, 100)
    ref_index = DEFAULT_REF_INDEX
    compute_bound_residual(U0_PATH; ref_index)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# exact(k) rather than true(k) 
