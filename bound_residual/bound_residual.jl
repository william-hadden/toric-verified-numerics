using IntervalArithmetic

include(joinpath(@__DIR__, "util", "io.jl"))
include(joinpath(@__DIR__, "util", "chebyshev_algebra.jl"))
include(joinpath(@__DIR__, "util", "problem.jl"))

"""
Run the coefficient-space five-step bound computation for `E = exp(F) - 1`.
"""
function compute_bound_residual(coeffs_path::AbstractString; p::Integer = DEFAULT_TAYLOR_DEGREE, ref_index::Tuple{Int, Int} = DEFAULT_REF_INDEX)
    step1 = step1_load_and_normalize_u0(coeffs_path; ref_index)
    println("Step 1: Load and normalize u0 ... ok")

    step2 = step2_coarse_sup_bounds(step1.pack)
    #println("Step 2: ||H||_∞ <= $(step2.H_bound), ||G||_∞ <= $(step2.G_bound)")
    println("Step 2: ||H||_∞ <= [$(inf(step2.H_bound)), $(sup(step2.H_bound))], " * "||G||_∞ <= [$(inf(step2.G_bound)), $(sup(step2.G_bound))]")

    step3 = step3_taylor_tail_bound(step2.H_bound, step2.G_bound, p)
    println("Step 3: ||G * (exp(H) - exp_p(H))||_∞ <= $(step3.total_bound)")

    step4 = step4_poly_bound(step2.H_coeffs, step2.G_coeffs, p)
    println("Step 4: ||G * exp_p(H) - 1||_∞ <= $(step4.poly_bound)")

    total_bound = step3.total_bound + step4.poly_bound
    println("Step 5: ||E||_∞ <= $(total_bound)")

    return (;
        p,
        step1,
        step2,
        step3,
        step4,
        total_bound,
    )
end

"""
Run the default residual-bound computation from the command line.
"""
function main()
    p = DEFAULT_TAYLOR_DEGREE
    ref_index = DEFAULT_REF_INDEX
    compute_bound_residual(U0_PATH; p, ref_index)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

# exact(k) rather than true(k) 
