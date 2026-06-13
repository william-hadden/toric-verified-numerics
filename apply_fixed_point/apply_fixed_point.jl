using IntervalArithmetic

include(joinpath(@__DIR__, "..", "bound_residual", "util", "io.jl"))

"""
Return the placeholder spectral gap for the Lichnerowicz operator.

We use the temporary lower bound `mu = 6` for the D6xT^2-invariant
Laplace eigenvalue. Since
`L = -1/2 Delta + e^F`, the non-constant modes are bounded away from zero by
`mu/2 - ||e^F||_C0`. The constant mode is bounded by `e^F`.
"""
function get_spectral_gap(; path::AbstractString = VERIFIED_BOUNDS_PATH)
    placeholder_laplace_spectral_gap = interval(BigFloat(6)) # <---- this is a placeholder!!
    MA_residual_C0_bound = read_bound("ma_residual_bounds"; path)["C0"] # C^0-bound for |e^F - 1|

    eF_C0_bound = interval(BigFloat(1)) + MA_residual_C0_bound
    constant_mode_spectral_gap = interval(BigFloat(1)) - MA_residual_C0_bound
    nonconstant_mode_spectral_gap =
        placeholder_laplace_spectral_gap / exact(2) - eF_C0_bound

    if !(inf(constant_mode_spectral_gap) > 0)
        error("The MA residual is too large to certify a positive constant-mode spectral gap.")
    end
    if !(inf(nonconstant_mode_spectral_gap - constant_mode_spectral_gap) > 0)
        error("The Laplace spectral gap gives a spectral gap smaller than the constant-mode spectral gap.")
    end

    return constant_mode_spectral_gap
end

function get_injectivity_constant(; path::AbstractString = VERIFIED_BOUNDS_PATH)
    curvature_bounds = read_bound("curvature_bounds"; path)
    K1 = curvature_bounds["ricci_C0"]
    K2 = curvature_bounds["riemann_C0"]
    K3 = curvature_bounds["nabla_riemann_C0"]
    K4 = curvature_bounds["nabla2_riemann_C0"]
    n = interval(BigFloat(4))

    lambda_1 = get_spectral_gap(; path)

    Cl_D3u_estimate_second_summand =
        interval(BigFloat(2)) * n * K2 + n * K3
    Cl_D4u_estimate_second_summand =
        interval(BigFloat(2)) * (interval(BigFloat(2)) * n * K2 + n * K3) +
        interval(BigFloat(4)) * n * K2 +
        interval(BigFloat(2)) * n * K3
    Cl_D5u_estimate_second_summand =
        interval(BigFloat(6)) * K2 * n +
        interval(BigFloat(3)) * K3 * n +
        interval(BigFloat(60)) * K2^2 * n^2 +
        interval(BigFloat(84)) * K2 * K3 * n^2 +
        interval(BigFloat(39)) * K3^2 * n^2 +
        interval(BigFloat(12)) * K2 * K4 * n^2 +
        interval(BigFloat(18)) * K3 * K4 * n^2 +
        interval(BigFloat(3)) * K4^2 * n^2

    Cl_L2_1_estimate_Laplace_term = interval(BigFloat(1)) / sqrt(interval(BigFloat(2)))
    Cl_L2_1_estimate_u_term =
        interval(BigFloat(1)) + interval(BigFloat(1)) / sqrt(interval(BigFloat(2)))

    Cl_L2_2_estimate_Laplace_term =
        interval(BigFloat(1)) +
        Cl_L2_1_estimate_Laplace_term * (interval(BigFloat(1)) + sqrt(K1))
    Cl_L2_2_estimate_u_term =
        interval(BigFloat(1)) +
        Cl_L2_1_estimate_u_term * (interval(BigFloat(1)) + sqrt(K1))

    Cl_L2_3_estimate_Laplace_term =
        interval(BigFloat(1)) +
        (interval(BigFloat(2)) + sqrt(Cl_D3u_estimate_second_summand)) *
        Cl_L2_2_estimate_Laplace_term
    Cl_L2_3_estimate_u_term =
        (interval(BigFloat(2)) + sqrt(Cl_D3u_estimate_second_summand)) *
        Cl_L2_2_estimate_u_term

    Cl_L2_4_estimate_Laplace_term =
        sqrt(interval(BigFloat(2))) +
        (interval(BigFloat(2)) + sqrt(Cl_D4u_estimate_second_summand)) *
        Cl_L2_3_estimate_Laplace_term
    Cl_L2_4_estimate_u_term =
        (interval(BigFloat(2)) + sqrt(Cl_D4u_estimate_second_summand)) *
        Cl_L2_3_estimate_u_term

    Cl_L2_5_estimate_Laplace_term =
        sqrt(interval(BigFloat(3))) +
        (interval(BigFloat(2)) + sqrt(Cl_D5u_estimate_second_summand)) *
        Cl_L2_4_estimate_Laplace_term
    Cl_L2_5_estimate_u_term =
        (interval(BigFloat(2)) + sqrt(Cl_D5u_estimate_second_summand)) *
        Cl_L2_4_estimate_u_term

    Cr_const_injectivity_estimate =
        Cl_L2_5_estimate_Laplace_term +
        Cl_L2_5_estimate_u_term / lambda_1

    write_bound_entry("fixed_point_bounds", "injectivity_constant", Cr_const_injectivity_estimate; path)
    return nothing
end
