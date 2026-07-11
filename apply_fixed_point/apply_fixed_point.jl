using IntervalArithmetic

include(joinpath(@__DIR__, "..", "bound_residual", "util", "io.jl"))

"""
Return the Sobolev embedding and multiplication constants used in the fixed-point
estimates.

The names mirror the corresponding constants in the manuscript.
"""
function get_sobolev_multiplication_constants(
    ; path::AbstractString = VERIFIED_BOUNDS_PATH,
)
    V = interval(BigFloat(12)) * interval(BigFloat, pi)^2

    curvature_bounds = read_bound("curvature_bounds"; path)
    mu = curvature_bounds["ricci_lower_bound"]

    # Volumes of the unit-radius round spheres S^3 and S^4.
    omega_3 = interval(BigFloat(2)) * interval(BigFloat, pi)^2
    omega_4 = interval(BigFloat(8)) * interval(BigFloat, pi)^2 / interval(BigFloat(3))

    K_4_2 =
        interval(BigFloat(1)) /
        (interval(BigFloat(2)) * sqrt(interval(BigFloat(2)))) *
        sqrt(sqrt(interval(BigFloat(12)) / omega_3))
    said_isoperimetric_const =
        interval(BigFloat(4)) * sqrt(sqrt(omega_3 / interval(BigFloat(4))))
    said_isoperimetric_const_of_M =
        sqrt(sqrt(V)) /
        sqrt(sqrt(interval(BigFloat(2)))) *
        sqrt(mu / interval(BigFloat(3))) /
        (interval(BigFloat(2)) / interval(BigFloat(3)))

    const_emb_C1 = max(
        K_4_2 * sqrt(sqrt(omega_4 / (V * mu^2))),
        interval(BigFloat(1)) / sqrt(sqrt(V)),
    )
    const_emb_C2 = sqrt(sqrt(cbrt(V)))
    const_emb_C3 = max(
        K_4_2 * sqrt(sqrt(omega_4 / (V * mu^2))) * interval(BigFloat(3)),
        interval(BigFloat(1)) / sqrt(sqrt(V)),
    )
    const_emb_C4 = max(
        (interval(BigFloat(11)) / interval(BigFloat(8))) /
        sqrt(sqrt(cbrt(interval(BigFloat(11)) / interval(BigFloat(8))))) *
        (said_isoperimetric_const / said_isoperimetric_const_of_M) *
        sqrt(cbrt(interval(BigFloat(4)) * V)) /
        sqrt(sqrt(omega_3)),
        interval(BigFloat(2)) / sqrt(sqrt(cbrt(V))),
    )

    embedding_product = const_emb_C4 * const_emb_C3 * const_emb_C2 * const_emb_C1
    const_sobolev_low_order_multiplication_M0 =
        interval(BigFloat(2)) * embedding_product
    const_sobolev_low_order_multiplication_M1 =
        const_sobolev_low_order_multiplication_M0 + const_emb_C1^2
    const_sobolev_low_order_multiplication_M2 =
        const_sobolev_low_order_multiplication_M0 +
        interval(BigFloat(4)) * const_emb_C1^2

    const_multiplication_theorem = max(
        max(
            interval(BigFloat(4)) * embedding_product,
            interval(BigFloat(2)) * embedding_product +
            interval(BigFloat(3)) * const_emb_C1^2,
        ),
        interval(BigFloat(8)) * const_emb_C1^2,
    )

    return (;
        const_emb_C1,
        const_emb_C2,
        const_emb_C3,
        const_emb_C4,
        const_sobolev_low_order_multiplication_M0,
        const_sobolev_low_order_multiplication_M1,
        const_sobolev_low_order_multiplication_M2,
        const_multiplication_theorem,
    )
end

"""
Return a lower bound for the absolute spectral gap of the Lichnerowicz operator.

For `L_0 = -1/2 Delta + 1`, the constant mode has eigenvalue one and the
nonconstant D6xT^2-invariant modes have absolute eigenvalue at least
`lambda_Delta/2 - 1`. Multiplication by `E = e^F - 1` is a self-adjoint
perturbation of L^2-operator norm at most `||E||_C0`.
"""
function get_spectral_gap(; path::AbstractString = VERIFIED_BOUNDS_PATH)
    one = interval(BigFloat(1))
    laplace_spectral_gap_lower_bound = read_bound("lambda_1_lower_bound"; path)
    epsilon_C0 = read_bound("ma_residual_bounds"; path)["C0"]

    nonconstant_L0_spectral_gap =
        laplace_spectral_gap_lower_bound / interval(BigFloat(2)) - one
    if !(inf(nonconstant_L0_spectral_gap) > 0)
        error("The verified Laplace spectral gap is not large enough to separate the nonconstant Lichnerowicz spectrum from zero.")
    end

    L0_spectral_gap = min(one, nonconstant_L0_spectral_gap)
    L_spectral_gap = L0_spectral_gap - epsilon_C0
    if !(inf(L_spectral_gap) > 0)
        error("The MA residual is too large to certify a positive Lichnerowicz spectral gap.")
    end

    return L_spectral_gap
end

"""
Return the injectivity estimate for the linearised operator (Lichnerowicz operator)
as a map from L^2_5 to L^2_3. This is called alpha_2 in theorem:fixed-point.
"""
function get_injectivity_constant(; path::AbstractString = VERIFIED_BOUNDS_PATH)
    # Inputs from verified_bounds.json.
    curvature_bounds = read_bound("curvature_bounds"; path)
    K1 = curvature_bounds["ricci_C0"]
    K2 = curvature_bounds["riemann_C0"]
    K3 = curvature_bounds["nabla_riemann_C0"]
    K4 = curvature_bounds["nabla2_riemann_C0"]
    n = interval(BigFloat(4))

    lambda_1 = get_spectral_gap(; path)
    epsilon = get_residual_bound(; path)
    sobolev_constants = get_sobolev_multiplication_constants(; path)

    one = interval(BigFloat(1))
    two = interval(BigFloat(2))
    three = interval(BigFloat(3))
    four = interval(BigFloat(4))
    five = interval(BigFloat(5))
    seven = interval(BigFloat(7))

    # From proposition:estimates-from-commutator-formula.
    D3u_estimate_second_summand =
        two * K1^2 + three * n * K2 + n * K3
    D4u_estimate_second_summand =
        three * (K1 + n * K3)^2 +
        three * (three * n * K2 + n * K3)^2 +
        five * n * K2 +
        two * n * K3
    D5u_estimate_second_summand =
        four * (K1 + two * n * K3 + n * K4)^2 +
        four * (n * (n + three) * K3 + n * (n + two) * K2 + n * K4)^2 +
        four * (five * n * K2 + two * n * K3)^2 +
        seven * n * K2 +
        three * n * K3

    one_plus_M0_epsilon =
        one + sobolev_constants.const_sobolev_low_order_multiplication_M0 * epsilon
    one_plus_M1_epsilon =
        one + sobolev_constants.const_sobolev_low_order_multiplication_M1 * epsilon
    one_plus_M2_epsilon =
        one + sobolev_constants.const_sobolev_low_order_multiplication_M2 * epsilon
    one_plus_const_multiplication_theorem_epsilon =
        one + sobolev_constants.const_multiplication_theorem * epsilon

    # From corollary:a-priori-estimate.
    L2_1_estimate_Laplace_term = sqrt(two)
    L2_1_estimate_u_term =
        one + one / sqrt(two) + sqrt(two) * one_plus_M0_epsilon

    L2_2_estimate_Laplace_term =
        two + (one + sqrt(K1)) * L2_1_estimate_Laplace_term
    L2_2_estimate_u_term =
        two * one_plus_M0_epsilon + (one + sqrt(K1)) * L2_1_estimate_u_term

    L2_3_estimate_L2_2_term =
        one +
        sqrt(D3u_estimate_second_summand) +
        two * sqrt(two) * one_plus_M1_epsilon

    L2_3_estimate_Laplace_term =
        two * sqrt(two) + L2_3_estimate_L2_2_term * L2_2_estimate_Laplace_term
    L2_3_estimate_u_term =
        L2_3_estimate_L2_2_term * L2_2_estimate_u_term

    L2_4_estimate_L2_3_term =
        one +
        sqrt(D4u_estimate_second_summand) +
        two * sqrt(three) * one_plus_M2_epsilon

    L2_4_estimate_Laplace_term =
        two * sqrt(three) + L2_4_estimate_L2_3_term * L2_3_estimate_Laplace_term
    L2_4_estimate_u_term =
        L2_4_estimate_L2_3_term * L2_3_estimate_u_term

    L2_5_estimate_L2_4_term =
        one +
        sqrt(D5u_estimate_second_summand) +
        four * one_plus_const_multiplication_theorem_epsilon

    L2_5_estimate_Laplace_term =
        four + L2_5_estimate_L2_4_term * L2_4_estimate_Laplace_term
    L2_5_estimate_u_term =
        L2_5_estimate_L2_4_term * L2_4_estimate_u_term

    # From corollary:inj-estimate.
    const_injectivity_estimate =
        L2_5_estimate_Laplace_term +
        L2_5_estimate_u_term / lambda_1

    return const_injectivity_estimate
end

"""
Return the higher order estimate constant as a function of r.
This is called alpha_3 in theorem:fixed-point.
"""
function get_higher_order_constant(
    r::Real;
    path::AbstractString = VERIFIED_BOUNDS_PATH,
)
    r = interval(BigFloat(r))
    epsilon = get_residual_bound(; path)
    const_multiplication_theorem =
        get_sobolev_multiplication_constants(; path).const_multiplication_theorem

    # From proposition:non-linear-estimate, using ||u||_{L2_3} <= ||u||_{L2_5} <= r.
    return const_multiplication_theorem *
        (
            interval(BigFloat(1)) +
            (interval(BigFloat(1)) + const_multiplication_theorem * epsilon) *
            exp(interval(BigFloat(2)) * const_multiplication_theorem * r)
        )
end

"""
Return the residual bound in L^2_3 with the convention
||f||_{L^2_3}=||f||_{L^2}+...+||nabla^3 f||_{L^2}.
This is called alpha_1 in theorem:fixed-point.
"""
function get_residual_bound(; path::AbstractString = VERIFIED_BOUNDS_PATH)
    ma_sobolev_bounds = read_bound("ma_sobolev_bounds"; path)
    return ma_sobolev_bounds["L2"] +
        ma_sobolev_bounds["nabla_L2"] +
        ma_sobolev_bounds["nabla2_L2"] +
        ma_sobolev_bounds["nabla3_L2"]
end

function check_fixed_point(; path::AbstractString = VERIFIED_BOUNDS_PATH)
    alpha_1 = get_residual_bound(; path)
    alpha_2 = get_injectivity_constant(; path)

    alpha_2_alpha_1 = alpha_2 * alpha_1
    r_upper = nextfloat(sup(interval(BigFloat(2)) * alpha_2_alpha_1))
    r = interval(r_upper)
    alpha_3 = get_higher_order_constant(r_upper; path)

    condition1_lhs = alpha_2_alpha_1 + alpha_2 * alpha_3 * r^2
    condition2_lhs = interval(BigFloat(2)) * alpha_2 * alpha_3 * r

    condition1_holds = sup(condition1_lhs) <= inf(r)
    condition2_holds = sup(condition2_lhs) < 1

    if condition1_holds && condition2_holds
        println(
            "For alpha1=$(alpha_1), alpha2=$(alpha_2), alpha3=$(alpha_3), r=$(r) " *
            "the conditions alpha2*alpha1 + alpha2*alpha3*r^2 <= r and " *
            "2*alpha2*alpha3*r < 1 are satisfied"
        )
    else
        println(
            "For alpha1=$(alpha_1), alpha2=$(alpha_2), alpha3=$(alpha_3), r=$(r) " *
            "the fixed-point conditions are not satisfied. " *
            "Condition 1 lhs=$(condition1_lhs), rhs=$(r), holds=$(condition1_holds). " *
            "Condition 2 lhs=$(condition2_lhs), rhs=$(interval(BigFloat(1))), holds=$(condition2_holds)."
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_fixed_point()
end
