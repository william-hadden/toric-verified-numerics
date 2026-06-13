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

"""
Return the injectivity estimate for the linearised operator (Lichnerovicz operator)
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

    # From proposition:estimates-from-commutator-formula.
    D3u_estimate_second_summand =
        interval(BigFloat(2)) * n * K2 + n * K3
    D4u_estimate_second_summand =
        interval(BigFloat(2)) * (interval(BigFloat(2)) * n * K2 + n * K3) +
        interval(BigFloat(4)) * n * K2 +
        interval(BigFloat(2)) * n * K3
    D5u_estimate_second_summand =
        interval(BigFloat(6)) * K2 * n +
        interval(BigFloat(3)) * K3 * n +
        interval(BigFloat(60)) * K2^2 * n^2 +
        interval(BigFloat(84)) * K2 * K3 * n^2 +
        interval(BigFloat(39)) * K3^2 * n^2 +
        interval(BigFloat(12)) * K2 * K4 * n^2 +
        interval(BigFloat(18)) * K3 * K4 * n^2 +
        interval(BigFloat(3)) * K4^2 * n^2

    # From corollary:a-priori-estimate.
    L2_1_estimate_Laplace_term = interval(BigFloat(1)) / sqrt(interval(BigFloat(2)))
    L2_1_estimate_u_term =
        interval(BigFloat(1)) + interval(BigFloat(1)) / sqrt(interval(BigFloat(2)))

    L2_2_estimate_Laplace_term =
        interval(BigFloat(1)) +
        L2_1_estimate_Laplace_term * (interval(BigFloat(1)) + sqrt(K1))
    L2_2_estimate_u_term =
        interval(BigFloat(1)) +
        L2_1_estimate_u_term * (interval(BigFloat(1)) + sqrt(K1))

    L2_3_estimate_Laplace_term =
        interval(BigFloat(1)) +
        (interval(BigFloat(2)) + sqrt(D3u_estimate_second_summand)) *
        L2_2_estimate_Laplace_term
    L2_3_estimate_u_term =
        (interval(BigFloat(2)) + sqrt(D3u_estimate_second_summand)) *
        L2_2_estimate_u_term

    L2_4_estimate_Laplace_term =
        sqrt(interval(BigFloat(2))) +
        (interval(BigFloat(2)) + sqrt(D4u_estimate_second_summand)) *
        L2_3_estimate_Laplace_term
    L2_4_estimate_u_term =
        (interval(BigFloat(2)) + sqrt(D4u_estimate_second_summand)) *
        L2_3_estimate_u_term

    L2_5_estimate_Laplace_term =
        sqrt(interval(BigFloat(3))) +
        (interval(BigFloat(2)) + sqrt(D5u_estimate_second_summand)) *
        L2_4_estimate_Laplace_term
    L2_5_estimate_u_term =
        (interval(BigFloat(2)) + sqrt(D5u_estimate_second_summand)) *
        L2_4_estimate_u_term

    # From corollary:inj-estimate.
    const_injectivity_estimate =
        L2_5_estimate_Laplace_term +
        L2_5_estimate_u_term / lambda_1

    return const_injectivity_estimate
end

"""
Return the higher order erstimate constant as a function of r.
This is called alpha_3 in theorem:fixed-point.
"""
function get_higher_order_constant(r::Float64)
    r = interval(BigFloat(r))

    V = interval(BigFloat(12)) * interval(BigFloat, pi)^2

    # Ricci lower bound
    mu = interval(BigFloat(1)) # <-- Placeholder!!

    # Euclidean unit sphere volumes used in the geometric-analysis constants.
    omega_3 = interval(BigFloat(2)) * interval(BigFloat, pi)^2
    omega_4 = interval(BigFloat(8)) * interval(BigFloat, pi)^2 / interval(BigFloat(3))

    # From beginning of subsection: Inequalities from geometric analysis.
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

    # From corollary:explicit-embeddings and proposition:L4-L3-embedding.
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

    # From corollary:sobolev-multiplication-thm.
    const_multiplication_theorem = max(
        max(
            interval(BigFloat(2)) *
            const_emb_C4 *
            const_emb_C3 *
            const_emb_C2 *
            const_emb_C1,
            const_emb_C4 *
            const_emb_C3 *
            const_emb_C2 *
            const_emb_C1 +
            interval(BigFloat(6)) * const_emb_C1^2,
        ),
        interval(BigFloat(8)) * const_emb_C1^2,
    )

    # From proposition:non-linear-estimate, using ||u||_{L2_3} <= ||u||_{L2_5} <= r.
    return const_multiplication_theorem *
        (interval(BigFloat(1)) + exp(interval(BigFloat(2)) * const_multiplication_theorem * r))
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
    r_float = nextfloat(Float64(sup(interval(BigFloat(2)) * alpha_2_alpha_1)))
    r = interval(BigFloat(r_float))
    alpha_3 = get_higher_order_constant(r_float)

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
