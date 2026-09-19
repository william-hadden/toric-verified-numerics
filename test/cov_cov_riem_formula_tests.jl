using Test
using IntervalArithmetic
using Logging

global_logger(NullLogger())

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "curvature_bounds.jl"))

function point_derivative_enclosures(u11, u12, u22)
    components = Dict((1, 1) => u11, (1, 2) => u12, (2, 1) => u12, (2, 2) => u22)
    deriv_enc = Dict{Tuple{Int,Int,Tuple{Int,Int}},Any}()

    for i in 1:2, j in 1:2, total in 0:4, a1 in 0:total
        multiindex = (a1, total - a1)
        value = get(components[(i, j)], multiindex, 0 // 1)
        coeffs = reshape([interval(BigFloat, value)], 1, 1)
        deriv_enc[(i, j, multiindex)] = (; coeffs, tail = interval(BigFloat, 0), pdeg = 1)
    end

    return deriv_enc
end

function check_norm_squared(name, u11, u12, u22, expected)
    @testset "$name" begin
        deriv_enc = point_derivative_enclosures(u11, u12, u22)
        numerator = compute_cov_cov_riem_norm_squared_numerator_coeff_space(deriv_enc; pdeg = 1)
        result = only(numerator)
        expected_interval = interval(BigFloat, expected)

        @test all(isguaranteed, numerator)
        @test inf(result) <= inf(expected_interval)
        @test sup(expected_interval) <= sup(result)
        @test sup(abs(result - expected_interval)) < big"1e-60"
    end
end

function check_full_quotient_example()
    @testset "mixed cubic through quotient numerator pack" begin
        A11 = fill(interval(BigFloat, 0), 3, 3)
        A12 = fill(interval(BigFloat, 0), 3, 3)
        A22 = fill(interval(BigFloat, 0), 3, 3)
        D = fill(interval(BigFloat, 0), 3, 3)

        A11[1, 1], A11[2, 1] = interval(BigFloat, 3), interval(BigFloat, 1)
        A12[1, 2] = interval(BigFloat, -1)
        A22[1, 1] = interval(BigFloat, 2)
        D[1, 1] = interval(BigFloat, 11 // 2)
        D[2, 1], D[1, 3] = interval(BigFloat, 2), interval(BigFloat, -1 // 2)

        result = compute_inverse_derivative_numerator_components_truncated_coeff_space(
            (; A11, A12, A22, D); k = 4, pdeg = 3,
        )
        zero_interval = interval(BigFloat, 0)
        numerator = eval_cheb_2d_interval(result.cov_cov_riem_norm_squared_num, zero_interval, zero_interval)
        denominator = eval_cheb_2d_interval(D, zero_interval, zero_interval)
        norm_squared = numerator / denominator^12
        expected = interval(BigFloat, 107 // 2187)

        @test all(isguaranteed, result.cov_cov_riem_norm_squared_num)
        @test inf(norm_squared) <= inf(expected)
        @test sup(expected) <= sup(norm_squared)
    end
end

setprecision(BigFloat, 256) do
    flat_11 = Dict((0, 0) => 12 // 23)
    flat_12 = Dict((0, 0) => -2 // 23)
    flat_22 = Dict((0, 0) => 8 // 23)
    check_norm_squared("flat non-diagonal metric", flat_11, flat_12, flat_22, 0)

    sphere_11 = Dict((0, 0) => 8 // 9, (1, 0) => -2 // 3, (2, 0) => -2)
    sphere_22 = Dict((0, 0) => 15 // 16, (0, 1) => -1 // 2, (0, 2) => -2)
    check_norm_squared("S2 x S2", sphere_11, Dict(), sphere_22, 0)

    cubic_11 = Dict((0, 0) => 2, (1, 0) => 3, (2, 0) => 6, (3, 0) => 6)
    check_norm_squared("cubic product", cubic_11, Dict(), Dict((0, 0) => 1), 162)

    affine_cubic_11 = Dict(
        (0, 0) => 3, (1, 0) => 3, (0, 1) => 3,
        (2, 0) => 6, (1, 1) => 6, (0, 2) => 6,
        (3, 0) => 6, (2, 1) => 6, (1, 2) => 6, (0, 3) => 6,
    )
    check_norm_squared(
        "affine cubic with non-diagonal inverse metric",
        affine_cubic_11, Dict((0, 0) => -1), Dict((0, 0) => 1), 162,
    )

    quartic_11 = Dict(
        (0, 0) => 2, (1, 0) => 4, (2, 0) => 12,
        (3, 0) => 24, (4, 0) => 24,
    )
    check_norm_squared("quartic product", quartic_11, Dict(), Dict((0, 0) => 1), 11520)

    mixed_identity_11 = Dict(
        (0, 0) => 1, (0, 2) => 2, (1, 2) => -2,
        (0, 4) => 24, (2, 2) => 4,
    )
    mixed_identity_12 = Dict(
        (0, 1) => -1, (1, 1) => 1, (0, 3) => -6,
        (2, 1) => -2, (1, 3) => 12, (3, 1) => 6,
    )
    mixed_identity_22 = Dict(
        (0, 0) => 1, (1, 0) => -1, (0, 2) => 2,
        (2, 0) => 2, (1, 2) => -4, (3, 0) => -6,
        (0, 4) => 24, (2, 2) => 12, (4, 0) => 24,
    )
    check_norm_squared(
        "mixed cubic with identity Hessian",
        mixed_identity_11, mixed_identity_12, mixed_identity_22, 5136,
    )

    mixed_unequal_11 = Dict(
        (0, 0) => 1 // 2, (0, 2) => 1 // 6, (1, 2) => -1 // 18,
        (0, 4) => 1 // 3, (2, 2) => 1 // 27,
    )
    mixed_unequal_12 = Dict(
        (0, 1) => -1 // 6, (1, 1) => 1 // 18, (0, 3) => -1 // 6,
        (2, 1) => -1 // 27, (1, 3) => 1 // 9, (3, 1) => 1 // 27,
    )
    mixed_unequal_22 = Dict(
        (0, 0) => 1 // 3, (1, 0) => -1 // 9, (0, 2) => 1 // 9,
        (2, 0) => 2 // 27, (1, 2) => -2 // 27, (3, 0) => -2 // 27,
        (0, 4) => 2 // 9, (2, 2) => 2 // 27, (4, 0) => 8 // 81,
    )
    check_norm_squared(
        "mixed cubic with unequal Hessian",
        mixed_unequal_11, mixed_unequal_12, mixed_unequal_22, 107 // 2187,
    )

    check_full_quotient_example()
end
