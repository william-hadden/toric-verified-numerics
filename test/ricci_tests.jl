@testset "Ricci assembly for polynomial inverse metric with D = 1" begin
    # Mathematical purpose:
    # For u^{11}=x^2, u^{12}=xy, and u^{22}=y^2, the implemented convention
    # R^i_j = -1/2 sum_k ∂_{k j} u^{k i} gives
    # R^1_1 = -3/2, R^1_2 = 0, R^2_1 = 0, R^2_2 = -3/2.
    # Because D = 1, the quotient denominator contributes no additional terms.
    coords = interval_coordinate_series()
    x2 = cheb_mul_fast(coords.x, coords.x)
    y2 = cheb_mul_fast(coords.y, coords.y)
    xy = cheb_mul_fast(coords.x, coords.y)
    one_series = interval_one_series()

    inverse_coeffs = (; A11 = x2, A12 = xy, A21 = xy, A22 = y2, D = one_series)
    inverse_bounds = compute_inverse_bound_by_coefficient_bounds(inverse_coeffs)
    ricci = compute_ricci_bound_from_inverse_coeffs(inverse_coeffs, inverse_bounds)

    assert_constant_series_equals(ricci.R11_num, -3 // 2)
    assert_constant_series_equals(ricci.R12_num, 0)
    assert_constant_series_equals(ricci.R21_num, 0)
    assert_constant_series_equals(ricci.R22_num, -3 // 2)

    @test issubset_interval(interval(BigFloat(3) / exact(2)), ricci.R11_bound)
    @test issubset_interval(interval(BigFloat(0)), ricci.R12_bound)
    @test issubset_interval(interval(BigFloat(0)), ricci.R21_bound)
    @test issubset_interval(interval(BigFloat(3) / exact(2)), ricci.R22_bound)
    @test sup(ricci.R11_bound) == BigFloat(3) / exact(2)
    @test sup(ricci.R12_bound) == 0
    @test sup(ricci.R21_bound) == 0
    @test sup(ricci.R22_bound) == BigFloat(3) / exact(2)

    @test issubset_interval(interval(BigFloat(3)), ricci.ricci_norm_bound)
    @test sup(ricci.ricci_norm_bound) == 3
    @test sup(ricci.ricci_norm_squared_bound) == 9

    truncated_ricci = compute_ricci_numerators_from_inverse_coeffs(
        inverse_coeffs;
        method = :truncated_coefficient_space,
        pdeg = 1,
    )
    assert_constant_series_equals(truncated_ricci.R11_num, -3 // 2)
    assert_constant_series_equals(truncated_ricci.R12_num, 0)
    assert_constant_series_equals(truncated_ricci.R21_num, 0)
    assert_constant_series_equals(truncated_ricci.R22_num, -3 // 2)
end
