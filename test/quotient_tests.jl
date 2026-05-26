@testset "quotient derivative numerator with D = 1" begin
    # Mathematical purpose:
    # When D is exactly one, A / D is just A and the D^3 numerator for
    # ∂_{ab}(A / D) must reduce exactly to the Chebyshev coefficient array
    # of ∂_{ab}A. This verifies the quotient-rule implementation without
    # any denominator effects.
    coords = interval_coordinate_series()
    x = coords.x
    y = coords.y
    x2 = cheb_mul_fast(x, x)
    y2 = cheb_mul_fast(y, y)
    xy = cheb_mul_fast(x, y)
    one_series = interval_one_series()

    for A in (x, y, x2, y2, xy)
        for (a, b) in ((1, 1), (2, 2), (1, 2), (2, 1))
            numerator = quotient_second_derivative_numerator(A, one_series, one_series, a, b)
            expected = partial_coeffs(partial_coeffs(A, a), b)
            assert_coeff_arrays_equal(numerator, expected)
        end
    end

    assert_constant_series_equals(quotient_second_derivative_numerator(x2, one_series, one_series, 1, 1), 2)
    assert_constant_series_equals(quotient_second_derivative_numerator(y2, one_series, one_series, 2, 2), 2)
    assert_constant_series_equals(quotient_second_derivative_numerator(xy, one_series, one_series, 1, 2), 1)
    assert_constant_series_equals(quotient_second_derivative_numerator(xy, one_series, one_series, 2, 1), 1)
end
