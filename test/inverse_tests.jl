@testset "small rigorous inverse and Ricci bounds" begin
    # Mathematical purpose:
    # This is a small end-to-end rigorous integration test. It uses a 3x3
    # truncation of the real coefficient data, writes it through the public CSV
    # path, and checks that both inverse and Ricci certificates remain positive,
    # guaranteed, and finite.
    path = write_small_metric_coeffs_csv()

    try
        inverse_result = compute_bound_inverse(path)
        @test inf(inverse_result.step4.D_lower) > 0
        @test all(isguaranteed, inverse_result.step3.A11)
        @test all(isguaranteed, inverse_result.step3.A12)
        @test all(isguaranteed, inverse_result.step3.A22)
        @test all(isguaranteed, inverse_result.step3.D)
        @test isguaranteed(inverse_result.step4.u11_bound)
        @test isguaranteed(inverse_result.step4.u12_bound)
        @test isguaranteed(inverse_result.step4.u22_bound)
        @test sup(inverse_result.step4.u11_bound) < 10
        @test sup(inverse_result.step4.u12_bound) < 10
        @test sup(inverse_result.step4.u22_bound) < 10

        ricci_result = bound_ricci(path)
        @test inf(ricci_result.step4.D_lower) > 0
        @test isguaranteed(ricci_result.step5.ricci_norm_bound)
        @test isguaranteed(ricci_result.step5.ricci_norm_squared_bound)
        @test sup(ricci_result.step5.ricci_norm_bound) < 100
        @test sup(ricci_result.step5.R11_bound) < 50
        @test sup(ricci_result.step5.R12_bound) < 50
        @test sup(ricci_result.step5.R21_bound) < 50
        @test sup(ricci_result.step5.R22_bound) < 50

        real_inverse_result = compute_bound_inverse_real(path)
        @test inf(real_inverse_result.step4.D_lower) > 0
        @test all(isguaranteed, real_inverse_result.step3.A11)
        @test all(isguaranteed, real_inverse_result.step3.A12)
        @test all(isguaranteed, real_inverse_result.step3.A22)
        @test all(isguaranteed, real_inverse_result.step3.D)
        @test sup(real_inverse_result.step4.u11_bound) <= 1.1 * sup(inverse_result.step4.u11_bound)
        @test sup(real_inverse_result.step4.u12_bound) <= 1.1 * sup(inverse_result.step4.u12_bound)
        @test sup(real_inverse_result.step4.u22_bound) <= 1.1 * sup(inverse_result.step4.u22_bound)
        @test inf(real_inverse_result.step4.D_lower) >= 0.9 * inf(inverse_result.step4.D_lower)

        derivatives = compute_second_derivatives(inverse_result.step1.normalized_coeffs)
        coefficient_space = build_inverse_metric(derivatives; method = :coefficient_space)
        real_space = build_inverse_metric(derivatives; method = :real_space)
        @test interval_coeffs_overlap(coefficient_space.A11, real_space.A11)
        @test interval_coeffs_overlap(coefficient_space.A12, real_space.A12)
        @test interval_coeffs_overlap(coefficient_space.A22, real_space.A22)
        @test interval_coeffs_overlap(coefficient_space.D, real_space.D)
        @test all(isequal_interval.(build_inverse_metric_real_space(derivatives).A21, real_space.A12))

        ricci_numerators_coefficient_space =
            compute_ricci_numerators_from_inverse_coeffs(coefficient_space; method = :coefficient_space)
        ricci_numerators_real_space =
            compute_ricci_numerators_from_inverse_coeffs(coefficient_space; method = :real_space)
        @test interval_coeffs_overlap(ricci_numerators_coefficient_space.R11_num, ricci_numerators_real_space.R11_num)
        @test interval_coeffs_overlap(ricci_numerators_coefficient_space.R12_num, ricci_numerators_real_space.R12_num)
        @test interval_coeffs_overlap(ricci_numerators_coefficient_space.R21_num, ricci_numerators_real_space.R21_num)
        @test interval_coeffs_overlap(ricci_numerators_coefficient_space.R22_num, ricci_numerators_real_space.R22_num)

        ricci_subdivision = bound_ricci_subdivision_local(path, 8, 2)
        @test inf(ricci_subdivision.step5.D_lower) > 0
        @test ricci_subdivision.step5.ricci_norm_bound < 100
        @test ricci_subdivision.step5.R11_bound < 50
        @test ricci_subdivision.step5.R12_bound < 50
        @test ricci_subdivision.step5.R21_bound < 50
        @test ricci_subdivision.step5.R22_bound < 50
    finally
        rm(path; force = true)
    end
end
