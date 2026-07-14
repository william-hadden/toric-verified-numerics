@testset "small rigorous inverse and Ricci bounds" begin
    # Small end-to-end integration test using a 3x3 truncation of the real data.
    path = write_small_metric_coeffs_csv()

    try
        inverse_result = subdivide_and_bound_rigorous_local(path, 8, 2)
        @test inf(inverse_result.step4.D_lower) > 0
        @test all(isguaranteed, inverse_result.step3.A11)
        @test all(isguaranteed, inverse_result.step3.A12)
        @test all(isguaranteed, inverse_result.step3.A22)
        @test all(isguaranteed, inverse_result.step3.D)
        @test inverse_result.step4.u11_bound < 10
        @test inverse_result.step4.u12_bound < 10
        @test inverse_result.step4.u22_bound < 10

        ricci_result = bound_ricci_subdivision_local(path, 8, 2)
        @test inf(ricci_result.step5.D_lower) > 0
        @test ricci_result.step5.ricci_norm_bound < 100
        @test ricci_result.step5.R11_bound < 50
        @test ricci_result.step5.R12_bound < 50
        @test ricci_result.step5.R21_bound < 50
        @test ricci_result.step5.R22_bound < 50
    finally
        rm(path; force = true)
    end
end
