@testset "small rigorous inverse and Ricci bounds" begin
    # Small end-to-end integration test using a 3x3 truncation of the real data.
    path = write_small_metric_coeffs_csv()
    output_dir = mktempdir()
    output_path = joinpath(output_dir, "verified_bounds.json")

    try
        write_bound_entry("ma_first_derivatives", "x", 3; path = output_path)
        inverse_result = redirect_stdout(devnull) do
            subdivide_and_bound_rigorous_local(path, 8, 2; output_path)
        end
        @test inf(inverse_result.step4.D_lower) > 0
        @test all(isguaranteed, inverse_result.step3.A11)
        @test all(isguaranteed, inverse_result.step3.A12)
        @test all(isguaranteed, inverse_result.step3.A22)
        @test all(isguaranteed, inverse_result.step3.D)
        @test inverse_result.step4.u11_bound < 10
        @test inverse_result.step4.u12_bound < 10
        @test inverse_result.step4.u22_bound < 10
        saved_inverse = read_bound("metric_inverse"; path = output_path)
        @test Set(keys(saved_inverse)) == Set(("value", "d1", "d2"))
        @test Set(keys(saved_inverse["d1"])) == Set(("x", "y"))
        @test Set(keys(saved_inverse["d2"])) == Set(("xx", "xy", "yx", "yy"))
        @test sup(read_bound("ma_first_derivatives"; path = output_path)["x"]) == 3
        for (group, entries) in pairs(inverse_result.metric_inverse)
            saved = saved_inverse[String(group)]
            if group == :value
                @test all(sup(saved[key]) >= bound for (key, bound) in entries)
            else
                for (derivative, components) in entries
                    @test Set(keys(saved[derivative])) == Set(("xx", "xy", "yx", "yy"))
                    @test all(isfinite(bound) && bound >= 0 && sup(saved[derivative][key]) >= bound
                        for (key, bound) in components)
                    @test sup(saved[derivative]["xy"]) == sup(saved[derivative]["yx"])
                end
            end
        end
        @test all(isequal_interval(saved_inverse["d2"]["xy"][key], saved_inverse["d2"]["yx"][key])
            for key in keys(saved_inverse["d2"]["xy"]))

        ricci_result = redirect_stdout(devnull) do
            bound_ricci_subdivision_local(path, 8, 2)
        end
        @test inf(ricci_result.step5.D_lower) > 0
        @test ricci_result.step5.ricci_norm_bound < 100
        @test ricci_result.step5.R11_bound < 50
        @test ricci_result.step5.R12_bound < 50
        @test ricci_result.step5.R21_bound < 50
        @test ricci_result.step5.R22_bound < 50
    finally
        rm(path; force = true)
        rm(output_dir; recursive = true)
    end
end

@testset "inverse derivative denominator powers and positivity" begin
    # u = 1/(2+t), t = 2x-1: the exact suprema of u, u_x, u_xx
    # on [-1,1] are 1, 2, 8, respectively.
    numerators = Dict((1, 1, (0, 0)) => reshape([interval(BigFloat(1))], 1, 1),
        (1, 1, (1, 0)) => reshape([interval(BigFloat(-2))], 1, 1),
        (1, 1, (2, 0)) => reshape([interval(BigFloat(8))], 1, 1))
    D = reshape(interval.(BigFloat[2, 1]), 2, 1)
    for pdeg in (1, 2) # pdeg=1 places T_1 in the denominator tail.
        result = bound_inverse_derivatives_by_local_subdivision((deriv_num = numerators, D = D);
            pdeg, nx = 2)
        @test result.D_lower == 1
        @test result.derivative_bounds[(1, 1, (0, 0))] == 1
        @test result.derivative_bounds[(1, 1, (1, 0))] == 2
        @test result.derivative_bounds[(1, 1, (2, 0))] == 8
    end
    @test_throws ErrorException bound_inverse_derivatives_by_local_subdivision(
        (deriv_num = numerators, D = zeros(Interval{BigFloat}, 1, 1)); pdeg = 1, nx = 1)
end
