@testset "shared I/O, interval, and Chebyshev utilities" begin
    @test chebyshev_values(interval(0), 3) == interval.([1, 0, -1, 0])

    coeffs = interval.([1 2; 3 4])
    trunc = truncate_coeffs_with_tail(coeffs, 1)
    @test trunc.coeffs == reshape([interval(1)], 1, 1)
    @test sup(trunc.tail) == 9

    mktemp() do path, io
        write(io, "1/2,-3/4\n5/6,7/8\n")
        flush(io)
        loaded = load_rational_coeffs_csv(path)
        @test loaded[1, 1] == interval(BigInt(1)) / interval(BigInt(2))
        @test loaded[2, 2] == interval(BigInt(7)) / interval(BigInt(8))
    end

    mktemp() do path, io
        write(io, "0.125,-2.5\n3.0,4.25\n")
        flush(io)
        loaded = load_metric_coeffs_csv(path)
        @test 0.125 ∈ loaded[1, 1]
        @test -2.5 ∈ loaded[1, 2]
    end

    mktemp() do path, io
        close(io)
        rm(path)
        write_bound_entry("test_group", "test_value", interval(BigFloat(3)); path)
        @test issubset_interval(interval(BigFloat(3)), read_bound("test_group"; path)["test_value"])
    end
end
