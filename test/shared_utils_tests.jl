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

    @testset "repeated JSON updates on Windows" begin
        mktempdir() do dir
            path = joinpath(dir, "nested", "verified_bounds.json")
            # Keep GC disabled so a memory-mapped read cannot be released by
            # chance before the next write (the original pipeline failure).
            gc_enabled = GC.enable(false)
            try
                write_bound_entry("untouched", "value", 7; path)
                for i in 1:20
                    write_bound_entry("ma_first_derivatives", "x", interval(BigFloat(i)); path)
                    @test sup(read_bound("ma_first_derivatives"; path)["x"]) == i
                    write_bound_entry(:ma_first_derivatives, :y, interval(BigFloat(i + 1)); path)
                end
                write_bound_entry("metric_inverse", "d1", (x = (xx = 1, xy = 2),); path)
                @test sup(read_bound("untouched"; path)["value"]) == 7
                @test sup(read_bound("metric_inverse"; path)["d1"]["x"]["xy"]) == 2
                @test sup(read_bound("ma_first_derivatives"; path)["y"]) == 21
                @test read_bounds_json(path)["untouched"]["value"] isa String
            finally
                GC.enable(gc_enabled)
            end
        end
    end

    @testset "serialization preserves bound directions" begin
        scale = big(10)^SERIALIZED_BOUND_DECIMAL_DIGITS
        decimal_rational(s) = parse(BigInt, replace(s, "." => "")) // scale
        setprecision(BigFloat, 256) do
            for x in (big"0", big"1", big"-1", big"0.1", big"-0.1", big"1e-100", big"-1e-100",
                big"8.745171243585107020659805304692021446349081543e-32", big"0.9946068170915334")
                lower = serialize_bound_value(x; rounding = RoundDown)
                upper = serialize_bound_value(x)
                exact = Rational{BigInt}(x)
                @test decimal_rational(lower) <= exact <= decimal_rational(upper)
                @test decimal_rational(upper) - decimal_rational(lower) <= 1 // scale
                @test length(last(split(upper, '.'))) == SERIALIZED_BOUND_DECIMAL_DIGITS
            end
            enclosure = interval(big"0.1", big"0.3")
            @test decimal_rational(serialize_bound_value(enclosure)) >= Rational{BigInt}(sup(enclosure))
            @test decimal_rational(serialize_bound_value(enclosure; rounding = RoundDown)) <= Rational{BigInt}(inf(enclosure))
            mktempdir() do dir
                path = joinpath(dir, "bounds.json")
                write_bound_entry(:curvature_bounds, :ricci_lower_bound, enclosure; path, rounding = RoundDown)
                saved = read_bounds_json(path)["curvature_bounds"]["ricci_lower_bound"]
                @test decimal_rational(saved) <= Rational{BigInt}(inf(enclosure))
            end
            @test_throws ArgumentError serialize_bound_value(big"Inf")
        end
    end
end
