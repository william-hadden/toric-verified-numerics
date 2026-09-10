function interval_one_series()
    return cheb_constant(interval(BigFloat(1)))
end

function interval_coordinate_series()
    half = interval(BigFloat(1)) / exact(2)
    x = zeros(Interval{BigFloat}, 2, 1)
    y = zeros(Interval{BigFloat}, 1, 2)
    x[1, 1] = half
    x[2, 1] = half
    y[1, 1] = half
    y[1, 2] = -half
    return (; x, y)
end

function assert_coeff_arrays_equal(A::AbstractMatrix{<:Interval}, B::AbstractMatrix{<:Interval})
    A_pad, B_pad = pad_to_common_size(A, B)
    @test size(A_pad) == size(B_pad)
    for I in eachindex(A_pad, B_pad)
        @test isequal_interval(A_pad[I], B_pad[I])
    end
end

function assert_constant_series_equals(coeffs::AbstractMatrix{<:Interval}, expected)
    expected_interval = interval(BigFloat(expected))
    @test issubset_interval(expected_interval, coeffs[1, 1])
    @test issubset_interval(coeffs[1, 1], expected_interval)

    for j in axes(coeffs, 2), i in axes(coeffs, 1)
        i == 1 && j == 1 && continue
        @test isequal_interval(coeffs[i, j], zero(coeffs[i, j]))
    end
end

function write_small_metric_coeffs_csv()
    coeffs = BigFloat[
        big"-0.1183128879989351138987357632027417962631449084422710851994831902791517152301179"  big"0.06009238672538315396769305574076245997880534096947407772171244476878500026485116"   big"-0.03144666450015627974946616561725583808294468443121282301610258991790578336353206";
        big"-0.06009238672538315396769305574076245997880534096947407772171244476878500026485116"  big"-0.0594819779673940611826914678587690292345150796980406665670785989316781594745433"  big"-0.0001806009712969379278800194966186249681422357697732228068511973442733248131942189";
        big"-0.03144666450015627974946616561725583808294468443121282301610258991790578336353206"  big"0.0001806009712969379278800194966186249681422357697732228068511973442733248131942189"   big"-0.001243192889660587461377157087924345167414465649990736286776874443255145858549931";
    ]

    path = tempname() * ".csv"
    open(path, "w") do io
        for i in axes(coeffs, 1)
            entries = map(coeffs[i, :]) do coefficient
                rational = Rational{BigInt}(coefficient)
                string(numerator(rational), '/', denominator(rational))
            end
            println(io, join(entries, ","))
        end
    end
    return path
end
