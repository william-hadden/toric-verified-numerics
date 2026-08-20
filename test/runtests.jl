import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using Random
using IntervalArithmetic

test_filter = isempty(ARGS) ? "all" : ARGS[1]

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "bound_inverse.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "bound_ricci.jl"))
include(joinpath(@__DIR__, "metric_geometry_test_helpers.jl"))

function pad_to_common_size(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number})
    degx = max(size(A, 1), size(B, 1)) - 1
    degy = max(size(A, 2), size(B, 2)) - 1
    T = promote_type(eltype(A), eltype(B))
    return cheb_pad(T.(A), degx, degy), cheb_pad(T.(B), degx, degy)
end

function coeffs_match(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number}; atol::Real = 1e-10, rtol::Real = 1e-10)
    A_pad, B_pad = pad_to_common_size(A, B)
    return isapprox(A_pad, B_pad; atol, rtol)
end

function interval_coeffs_enclose(A::AbstractMatrix{<:Interval}, B::AbstractMatrix{<:Interval})
    A_pad, B_pad = pad_to_common_size(A, B)
    for I in eachindex(A_pad, B_pad)
        issubset_interval(A_pad[I], B_pad[I]) || return false
    end
    return true
end

function interval_coeffs_overlap(A::AbstractMatrix{<:Interval}, B::AbstractMatrix{<:Interval})
    A_pad, B_pad = pad_to_common_size(A, B)
    for I in eachindex(A_pad, B_pad)
        sup(A_pad[I]) < inf(B_pad[I]) && return false
        sup(B_pad[I]) < inf(A_pad[I]) && return false
    end
    return true
end

interval_overlaps(A::Interval, B::Interval) = inf(A) <= sup(B) && inf(B) <= sup(A)
interval_endpoints_approx(A::Interval, B::Interval; rtol = big"1e-25", atol = big"1e-25") =
    isapprox(inf(A), inf(B); rtol, atol) && isapprox(sup(A), sup(B); rtol, atol)

max_interval_width(A::AbstractMatrix{<:Interval}) = maximum(sup(x) - inf(x) for x in A)

intervalize(A::AbstractMatrix{<:Real}) = interval.(A)

function interval_methods_enclose_direct(A::AbstractMatrix{<:Interval}, B::AbstractMatrix{<:Interval}, method::Symbol)
    direct = cheb_mul2(A, B; method = :direct)
    candidate = cheb_mul2(A, B; method = method)
    return interval_coeffs_enclose(direct, candidate)
end

if test_filter in ("all", "quotient")
    include(joinpath(@__DIR__, "quotient_tests.jl"))
end

if test_filter in ("all", "ricci")
    include(joinpath(@__DIR__, "ricci_tests.jl"))
end

if test_filter in ("all", "inverse")
    include(joinpath(@__DIR__, "inverse_tests.jl"))
end

if test_filter in ("all", "eigenvalues")
    include(joinpath(@__DIR__, "eigenvalue_assembly_tests.jl"))
end

if test_filter in ("all", "utils")
    include(joinpath(@__DIR__, "shared_utils_tests.jl"))
end

if test_filter in ("all", "cheb")

@testset "2D Chebyshev multiplication via DCT" begin
    @testset "simple exact cases" begin
        A = reshape([2.0], 1, 1)
        B = reshape([-3.0], 1, 1)
        @test coeffs_match(cheb_mul2(A, B; method = :direct), cheb_mul2(A, B; method = :dct); atol = 1e-12, rtol = 1e-12)

        A = [0.0 1.0; 2.0 -1.0]
        B = [1.0 -2.0 0.5; -0.25 0.0 3.0]
        @test coeffs_match(cheb_mul2(A, B; method = :direct), cheb_mul2(A, B; method = :dct); atol = 1e-11, rtol = 1e-11)

        A = [0.0 0.0 0.0; 0.0 0.0 0.0; 1.0 0.0 0.0]
        B = [0.0 2.0; 0.0 0.0; 0.0 -1.0]
        @test coeffs_match(cheb_mul2(A, B; method = :direct), cheb_mul2(A, B; method = :dct); atol = 1e-11, rtol = 1e-11)
    end

    @testset "seeded random cases" begin
        rng = MersenneTwister(20260428)

        for sizeA in ((1, 1), (2, 3), (3, 2), (4, 4), (5, 3))
            for sizeB in ((1, 1), (2, 2), (3, 4), (4, 3))
                A = randn(rng, sizeA...)
                B = randn(rng, sizeB...)

                # Introduce exact zeros so trimming paths are exercised too.
                A[rand(rng, eachindex(A), min(2, length(A)))] .= 0.0
                B[rand(rng, eachindex(B), min(2, length(B)))] .= 0.0

                direct = cheb_mul2(A, B; method = :direct)
                dct = cheb_mul2(A, B; method = :dct)
                @test coeffs_match(direct, dct; atol = 1e-10, rtol = 1e-10)
            end
        end
    end

    @testset "interval DCT enclosures" begin
        for x in (BigInt(1) // BigInt(3), BigInt(1) // BigInt(19))
            X = cheb_interval_constant(Interval{BigFloat}, x)

            @test Rational{BigInt}(inf(X)) <= x
            @test x <= Rational{BigInt}(sup(X))
        end

        A = reshape([interval(2)], 1, 1)
        B = reshape([interval(-3)], 1, 1)
        @test interval_coeffs_enclose(cheb_mul2(A, B; method = :direct), cheb_mul2(A, B; method = :interval_dct))

        A = reshape(interval.([0, 1]), 2, 1)
        B = reshape(interval.([0, 1]), 2, 1)
        direct = cheb_mul2(A, B; method = :direct)
        interval_dct = cheb_mul2(A, B; method = :interval_dct)
        expected = reshape(interval.([1 // 2, 0, 1 // 2]), 3, 1)
        @test interval_coeffs_enclose(direct, interval_dct)
        @test interval_coeffs_enclose(expected, interval_dct)

        A = reshape(interval.([0, 0, 1, 0]), 4, 1)
        B = reshape(interval.([0, 0, 0, 1]), 4, 1)
        expected = reshape(interval.([0, 1 // 2, 0, 0, 0, 1 // 2]), 6, 1)
        @test interval_methods_enclose_direct(A, B, :interval_dct)
        @test interval_coeffs_enclose(expected, cheb_mul2(A, B; method = :interval_dct))

        A = intervalize([
            0.0 0.0 0.0;
            0.0 0.0 1.0;
            0.0 0.0 0.0;
        ])
        B = intervalize([
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 1.0 0.0;
        ])
        expected = intervalize([
            0.0 0.0 0.0 0.0;
            0.0 0.25 0.0 0.25;
            0.0 0.0 0.0 0.0;
            0.0 0.25 0.0 0.25;
        ])
        @test interval_methods_enclose_direct(A, B, :interval_dct)
        @test interval_coeffs_enclose(expected, cheb_mul2(A, B; method = :interval_dct))

        A = intervalize([0.0 1.0; 2.0 -1.0])
        B = intervalize([1.0 -2.0 0.5; -0.25 0.0 3.0])
        @test interval_coeffs_enclose(cheb_mul2(A, B; method = :direct), cheb_mul2(A, B; method = :interval_dct))

        A = reshape([interval(-1e8, -1e8 + 1e-6), interval(1e8 - 1e-6, 1e8)], 2, 1)
        B = reshape([interval(1e8 - 1e-6, 1e8), interval(-1e8, -1e8 + 1e-6)], 2, 1)
        @test interval_coeffs_enclose(cheb_mul2(A, B; method = :direct), cheb_mul2(A, B; method = :interval_dct))

        A = interval.(fill(1.0, 3, 3) .- 1e-15, fill(1.0, 3, 3) .+ 1e-15)
        B = interval.(fill(-1.0, 3, 3) .- 1e-15, fill(-1.0, 3, 3) .+ 1e-15)
        @test interval_coeffs_enclose(cheb_mul2(A, B; method = :direct), cheb_mul2(A, B; method = :interval_dct))

        @test_throws ErrorException cheb_mul2(A, B; method = :dct)
    end

    @testset "seeded random interval cases" begin
        rng = MersenneTwister(20260428)

        for sizeA in ((1, 1), (2, 2), (3, 2), (3, 3))
            for sizeB in ((1, 1), (2, 3), (3, 2))
                midA = randn(rng, sizeA...)
                midB = randn(rng, sizeB...)
                radA = 0.05 .* rand(rng, sizeA...)
                radB = 0.05 .* rand(rng, sizeB...)

                A = interval.(midA .- radA, midA .+ radA)
                B = interval.(midB .- radB, midB .+ radB)

                A[rand(rng, eachindex(A), min(2, length(A)))] .= interval(0)
                B[rand(rng, eachindex(B), min(2, length(B)))] .= interval(0)

                direct = cheb_mul2(A, B; method = :direct)
                interval_dct = cheb_mul2(A, B; method = :interval_dct)
                @test interval_coeffs_enclose(direct, interval_dct)
            end
        end
    end

end

@testset "harder deterministic interval cases" begin
    # Mixed degrees with exact rational coefficients
    A = intervalize([
        1.0  -2.0   0.5
        0.25  0.0  -1.5
        2.0   1.0   0.0
        -0.75 0.25  3.0
    ])

    B = intervalize([
        -1.0  0.5
         2.0 -3.0
         0.0  1.25
    ])

    direct = cheb_mul2(A, B; method = :direct)
    interval_dct = cheb_mul2(A, B; method = :interval_dct)
    @test interval_coeffs_enclose(direct, interval_dct)

    # Non-point intervals
    A = interval.(
        [-1.0  0.0  2.0; 0.5 -0.25 1.0],
        [-0.9  0.1  2.1; 0.6 -0.15 1.1],
    )

    B = interval.(
        [0.9 -2.0; -0.5 1.0; 2.0 0.0],
        [1.1 -1.8; -0.3 1.2; 2.2 0.2],
    )

    direct = cheb_mul2(A, B; method = :direct)
    interval_dct = cheb_mul2(A, B; method = :interval_dct)
    @test interval_coeffs_enclose(direct, interval_dct)
end

@testset "known Chebyshev identities in 2D" begin
    # T_2(x) * T_3(x) = 1/2 T_5(x) + 1/2 T_1(x)
    A = zeros(Interval{Float64}, 3, 1)
    B = zeros(Interval{Float64}, 4, 1)
    A[3, 1] = interval(1)
    B[4, 1] = interval(1)

    expected = zeros(Interval{Float64}, 6, 1)
    expected[2, 1] = interval(1//2)
    expected[6, 1] = interval(1//2)

    interval_dct = cheb_mul2(A, B; method = :interval_dct)
    @test interval_coeffs_enclose(expected, interval_dct)

    # T_1(x)T_2(y) * T_2(x)T_1(y)
    # = 1/4 (T_3(x)+T_1(x))(T_3(y)+T_1(y))
    A = zeros(Interval{Float64}, 2, 3)
    B = zeros(Interval{Float64}, 3, 2)
    A[2, 3] = interval(1)
    B[3, 2] = interval(1)

    expected = zeros(Interval{Float64}, 4, 4)
    expected[4, 4] = interval(1//4)
    expected[4, 2] = interval(1//4)
    expected[2, 4] = interval(1//4)
    expected[2, 2] = interval(1//4)

    interval_dct = cheb_mul2(A, B; method = :interval_dct)
    @test interval_coeffs_enclose(expected, interval_dct)
end

@testset "larger seeded interval stress cases" begin
    rng = MersenneTwister(20260428)

    for sizeA in ((4, 4), (5, 3), (3, 5), (6, 6))
        for sizeB in ((3, 4), (4, 3), (5, 5))
            midA = randn(rng, sizeA...)
            midB = randn(rng, sizeB...)
            radA = 1e-3 .* rand(rng, sizeA...)
            radB = 1e-3 .* rand(rng, sizeB...)

            A = interval.(midA .- radA, midA .+ radA)
            B = interval.(midB .- radB, midB .+ radB)

            direct = cheb_mul2(A, B; method = :direct)
            interval_dct = cheb_mul2(A, B; method = :interval_dct)

            @test interval_coeffs_enclose(direct, interval_dct)
        end
    end
end

@testset "BigFloat interval DCT" begin
    setprecision(BigFloat, 256) do
        A_mid = BigFloat[
            big"1.25"   big"-0.5"   big"0.125"  big"0.0";
            big"0.75"   big"0.0"    big"-0.25" big"0.5";
            big"-0.125" big"0.375"  big"1.0"   big"-0.625";
            big"0.5"    big"-0.75"  big"0.25"  big"0.875";
        ]

        B_mid = BigFloat[
            big"-0.375" big"0.25"   big"0.5"   big"-0.125";
            big"0.625"  big"-0.875" big"0.0"   big"0.375";
            big"0.25"   big"0.125"  big"-0.5"  big"0.75";
            big"-0.5"   big"0.625"  big"-0.25" big"0.0";
        ]

        rad = big"1e-60"

        A_big = interval.(A_mid .- rad, A_mid .+ rad)
        B_big = interval.(B_mid .- rad, B_mid .+ rad)

        direct_big = cheb_mul2(A_big, B_big; method = :direct)
        dct_big = cheb_mul2(A_big, B_big; method = :interval_dct)

        @test eltype(dct_big) == Interval{BigFloat}
        @test interval_coeffs_enclose(direct_big, dct_big)
        @test interval_coeffs_overlap(direct_big, dct_big)

        A64 = interval.(Float64.(A_mid) .- 1e-16, Float64.(A_mid) .+ 1e-16)
        B64 = interval.(Float64.(B_mid) .- 1e-16, Float64.(B_mid) .+ 1e-16)

        dct64 = cheb_mul2(A64, B64; method = :interval_dct)

        width_big = max_interval_width(dct_big)
        width_64 = BigFloat(max_interval_width(dct64))
        improvement = width_64 / width_big

        println("\nBigFloat interval DCT diagnostics:")
        println("  max width (Float64 DCT):   ", width_64)
        println("  max width (BigFloat DCT):  ", width_big)
        println("  improvement factor:        ", improvement)
        println("  log10 improvement:         ", log10(improvement))

        @test width_big < big"1e-40"
        @test width_big < width_64
    end
end

@testset "timing sanity check" begin
    rng = MersenneTwister(20260428)

    for n in (8, 16, 24, 32, 64)
        A = interval.(randn(rng, n, n))
        B = interval.(randn(rng, n, n))

        cheb_mul2(A, B; method = :direct)
        cheb_mul2(A, B; method = :interval_dct)

        t_direct = minimum(@elapsed cheb_mul2(A, B; method = :direct) for _ in 1:3)
        t_interval_dct = minimum(@elapsed cheb_mul2(A, B; method = :interval_dct) for _ in 1:3)

        println("n = ", n)
        println("  interval direct time:   ", t_direct)
        println("  full interval DCT time: ", t_interval_dct)
        println("  full interval DCT/direct: ", t_interval_dct / t_direct)
    end
end

@testset "full real data interval coefficient multiplication" begin
    coeffs = load_rational_coeffs_csv(U0_PATH)
    @test size(coeffs) == (80, 80)

    block = coeffs

    println("\nRunning full 80x80 rigorous comparison...")

    direct_ref = Ref{Any}()
    dct_ref = Ref{Any}()

    t_direct = @elapsed begin
        direct_ref[] = cheb_mul2(block, block; method = :direct)
    end

    t_dct = @elapsed begin
        dct_ref[] = cheb_mul2(block, block; method = :interval_dct)
    end

    direct = direct_ref[]
    dct = dct_ref[]

    println("direct interval multiplication time: ", t_direct, " seconds")
    println("interval DCT multiplication time:   ", t_dct, " seconds")
    println("DCT/direct ratio:                   ", t_dct / t_direct)

    @test size(direct) == size(dct)

    overlaps = [
        inf(direct[I]) <= sup(dct[I]) &&
        inf(dct[I]) <= sup(direct[I])
        for I in eachindex(direct)
    ]

    println("overlapping coefficients: ", count(identity, overlaps), " / ", length(overlaps))

    for ok in overlaps
        @test ok
    end

    direct_widths = [sup(direct[I]) - inf(direct[I]) for I in eachindex(direct)]
    dct_widths = [sup(dct[I]) - inf(dct[I]) for I in eachindex(dct)]

    abs_width_diffs = dct_widths .- direct_widths

    println("\nInterval width diagnostics:")
    println("  max direct width:        ", maximum(direct_widths))
    println("  max interval DCT width:  ", maximum(dct_widths))
    println("  mean direct width:       ", sum(direct_widths) / length(direct_widths))
    println("  mean interval DCT width: ", sum(dct_widths) / length(dct_widths))

    num_tighter_dct = count(dct_widths .< direct_widths)
    num_tighter_direct = count(direct_widths .< dct_widths)
    num_equal = length(direct_widths) - num_tighter_dct - num_tighter_direct

    println("\nWhich method is tighter coefficientwise?")
    println("  DCT tighter:     ", num_tighter_dct)
    println("  direct tighter:  ", num_tighter_direct)
    println("  equal widths:    ", num_equal)

    println("\nAbsolute width difference, DCT - direct:")
    println("  min:  ", minimum(abs_width_diffs))
    println("  mean: ", sum(abs_width_diffs) / length(abs_width_diffs))
    println("  max:  ", maximum(abs_width_diffs))

    ratio_tol = 1e-12

    width_ratios = [
        dct_widths[i] / direct_widths[i]
        for i in eachindex(direct_widths)
        if direct_widths[i] > ratio_tol
    ]

    println("\nDCT width / direct width, excluding direct widths <= $(ratio_tol):")

    if isempty(width_ratios)
        println("  No coefficients had sufficiently large direct width for stable ratio analysis.")
    else
        sorted_ratios = sort(width_ratios)
        median_ratio = sorted_ratios[cld(length(sorted_ratios), 2)]

        println("  number used:  ", length(width_ratios))
        println("  min ratio:    ", minimum(width_ratios))
        println("  median ratio: ", median_ratio)
        println("  mean ratio:   ", sum(width_ratios) / length(width_ratios))
        println("  max ratio:    ", maximum(width_ratios))
    end
end

end # test_filter in ("all", "cheb")
