import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using Random
using IntervalArithmetic

include(joinpath(@__DIR__, "..", "bound_residual", "util", "io.jl"))
include(joinpath(@__DIR__, "..", "bound_residual", "util", "chebyshev_algebra.jl"))

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

intervalize(A::AbstractMatrix{<:Real}) = interval.(A)

function interval_methods_enclose_direct(A::AbstractMatrix{<:Interval}, B::AbstractMatrix{<:Interval}, method::Symbol)
    direct = cheb_mul2(A, B; method = :direct)
    candidate = cheb_mul2(A, B; method = method)
    return interval_coeffs_enclose(direct, candidate)
end

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

@testset "full real data interval coefficient multiplication" begin
    coeffs = load_coeffs_csv(joinpath(@__DIR__, "..", "data", "happrox", "coeffs.csv"))
    @test size(coeffs) == (80, 80)

    block = interval.(coeffs)

    println("\nRunning full 80x80 rigorous comparison...")

    t_direct = @elapsed begin
        global direct = cheb_mul2(block, block; method = :direct)
    end

    println("direct interval multiplication time: ", t_direct, " seconds")

    t_dct = @elapsed begin
        global dct = cheb_mul2(block, block; method = :interval_dct)
    end

    println("interval DCT multiplication time:   ", t_dct, " seconds")
    println("DCT/direct ratio:                   ", t_dct / t_direct)

    @test size(direct) == size(dct)

    overlap_count = 0
    total_count = length(direct)

    for I in eachindex(direct)
        overlap =
            inf(direct[I]) <= sup(dct[I]) &&
            inf(dct[I]) <= sup(direct[I])

        @test overlap

        if overlap
            overlap_count += 1
        end
    end

    println("overlapping coefficients: ", overlap_count, " / ", total_count)
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
