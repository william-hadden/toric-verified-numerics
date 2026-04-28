import Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using Random
using IntervalArithmetic

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

        A = intervalize([0.0 1.0; 2.0 -1.0])
        B = intervalize([1.0 -2.0 0.5; -0.25 0.0 3.0])
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
