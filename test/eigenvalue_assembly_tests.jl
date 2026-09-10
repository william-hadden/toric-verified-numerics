module EigenvalueAssemblyTests

using Test
using IntervalArithmetic

include(joinpath(@__DIR__, "..", "eigenvalues", "orchestrate.jl"))
include(joinpath(@__DIR__, "..", "eigenvalues", "update_bound.jl"))

@testset "reciprocal Neumann contraction" begin
    contracting = reshape([interval(BigFloat(2)), interval(BigFloat(1) / 2)], 2, 1)
    reciprocal = reciprocal_polynomial_neumann(contracting; deg = 4, terms = 4)
    @test !isnothing(reciprocal)
    @test reciprocal.ratio < 1

    noncontracting = reshape([interval(BigFloat(1)), interval(BigFloat(2))], 2, 1)
    @test isnothing(reciprocal_polynomial_neumann(noncontracting; deg = 4, terms = 4))

    zero_center = reshape([interval(BigFloat(-1), BigFloat(1))], 1, 1)
    @test isnothing(reciprocal_polynomial_neumann(zero_center; deg = 4, terms = 4))
end

@testset "symmetric lower-matrix bound" begin
    indefinite_box = [
        interval(BigFloat(1)) interval(BigFloat(-2), BigFloat(2))
        interval(BigFloat(-2), BigFloat(2)) interval(BigFloat(1))
    ]
    @test symmetric_eigenvalue_lower_bound(indefinite_box) <= 0
end

@testset "Liu comparison and lower-bound serialization" begin
    transformed = liu_lower_bound(BigFloat(6), BigFloat("0.2"))
    exact_value = BigFloat(6) / (BigFloat(1) + BigFloat("0.2")^2 * BigFloat(6))
    @test BigFloat("4.83") < transformed <= exact_value

    value = BigFloat("0.516807414131973603069171907824031294")
    text = serialize_bound_value(value; rounding = RoundDown)
    parsed_upper = setrounding(BigFloat, RoundUp) do
        parse(BigFloat, text)
    end
    @test parsed_upper <= value
end

end
