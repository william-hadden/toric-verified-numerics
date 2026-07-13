module EigenvalueAssemblyTests

using Test
using IntervalArithmetic

include(joinpath(@__DIR__, "..", "eigenvalues", "orchestrate.jl"))

"""Count the triangles incident to each edge of a conforming mesh."""
function edge_incidence(triangles)
    incidence = Dict{Tuple{Int,Int},Int}()
    for tri in triangles, edge in triangle_edges(tri)
        incidence[edge] = get(incidence, edge, 0) + 1
    end
    return incidence
end

"""Return whether an edge contains another mesh vertex in its relative interior."""
function has_hanging_vertex(node_key_set, node_keys, edge)
    a, b = node_keys[edge[1]], node_keys[edge[2]]
    dx, dy = b[1] - a[1], b[2] - a[2]
    lattice_length = gcd(abs(dx), abs(dy))
    lattice_length <= 1 && return false
    step = (dx ÷ lattice_length, dy ÷ lattice_length)
    for offset in 1:(lattice_length - 1)
        key = (a[1] + offset * step[1], a[2] + offset * step[2])
        key in node_key_set && return true
    end
    return false
end

"""Return a canonical exact key for a lattice triangle."""
canonical_triangle_key(node_keys, tri) =
    Tuple(sort!([node_keys[vertex] for vertex in tri]))

@testset "graded sector mesh is conforming and D6-compatible" begin
    delta, N = 1 // 200, 8
    mesh = sector_mesh(delta, N, 2, 1, 1, 1)
    incidence = edge_incidence(mesh.triangles)
    node_key_set = Set(mesh.node_keys)
    triangle_key_set = Set(
        canonical_triangle_key(mesh.node_keys, tri) for tri in mesh.triangles
    )

    @test length(node_key_set) == length(mesh.node_keys)
    @test all(lattice_double_area(mesh.node_keys, tri) > 0 for tri in mesh.triangles)
    @test maximum(values(incidence)) <= 2
    @test all(
        !has_hanging_vertex(node_key_set, mesh.node_keys, edge)
        for edge in keys(incidence)
    )
    @test length(mesh.node_keys) - length(incidence) + length(mesh.triangles) == 1
    @test sum(lattice_double_area(mesh.node_keys, tri) for tri in mesh.triangles) ==
          (N * mesh.scale)^2

    reflected_nodes = Set(sector_reflect_key(key) for key in mesh.node_keys)
    @test reflected_nodes == node_key_set
    @test all(
        Tuple(sort!(sector_reflect_key.(collect(key)))) in triangle_key_set
        for key in triangle_key_set
    )

    edges, _ = build_cr_edges(mesh.triangles)
    midpoint_keys = edge_midpoint_keys(edges, mesh.node_keys)
    edge_for_midpoint = Dict(key => index for (index, key) in pairs(midpoint_keys))
    quotient = d6_edge_quotient(edges, mesh.node_keys)
    @test length(edge_for_midpoint) == length(edges)
    @test all(haskey(edge_for_midpoint, sector_reflect_key(key)) for key in midpoint_keys)
    for (index, key) in pairs(midpoint_keys)
        @test quotient[index] == quotient[edge_for_midpoint[sector_reflect_key(key)]]
    end
end

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

@testset "adaptive metric cover and lower-matrix recertification" begin
    indefinite_box = [
        interval(BigFloat(1)) interval(BigFloat(-2), BigFloat(2))
        interval(BigFloat(-2), BigFloat(2)) interval(BigFloat(1))
    ]
    @test symmetric_eigenvalue_lower_bound(indefinite_box) <= 0

    mesh = sector_mesh(1 // 100, 8, 3, 1, 0, 1)
    oracle = InverseMetricOracle(30)
    triangle = mesh.triangles[436]
    bound = inverse_metric_lower_bound(
        oracle,
        mesh.nodes,
        triangle,
        0,
        8,
        8,
        40,
    )

    @test length(bound.leaves) > 1
    @test maximum(leaf.depth for leaf in bound.leaves) > 0
    @test all(symmetric_eigenvalue_lower_bound(leaf.G) > 0 for leaf in bound.leaves)
    alpha = symmetric_eigenvalue_lower_bound(interval.(bound.B))
    residuals = residual_eigenvalue_lower_bounds(bound.leaves, bound.B)
    @test alpha == bound.alpha > 0
    @test all(bound -> bound > 0, residuals)
end

"""Oracle used to exercise the quality-driven subdivision loop cheaply."""
struct QualityTestOracle end

"""Return a lower matrix which crosses the quality target after one depth."""
function inverse_metric_lower_bound(
    ::QualityTestOracle,
    nodes,
    tri,
    min_depth,
    degree,
    terms,
    bisection_steps,
)
    diameter = triangle_diameter_upper(nodes, tri)
    factor = BigFloat(1893) / BigFloat(10000)
    threshold = BigFloat(7) / BigFloat(50)
    crossing_alpha = (factor * diameter / threshold)^2
    alpha = min_depth == 0 ? crossing_alpha / BigFloat(4) :
        crossing_alpha * BigFloat(4)
    B = BigFloat[alpha 0; 0 alpha]
    return (; B, alpha, leaves = NamedTuple[])
end

@testset "quality target increases the minimum cover depth" begin
    mesh = sector_mesh(1 // 10, 2, 0, 1, 0, 1)
    triangle = first(mesh.triangles)
    threshold = BigFloat(7) / BigFloat(50)
    certificate = certify_element(
        QualityTestOracle(),
        mesh,
        triangle,
        8,
        8,
        40,
        threshold,
    )

    @test certificate.liu_constant <= threshold
    @test certificate.alpha > 0
end

@testset "D6 quotient assembly and MATLAB interval data" begin
    mesh = sector_mesh(1 // 10, 4, 1, 1, 1, 1)
    identity_B = BigFloat[1 0; 0 1]
    certificates = [
        (; B = copy(identity_B), alpha = BigFloat(1), liu_constant = BigFloat(1) / 10)
        for _ in mesh.triangles
    ]
    assembly = assemble_matrices(mesh, certificates)

    @test assembly.matrix_size > 1
    @test !isempty(assembly.mass)
    @test !isempty(assembly.stiffness)
    @test all(isguaranteed, values(assembly.mass))
    @test all(isguaranteed, values(assembly.stiffness))
    for row in 1:assembly.matrix_size
        row_sum = interval(BigFloat(0))
        for column in 1:assembly.matrix_size
            row_sum += get(
                assembly.stiffness,
                (row, column),
                interval(BigFloat(0)),
            )
        end
        @test inf(row_sum) <= 0 <= sup(row_sum)
    end

    mktempdir(joinpath(@__DIR__, "..", "eigenvalues")) do directory
        write_matlab_matrices(assembly, directory)
        for filename in ("stiff_matrix.mat", "mass_matrix.mat")
            variables = read_mat4_variables(joinpath(directory, filename))
            @test Set(keys(variables)) == Set(["n", "i", "j", "lo", "hi"])
            @test variables["n"][1] == assembly.matrix_size
            @test length(variables["i"]) == length(variables["j"])
            @test length(variables["lo"]) == length(variables["hi"])
            @test all(variables["lo"] .<= variables["hi"])
        end

        result_path = joinpath(directory, "verified_eigenvalue.mat")
        write_mat4(result_path, [
            "lambda_fem_lb" => 5.1,
            "lambda_fem_ub" => 5.2,
            "lambda_fem_ind" => 2,
        ])
        @test verified_fem_endpoints(result_path) == (5.1, 5.2)

        write_mat4(result_path, [
            "lambda_fem_lb" => 5.2,
            "lambda_fem_ub" => 5.1,
            "lambda_fem_ind" => 2,
        ])
        @test_throws ErrorException verified_fem_endpoints(result_path)

        write_mat4(result_path, [
            "lambda_fem_lb" => 5.1,
            "lambda_fem_ub" => 5.2,
            "lambda_fem_ind" => 1,
        ])
        @test_throws ErrorException verified_fem_endpoints(result_path)
    end
end

@testset "Liu comparison and directed lower serialization" begin
    transformed = liu_lower_bound(BigFloat(6), BigFloat("0.2"))
    exact_value = BigFloat(6) / (BigFloat(1) + BigFloat("0.2")^2 * BigFloat(6))
    @test BigFloat("4.83") < transformed <= exact_value

    value = BigFloat("0.516807414131973603069171907824031294")
    text = directed_lower_decimal(value, 77)
    parsed_upper = setrounding(BigFloat, RoundUp) do
        parse(BigFloat, text)
    end
    @test parsed_upper <= value
    @test length(split(text, '.')[2]) == 77

    mktempdir(joinpath(@__DIR__, "..", "eigenvalues")) do directory
        path = joinpath(directory, "bounds.json")
        bounds = Dict{String,Any}(
            "unrelated" => Dict("kept" => true),
            "lambda_1_delta_lower_bound" => "old inset",
            "lambda_1_lower_bound" => "old compact",
        )
        write_eigenvalue_bounds(
            path,
            bounds,
            BigFloat("0.75"),
            BigFloat("0.5"),
            77,
        )
        written = JSON.parsefile(path)
        @test written["unrelated"] == Dict("kept" => true)
        @test parse(BigFloat, written["lambda_1_delta_lower_bound"]) <=
              BigFloat("0.75")
        @test parse(BigFloat, written["lambda_1_lower_bound"]) <= BigFloat("0.5")
    end
end

end
