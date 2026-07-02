include(joinpath(@__DIR__, "..", "eigenvalues", "assemble_matrices.jl"))

interval_midpoint(a) = (inf(a) + sup(a)) / 2

reflect_lower_square_key((x, y)) = (x, -x - y)
triangle_key(keys) = Tuple(sort!(collect(keys)))

@testset "metric stiffness is invariant under lower-square reflection" begin
    pmesh = pdelta_mesh_data()
    nodes, triangles = pmesh.nodes, pmesh.triangles
    node_for_key = Dict(pmesh.node_keys[i] => i for i in eachindex(pmesh.node_keys))
    triangle_for_key = Dict(
        triangle_key(pmesh.node_keys[i] for i in tri) => t
        for (t, tri) in pairs(triangles)
    )
    oracle = InverseMetricBoxOracleV2()

    for t in (1, 37, 80, 130, 196)
        reflected_key = triangle_key(reflect_lower_square_key(pmesh.node_keys[i]) for i in triangles[t])
        reflected_t = triangle_for_key[reflected_key]
        U = metric_integral_for_single_triangle(oracle, nodes, triangles[t])
        reflected_U = metric_integral_for_single_triangle(oracle, nodes, triangles[reflected_t])
        K = local_cr_stiffness_matrix_from_integrals(nodes, triangles[t], U)
        reflected_K = local_cr_stiffness_matrix_from_integrals(nodes, triangles[reflected_t], reflected_U)

        edge_perm = ntuple(3) do a
            edge = local_cr_edges(triangles[t])[a]
            reflected_edge = edge_key(
                node_for_key[reflect_lower_square_key(pmesh.node_keys[edge[1]])],
                node_for_key[reflect_lower_square_key(pmesh.node_keys[edge[2]])],
            )
            findfirst(b -> edge_key(local_cr_edges(triangles[reflected_t])[b]...) == reflected_edge, 1:3)
        end

        for a in 1:3, b in 1:3
            @test isapprox(
                Float64(interval_midpoint(K[a, b])),
                Float64(interval_midpoint(reflected_K[edge_perm[a], edge_perm[b]]));
                rtol = 1e-9,
                atol = 1e-9,
            )
        end
    end
end

@testset "D6-invariant sector assembly bookkeeping" begin
    pmesh = pdelta_mesh_data()
    nodes, triangles = pmesh.nodes, pmesh.triangles
    bounds = [
        (;
            xx = interval(BigFloat(2)),
            xy = interval(BigFloat(1)) / interval(BigFloat(5)),
            yx = interval(BigFloat(1)) / interval(BigFloat(5)),
            yy = interval(BigFloat(3)),
        )
        for _ in eachindex(triangles)
    ]

    assembly = assemble_d6_invariant_cr_matrices(nodes, triangles, bounds)
    edges, _ = build_cr_edges(triangles)
    quotient = d6_sector_edge_quotient(edges, pmesh.node_keys)
    sums = edge_key_sums(edges, pmesh.node_keys)
    edge_id = Dict(sums[i] => i for i in eachindex(sums))

    @test size(nodes, 1) == 120
    @test length(triangles) == 196
    @test length(edges) == 315
    @test assembly.matrix_size == 161
    @test length(assembly.mass) == 161
    @test length(assembly.stiffness) == 735

    for i in eachindex(sums)
        reflected = sector_reflect_key(sums[i])
        if haskey(edge_id, reflected)
            @test quotient[i] == quotient[edge_id[reflected]]
        end

        if sums[i][2] == -sums[i][1]
            rotated = rotkey(sums[i], 1)
            if haskey(edge_id, rotated)
                @test quotient[i] == quotient[edge_id[rotated]]
            end
        end
    end
end
