using Test

include(joinpath(@__DIR__, "..", "assemble_matrices.jl"))
include(joinpath(@__DIR__, "triangulation.jl"))

function mesh_edges(triangles)
    incidence = Dict{Tuple{Int,Int},Int}()
    for tri in triangles, edge in _improved_triangle_edges(tri)
        incidence[edge] = get(incidence, edge, 0) + 1
    end
    return incidence
end

include(joinpath(@__DIR__, "assemble_matrices.jl"))

@testset "2x2 lower-matrix certification and Liu transform" begin
    good = [
        interval(BigFloat(2), BigFloat("2.1")) interval(BigFloat("-0.1"), BigFloat("0.1"))
        interval(BigFloat("-0.1"), BigFloat("0.1")) interval(BigFloat(1), BigFloat("1.1"))
    ]
    bad = [
        interval(BigFloat(1)) interval(BigFloat(-2), BigFloat(2))
        interval(BigFloat(-2), BigFloat(2)) interval(BigFloat(1))
    ]
    good_certificate = improved_spd_certificate_2x2(good)
    @test good_certificate.spd
    @test good_certificate.lambda_lower > 0
    @test !improved_spd_certificate_2x2(bad).spd

    liu = improved_liu_lower_bound(BigFloat(6), BigFloat("0.2"))
    @test liu <= BigFloat(6) / (BigFloat(1) + BigFloat("0.2")^2 * BigFloat(6))
    @test liu > BigFloat("4.83")
end

@testset "pointwise metric lower matrix on an interior cell" begin
    mesh = improved_pdelta_mesh_data(N = 14, refinement_level = 0)
    oracle = InverseMetricBoxOracleV2()
    certificate = improved_inverse_metric_lower_bound_for_triangle(
        oracle,
        mesh.nodes,
        mesh.triangles[1];
        deg = 8,
        terms = 8,
        bisection_iterations = 60,
    )
    @test 0 < certificate.theta <= 1
    @test certificate.B[1, 2] == certificate.B[2, 1]
    @test certificate.alpha > 0
    @test certificate.spd_margin > 0
    @test improved_spd_certificate_2x2(
        certificate.G - _improved_point_matrix_intervals(certificate.B),
    ).spd

    diameter = improved_triangle_diameter_upper(mesh.nodes, mesh.triangles[1])
    local_constant = improved_local_liu_constant_upper(diameter, certificate.alpha)
    @test diameter > 0
    @test local_constant > 0

    stiffness = improved_local_cr_stiffness_matrix(
        mesh.nodes,
        mesh.triangles[1],
        certificate.B,
    )
    @test size(stiffness) == (3, 3)
    @test all(!isempty_interval(stiffness[i, j]) for i in 1:3, j in 1:3)

    graded = improved_pdelta_mesh_data(
        delta = 1 // 100,
        N = 8,
        refinement_level = 3,
        boundary_layers = 1,
    )
    difficult_triangle = graded.triangles[436]
    @test_throws ErrorException improved_inverse_metric_lower_bound_for_triangle(
        oracle,
        graded.nodes,
        difficult_triangle;
        deg = 8,
        terms = 8,
        bisection_iterations = 50,
        max_subdivision_depth = 0,
    )
    adaptive = improved_inverse_metric_lower_bound_for_triangle(
        oracle,
        graded.nodes,
        difficult_triangle;
        deg = 8,
        terms = 8,
        bisection_iterations = 50,
        max_subdivision_depth = 1,
    )
    @test adaptive.diagnostics.cover_used
    @test adaptive.diagnostics.leaf_count == 4
    @test adaptive.diagnostics.max_leaf_depth == 1
    @test adaptive.alpha > 0
    @test adaptive.spd_margin > 0

    forced = improved_inverse_metric_lower_bound_for_triangle(
        oracle,
        mesh.nodes,
        mesh.triangles[1];
        deg = 8,
        terms = 8,
        bisection_iterations = 50,
        max_subdivision_depth = 1,
        minimum_subdivision_depth = 1,
    )
    @test forced.diagnostics.cover_used
    @test forced.diagnostics.leaf_count == 4
    @test forced.diagnostics.max_leaf_depth == 1
    @test forced.diagnostics.polynomial_parameters.minimum_subdivision_depth == 1

    quality_mesh = improved_pdelta_mesh_data(
        delta = 1 // 200,
        N = 8,
        refinement_level = 3,
        boundary_layers = 1,
        corner_refinement_level = 2,
        corner_layers = 1,
    )
    quality_singleton = (;
        nodes = quality_mesh.nodes,
        triangles = [quality_mesh.triangles[1425]],
    )
    quality_certificates = improved_element_certificates(
        quality_singleton;
        deg = 8,
        terms = 8,
        bisection_iterations = 50,
        max_subdivision_depth = 1,
        quality_liu_target = 3 // 20,
        max_quality_subdivision_depth = 1,
        progress_every = 0,
    )
    @test only(quality_certificates).quality_minimum_subdivision_depth == 1
    @test only(quality_certificates).liu_constant_upper <= BigFloat(3 // 20)
    @test only(quality_certificates).diagnostics.leaf_count == 4
end

@testset "global improved assembly accepts rounded zero entries" begin
    mesh = improved_pdelta_mesh_data(
        delta = 1 // 10,
        N = 4,
        refinement_level = 1,
        boundary_layers = 1,
        corner_refinement_level = 1,
        corner_layers = 1,
    )
    function is_right_triangle(tri)
        points = mesh.node_keys[collect(tri)]
        for vertex in 1:3
            other = filter(!=(vertex), 1:3)
            u = points[other[1]] .- points[vertex]
            v = points[other[2]] .- points[vertex]
            dot(u, v) == 0 && return true
        end
        return false
    end
    @test any(!is_right_triangle(tri) for tri in mesh.triangles)

    identity_B = BigFloat[1 0; 0 1]
    diameter = maximum(
        improved_triangle_diameter_upper(mesh.nodes, tri)
        for tri in mesh.triangles
    )
    local_constant = improved_local_liu_constant_upper(diameter, BigFloat(1))
    certificates = [
        (; B = copy(identity_B), liu_constant_upper = local_constant)
        for _ in mesh.triangles
    ]
    assembly = assemble_improved_d6_invariant_cr_matrices(mesh, certificates)

    @test assembly.matrix_size > 1
    @test !isempty(assembly.mass)
    @test !isempty(assembly.stiffness)
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

    mktempdir() do directory
        write_matlab_matrices(assembly; dir = directory)
        @test filesize(joinpath(directory, "stiff_matrix.mat")) > 0
        @test filesize(joinpath(directory, "mass_matrix.mat")) > 0
    end
end

include(joinpath(@__DIR__, "orchestrate.jl"))

@testset "orchestration guards and directed result serialization" begin
    @test_throws ErrorException improved_parse_options(["--delta=3/7"])
    @test_throws ErrorException improved_parse_options(["--delta=43/100"])
    @test_throws ErrorException improved_parse_options(["--tag=../escape"])
    @test_throws ErrorException improved_parse_options(["--corner-refinement-levle=3"])
    @test_throws ErrorException improved_parse_options(["--unknown-flag"])
    @test_throws ErrorException improved_parse_options(["--delta=1/10", "--delta=1/20"])
    @test_throws ErrorException improved_parse_options(["--parallel", "--parallel"])
    @test_throws ErrorException improved_parse_options(["--N=4", "--boundary-layers=5"])
    @test_throws ErrorException improved_parse_options(["--N=4", "--corner-layers=5"])
    @test_throws ErrorException improved_parse_options(["--quality-liu-target=3/20"])
    @test_throws ErrorException improved_compact_lower_bound(BigFloat(1), 43 // 100)

    options_60 = improved_parse_options(["--bisection-iterations=60"])
    options_80 = improved_parse_options(["--bisection-iterations=80"])
    @test improved_run_slug(options_60) != improved_run_slug(options_80)

    corner_options = improved_parse_options([
        "--N=8",
        "--refinement-level=3",
        "--corner-refinement-level=2",
        "--corner-layers=2",
        "--box-subdivision-depth=3",
        "--parallel",
        "--quality-liu-target=3/20",
        "--quality-subdivision-depth=2",
    ])
    @test corner_options.corner_refinement_level == 2
    @test corner_options.corner_layers == 2
    @test improved_run_slug(corner_options) != improved_run_slug(improved_parse_options(String[]))
    @test occursin("_corner2_clayers2_", improved_run_slug(corner_options))
    corner_config = improved_config_dictionary(corner_options)
    @test corner_config["corner_refinement_level"] == 2
    @test corner_config["corner_layers"] == 2
    assembly_options = improved_assembly_options(corner_options)
    @test assembly_options.corner_refinement_level == 2
    @test assembly_options.corner_layers == 2
    @test assembly_options.refinement_level == 3
    @test assembly_options.max_subdivision_depth == 3
    @test assembly_options.parallel
    @test assembly_options.quality_liu_target == 3 // 20
    @test assembly_options.max_quality_subdivision_depth == 2
    @test occursin("_qtarget3_20_qdepth2", improved_run_slug(corner_options))

    value = BigFloat("0.516807414131973603069171907824031294")
    lower_text = improved_lower_decimal(value)
    upper_text = improved_upper_decimal(value)
    parsed_lower_upper = setrounding(BigFloat, RoundUp) do
        parse(BigFloat, lower_text)
    end
    parsed_upper_lower = setrounding(BigFloat, RoundDown) do
        parse(BigFloat, upper_text)
    end
    @test parsed_lower_upper <= value
    @test parsed_upper_lower >= value
end

function has_strictly_interior_node(node_key_set, node_keys, edge)
    a, b = node_keys[edge[1]], node_keys[edge[2]]
    dx, dy = b[1] - a[1], b[2] - a[2]
    lattice_length = gcd(abs(dx), abs(dy))
    lattice_length <= 1 && return false
    step = (dx ÷ lattice_length, dy ÷ lattice_length)
    for offset in 1:(lattice_length - 1)
        candidate = (a[1] + offset * step[1], a[2] + offset * step[2])
        candidate in node_key_set && return true
    end
    return false
end

function maximum_squared_edge_length(node_keys, triangles)
    maximum_length = 0
    for tri in triangles, edge in _improved_triangle_edges(tri)
        a, b = node_keys[edge[1]], node_keys[edge[2]]
        squared_length = (a[1] - b[1])^2 + (a[2] - b[2])^2
        maximum_length = max(maximum_length, squared_length)
    end
    return maximum_length
end

"""
Squared scale-free triangle quality

    q^2 = 12 (2 area)^2 / (a^2 + b^2 + c^2)^2.

It lies in `(0, 1]`, equals `3/4` for a right-isosceles triangle, and avoids
floating-point trigonometry in the shape-regularity test.
"""
function squared_shape_quality(node_keys, tri)
    a, b, c = (node_keys[tri[1]], node_keys[tri[2]], node_keys[tri[3]])
    squared_length(p, q) = (p[1] - q[1])^2 + (p[2] - q[2])^2
    squared_length_sum =
        squared_length(a, b) + squared_length(b, c) + squared_length(c, a)
    double_area = abs(_improved_double_area(node_keys, tri))
    return 12 * double_area^2 // squared_length_sum^2
end

function reflected_triangle_key(node_keys, tri)
    reflected = [sector_reflect_key(node_keys[vertex]) for vertex in tri]
    return Tuple(sort!(reflected))
end

function triangle_key(node_keys, tri)
    return Tuple(sort!([node_keys[vertex] for vertex in tri]))
end

@testset "level-zero mesh exactly reproduces the original" begin
    for (delta, N) in ((0 // 1, 1), (1 // 137, 7), (1 // 20000, 14))
        original = pdelta_mesh_data(; delta, N)
        for boundary_layers in unique((1, min(3, N), N))
            improved = improved_pdelta_mesh_data(;
                delta,
                N,
                refinement_level = 0,
                boundary_layers,
            )

            @test improved.h == original.h
            @test improved.node_keys == original.node_keys
            @test improved.triangles == original.triangles
            @test inf.(improved.nodes) == inf.(original.nodes)
            @test sup.(improved.nodes) == sup.(original.nodes)
        end
    end

    @test_throws ArgumentError improved_pdelta_mesh_data(N = 4, boundary_layers = 0)
    @test_throws ArgumentError improved_pdelta_mesh_data(N = 4, boundary_layers = 5)
    @test_throws ArgumentError improved_pdelta_mesh_data(N = 4, corner_refinement_level = -1)
    @test_throws ArgumentError improved_pdelta_mesh_data(N = 4, corner_layers = 0)
    @test_throws ArgumentError improved_pdelta_mesh_data(N = 4, corner_layers = 5)
end

@testset "boundary refinement geometry and conformity" begin
    delta = 1 // 20000
    N = 14
    expected_counts = Dict(
        1 => [
            (120, 196),
            (175, 290),
            (417, 740),
            (1386, 2608),
            (5224, 10142),
        ],
        3 => [
            (120, 196),
            (248, 432),
            (772, 1438),
            (2855, 5518),
            (11127, 21888),
        ],
    )

    for boundary_layers in (1, 3), refinement_level in 0:4
        mesh = improved_pdelta_mesh_data(;
            delta,
            N,
            refinement_level,
            boundary_layers,
        )
        scale = 2^refinement_level
        outer_key = N * scale
        incidence = mesh_edges(mesh.triangles)
        node_key_set = Set(mesh.node_keys)

        @test length(node_key_set) == length(mesh.node_keys)
        @test length(Set(Tuple(sort(collect(tri))) for tri in mesh.triangles)) ==
              length(mesh.triangles)
        @test all(
            all(vertex -> 1 <= vertex <= length(mesh.node_keys), tri)
            for tri in mesh.triangles
        )
        @test all(_improved_double_area(mesh.node_keys, tri) > 0 for tri in mesh.triangles)
        @test maximum(values(incidence)) <= 2
        @test all(
            !has_strictly_interior_node(node_key_set, mesh.node_keys, edge)
            for edge in keys(incidence)
        )

        # The key lattice is scaled by 2^refinement_level.  Exact area
        # conservation therefore says that the summed doubled key-area is the
        # square of the outer key coordinate.
        summed_double_area = sum(
            _improved_double_area(mesh.node_keys, tri)
            for tri in mesh.triangles
        )
        @test summed_double_area == outer_key^2
        physical_double_area = summed_double_area * mesh.h^2 / scale^2
        @test physical_double_area == (1 - delta)^2

        # Euler characteristic of a triangulated closed disk.
        @test length(mesh.node_keys) - length(incidence) + length(mesh.triangles) == 1

        physical_edges = [
            edge for edge in keys(incidence) if
            mesh.node_keys[edge[1]][1] == outer_key &&
            mesh.node_keys[edge[2]][1] == outer_key
        ]
        @test length(physical_edges) == N * scale
        @test all(incidence[edge] == 1 for edge in physical_edges)
        physical_boundary_y = sort!(unique!(collect(Iterators.flatten(
            (
                (mesh.node_keys[edge[1]][2], mesh.node_keys[edge[2]][2])
                for edge in physical_edges
            ),
        ))))
        @test physical_boundary_y == collect(-outer_key:0)

        qualities = [squared_shape_quality(mesh.node_keys, tri) for tri in mesh.triangles]
        @test minimum(qualities) >= 12 // 49
        @test maximum(qualities) <= 1

        core_inner_x = (N - boundary_layers) * scale
        core_triangles = filter(
            tri -> all(vertex -> mesh.node_keys[vertex][1] >= core_inner_x, tri),
            mesh.triangles,
        )
        @test !isempty(core_triangles)
        @test maximum_squared_edge_length(mesh.node_keys, core_triangles) <= 2

        @test (length(mesh.node_keys), length(mesh.triangles)) ==
              expected_counts[boundary_layers][refinement_level + 1]

        if boundary_layers == 1
            default_mesh = improved_pdelta_mesh_data(; delta, N, refinement_level)
            @test default_mesh.node_keys == mesh.node_keys
            @test default_mesh.triangles == mesh.triangles
            @test inf.(default_mesh.nodes) == inf.(mesh.nodes)
            @test sup.(default_mesh.nodes) == sup.(mesh.nodes)
        end
    end
end

function in_corner_patch(node_keys, tri, N, scale, layers)
    inner_x = (N - layers) * scale
    transverse = layers * scale
    in_outer_band = all(vertex -> node_keys[vertex][1] >= inner_x, tri)
    in_top = all(vertex -> -node_keys[vertex][2] <= transverse, tri)
    in_bottom = all(vertex -> sum(node_keys[vertex]) <= transverse, tri)
    return in_outer_band && (in_top || in_bottom)
end

@testset "symmetric corner refinement" begin
    delta = 1 // 20000
    N = 14
    boundary_level = 1
    boundary_layers = 3
    corner_layers = 2
    expected_counts = [
        (248, 432),
        (360, 640),
        (920, 1720),
        (3230, 6254),
        (12497, 24610),
    ]

    baseline = improved_pdelta_mesh_data(;
        delta,
        N,
        refinement_level = boundary_level,
        boundary_layers,
    )
    for inactive_corner_layers in (1, 2, 5, N)
        inactive = improved_pdelta_mesh_data(;
            delta,
            N,
            refinement_level = boundary_level,
            boundary_layers,
            corner_refinement_level = 0,
            corner_layers = inactive_corner_layers,
        )
        @test inactive.node_keys == baseline.node_keys
        @test inactive.triangles == baseline.triangles
        @test inf.(inactive.nodes) == inf.(baseline.nodes)
        @test sup.(inactive.nodes) == sup.(baseline.nodes)
    end

    previous_corner_size = nothing
    for corner_level in 0:4
        mesh = improved_pdelta_mesh_data(;
            delta,
            N,
            refinement_level = boundary_level,
            boundary_layers,
            corner_refinement_level = corner_level,
            corner_layers,
        )
        scale = 2^(boundary_level + corner_level)
        outer_key = N * scale
        incidence = mesh_edges(mesh.triangles)
        node_key_set = Set(mesh.node_keys)

        @test (length(mesh.node_keys), length(mesh.triangles)) ==
              expected_counts[corner_level + 1]
        @test length(node_key_set) == length(mesh.node_keys)
        @test all(_improved_double_area(mesh.node_keys, tri) > 0 for tri in mesh.triangles)
        @test maximum(values(incidence)) <= 2
        @test all(
            !has_strictly_interior_node(node_key_set, mesh.node_keys, edge)
            for edge in keys(incidence)
        )
        @test length(mesh.node_keys) - length(incidence) + length(mesh.triangles) == 1

        summed_double_area = sum(
            _improved_double_area(mesh.node_keys, tri)
            for tri in mesh.triangles
        )
        @test summed_double_area == outer_key^2
        @test summed_double_area * mesh.h^2 / scale^2 == (1 - delta)^2

        qualities = [squared_shape_quality(mesh.node_keys, tri) for tri in mesh.triangles]
        # The interaction of a red corner patch with an existing green
        # boundary transition introduces one additional, but fixed, template.
        @test minimum(qualities) >= 48 // 841
        @test maximum(qualities) <= 1

        corner_core = filter(
            tri -> in_corner_patch(mesh.node_keys, tri, N, scale, corner_layers),
            mesh.triangles,
        )
        @test !isempty(corner_core)
        corner_size = maximum_squared_edge_length(mesh.node_keys, corner_core) // scale^2
        if !isnothing(previous_corner_size)
            @test corner_size == previous_corner_size / 4
        end
        previous_corner_size = corner_size

        # Probe a central base-grid boundary cell, outside the largest graded
        # corner closure used above.  It retains exactly the boundary-stage
        # subdivision and is not refined by any corner pass.
        middle_layer = N ÷ 2 - 1
        middle_edges = [
            edge for edge in keys(incidence) if
            all(
                vertex -> begin
                    x, y = mesh.node_keys[vertex]
                    x == outer_key &&
                        -(middle_layer + 1) * scale <= y <= -middle_layer * scale
                end,
                edge,
            )
        ]
        @test length(middle_edges) == 2^boundary_level
        @test all(
            abs(mesh.node_keys[edge[1]][2] - mesh.node_keys[edge[2]][2]) ==
            2^corner_level
            for edge in middle_edges
        )

        # The two transverse corner coordinates are exactly exchanged by the
        # sector reflection.
        @test all(
            begin
                reflected = sector_reflect_key(key)
                -reflected[2] == sum(key) && sum(reflected) == -key[2]
            end
            for key in mesh.node_keys
        )
        @test Set(sector_reflect_key(key) for key in mesh.node_keys) == node_key_set
        triangle_keys = Set(triangle_key(mesh.node_keys, tri) for tri in mesh.triangles)
        @test all(
            reflected_triangle_key(mesh.node_keys, tri) in triangle_keys
            for tri in mesh.triangles
        )

        edges, _ = build_cr_edges(mesh.triangles)
        midpoint_keys = edge_key_sums(edges, mesh.node_keys)
        midpoint_to_edge = Dict(key => edge for (edge, key) in pairs(midpoint_keys))
        @test length(midpoint_to_edge) == length(edges)
        @test all(haskey(midpoint_to_edge, sector_reflect_key(key)) for key in midpoint_keys)
        diagonal_keys = filter(key -> key[2] == -key[1], midpoint_keys)
        @test all(haskey(midpoint_to_edge, rotkey(key, 1)) for key in diagonal_keys)

        quotient = d6_sector_edge_quotient(edges, mesh.node_keys)
        for (edge_index, key) in pairs(midpoint_keys)
            @test quotient[edge_index] == quotient[midpoint_to_edge[sector_reflect_key(key)]]
            if key[2] == -key[1]
                @test quotient[edge_index] == quotient[midpoint_to_edge[rotkey(key, 1)]]
            end
        end
    end
end

@testset "D6 quotient key compatibility" begin
    N = 14
    for boundary_layers in (1, 3), refinement_level in 0:4
        mesh = improved_pdelta_mesh_data(; N, refinement_level, boundary_layers)
        reflected_nodes = Set(sector_reflect_key(key) for key in mesh.node_keys)
        @test reflected_nodes == Set(mesh.node_keys)

        triangle_keys = Set(triangle_key(mesh.node_keys, tri) for tri in mesh.triangles)
        @test all(
            reflected_triangle_key(mesh.node_keys, tri) in triangle_keys
            for tri in mesh.triangles
        )

        edges, _ = build_cr_edges(mesh.triangles)
        midpoint_keys = edge_key_sums(edges, mesh.node_keys)
        midpoint_to_edge = Dict(key => edge for (edge, key) in pairs(midpoint_keys))
        @test length(midpoint_to_edge) == length(edges)
        @test all(haskey(midpoint_to_edge, sector_reflect_key(key)) for key in midpoint_keys)

        diagonal_keys = filter(key -> key[2] == -key[1], midpoint_keys)
        @test !isempty(diagonal_keys)
        @test all(haskey(midpoint_to_edge, rotkey(key, 1)) for key in diagonal_keys)

        quotient = d6_sector_edge_quotient(edges, mesh.node_keys)
        @test length(quotient) == length(edges)
        @test minimum(quotient) == 1
        @test maximum(quotient) <= length(edges)
        for (edge_index, key) in pairs(midpoint_keys)
            @test quotient[edge_index] == quotient[midpoint_to_edge[sector_reflect_key(key)]]
            if key[2] == -key[1]
                @test quotient[edge_index] == quotient[midpoint_to_edge[rotkey(key, 1)]]
            end
        end
    end
end
