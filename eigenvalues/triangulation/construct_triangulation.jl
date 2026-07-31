"""Return the base-grid index of the lattice point `(i, -j)`."""
lattice_index(i::Integer, j::Integer) = i * (i + 1) ÷ 2 + j + 1

"""Return the signed double area of a triangle in integer lattice coordinates."""
function lattice_double_area(node_keys, triangle)
    x1, y1 = node_keys[triangle[1]]
    x2, y2 = node_keys[triangle[2]]
    x3, y3 = node_keys[triangle[3]]
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
end

"""Orient a nondegenerate lattice triangle counterclockwise."""
function positive_triangle(node_keys, triangle)
    area = lattice_double_area(node_keys, triangle)
    iszero(area) && error("Degenerate triangle $triangle")
    return area > 0 ? triangle : (triangle[3], triangle[2], triangle[1])
end

"""Return the three canonical undirected edges of a triangle."""
function triangle_edges(triangle)
    return (
        minmax(triangle[1], triangle[2]),
        minmax(triangle[2], triangle[3]),
        minmax(triangle[3], triangle[1]),
    )
end

"""
Construct the uniform base topology of the lower fundamental sector.

`N` is the positive number of base-grid intervals along each edge. Return
integer lattice `node_keys`, counterclockwise `triangles`, and lattice `scale`.
Physical coordinates and the inset parameter do not affect this topology.
"""
function base_sector_mesh(N::Integer)
    node_keys = Vector{Tuple{Int,Int}}(undef, (N + 1) * (N + 2) ÷ 2)
    for i in 0:N, j in 0:i
        node_keys[lattice_index(i, j)] = (i, -j)
    end

    triangles = Tuple{Int,Int,Int}[]
    sizehint!(triangles, N^2)
    for i in 0:(N - 1)
        for j in 0:i
            push!(triangles, (
                lattice_index(i + 1, j + 1),
                lattice_index(i + 1, j),
                lattice_index(i, j),
            ))
        end
        for j in 0:(i - 1)
            push!(triangles, (
                lattice_index(i, j),
                lattice_index(i, j + 1),
                lattice_index(i + 1, j + 1),
            ))
        end
    end
    return (; node_keys, triangles, scale = 1)
end

"""Return all edges of triangles wholly inside a boundary band."""
function boundary_patch_edges(node_keys, triangles, inner_x)
    marked = Set{Tuple{Int,Int}}()
    for triangle in triangles
        all(vertex -> node_keys[vertex][1] >= inner_x, triangle) || continue
        union!(marked, triangle_edges(triangle))
    end
    isempty(marked) && error("The boundary refinement patch is empty")
    return marked
end

"""Return all edges of triangles wholly inside either boundary-corner patch."""
function corner_patch_edges(node_keys, triangles, inner_x, transverse_width)
    marked = Set{Tuple{Int,Int}}()
    for triangle in triangles
        all(vertex -> node_keys[vertex][1] >= inner_x, triangle) || continue
        top = all(vertex -> -node_keys[vertex][2] <= transverse_width, triangle)
        bottom = all(vertex -> sum(node_keys[vertex]) <= transverse_width, triangle)
        (top || bottom) || continue
        union!(marked, triangle_edges(triangle))
    end
    isempty(marked) && error("The corner refinement patches are empty")
    return marked
end

"""Promote every two-edge marking to three, leaving only green or red templates."""
function close_edge_marking!(triangles, marked)
    changed = true
    while changed
        changed = false
        for triangle in triangles
            edges = triangle_edges(triangle)
            count(edge -> edge in marked, edges) == 2 || continue
            old_length = length(marked)
            union!(marked, edges)
            changed |= length(marked) > old_length
        end
    end
    return marked
end

"""
Insert the midpoint of every marked edge on the current integer lattice.

Each refinement pass first doubles all `node_keys`, so every marked-edge
midpoint has integer coordinates. Return the map from marked edges to their
new midpoint indices.
"""
function insert_midpoints!(node_keys, marked)
    existing = Dict(key => index for (index, key) in pairs(node_keys))
    midpoint_ids = Dict{Tuple{Int,Int},Int}()
    for edge in sort!(collect(marked))
        u, v = edge
        sums = node_keys[u] .+ node_keys[v]
        key = (sums[1] ÷ 2, sums[2] ÷ 2)
        haskey(existing, key) &&
            error("Marked edge $edge already has a midpoint vertex")
        push!(node_keys, key)
        midpoint_ids[edge] = length(node_keys)
        existing[key] = length(node_keys)
    end
    return midpoint_ids
end

"""Bisect a triangle across its single marked edge."""
function green_children(node_keys, triangle, edge, midpoint)
    u, v = edge
    opposite = only(
        vertex for vertex in triangle if vertex != u && vertex != v
    )
    return (
        positive_triangle(node_keys, (u, midpoint, opposite)),
        positive_triangle(node_keys, (midpoint, v, opposite)),
    )
end

"""Bisect all three edges of a triangle to produce four children."""
function red_children(node_keys, triangle, midpoint_ids)
    a, b, c = triangle
    ab = midpoint_ids[minmax(a, b)]
    bc = midpoint_ids[minmax(b, c)]
    ca = midpoint_ids[minmax(c, a)]
    return (
        positive_triangle(node_keys, (a, ab, ca)),
        positive_triangle(node_keys, (ab, b, bc)),
        positive_triangle(node_keys, (ca, bc, c)),
        positive_triangle(node_keys, (ab, bc, ca)),
    )
end

"""Apply the standard red-green templates selected by a closed edge marking."""
function refine_marked_triangles(
    node_keys,
    triangles,
    marked,
    midpoint_ids,
)
    refined = Tuple{Int,Int,Int}[]
    for triangle in triangles
        edges = triangle_edges(triangle)
        split = [edge for edge in edges if edge in marked]
        if isempty(split)
            push!(refined, triangle)
        elseif length(split) == 1
            edge = only(split)
            append!(
                refined,
                green_children(
                    node_keys,
                    triangle,
                    edge,
                    midpoint_ids[edge],
                ),
            )
        elseif length(split) == 3
            append!(
                refined,
                red_children(node_keys, triangle, midpoint_ids),
            )
        else
            error("Edge closure left a triangle with two marked edges")
        end
    end
    return refined
end

"""Apply one red-green refinement pass next to the physical boundary."""
function refine_boundary_once(node_keys, triangles, scale, N, layers, pass)
    node_keys = [(2 * x, 2 * y) for (x, y) in node_keys]
    scale *= 2
    closure_layers = min(N, layers + pass - 1)
    marked = boundary_patch_edges(
        node_keys,
        triangles,
        (N - closure_layers) * scale,
    )
    close_edge_marking!(triangles, marked)
    midpoint_ids = insert_midpoints!(node_keys, marked)
    triangles = refine_marked_triangles(
        node_keys,
        triangles,
        marked,
        midpoint_ids,
    )
    return node_keys, triangles, scale
end

"""Apply one red-green refinement pass in the two physical corner patches."""
function refine_corners_once(node_keys, triangles, scale, N, layers, pass)
    node_keys = [(2 * x, 2 * y) for (x, y) in node_keys]
    scale *= 2
    closure_layers = min(N, layers + pass - 1)
    width = closure_layers * scale
    marked = corner_patch_edges(
        node_keys,
        triangles,
        (N - closure_layers) * scale,
        width,
    )
    close_edge_marking!(triangles, marked)
    midpoint_ids = insert_midpoints!(node_keys, marked)
    triangles = refine_marked_triangles(
        node_keys,
        triangles,
        marked,
        midpoint_ids,
    )
    return node_keys, triangles, scale
end

const SELECTIVE_GROUP_RANGES = (
    # Groups with C_h > 1/50 in balanced_n800_q037.
    (
        401_609:407_960,
        408_639:408_655,
        408_963:408_964,
        408_966:408_973,
        409_105:409_105,
        409_126:409_126,
        409_133:409_133,
    ),
    # Groups with C_h > 1/100 in balanced_n800_selective_q02.
    (
        318_809:318_811,
        421_387:434_104,
        434_434:434_471,
        434_482:434_491,
        434_628:434_631,
        434_651:434_651,
        434_656:434_656,
        434_661:434_662,
        434_665:434_666,
    ),
    # Groups with C_h > 21/2000 in balanced_n800_delta4_q021.
    (
        459_997:459_998,
        460_001:485_473,
        485_499:485_499,
        485_569:485_661,
        485_673:485_673,
        485_692:485_692,
        485_694:485_727,
        485_737:485_746,
        485_748:485_751,
        485_762:485_766,
        485_771:485_773,
        485_775:485_777,
        485_780:485_780,
        485_782:485_782,
        485_793:485_793,
    ),
)

"""
Return the metric-box key of `triangle` indexed into integer `node_keys`.

The key is `(xmin, xmax, ymin, ymax, diameter_squared)`.
"""
function metric_box_key(node_keys, triangle)
    keys = node_keys[collect(triangle)]
    xs, ys = first.(keys), last.(keys)
    diameter_squared = maximum(
        (keys[i][1] - keys[j][1])^2 + (keys[i][2] - keys[j][2])^2
        for i in 1:2 for j in (i + 1):3
    )
    return (minimum(xs), maximum(xs), minimum(ys), maximum(ys), diameter_squared)
end

"""
Group the triangles of `mesh` by metric-box key.

Return triangle-index vectors in lexicographic key order. The hardcoded
selective ranges index these groups, not individual triangles.
"""
function metric_box_groups(mesh)
    grouped = Dict{NTuple{5,Int},Vector{Int}}()
    for (index, triangle) in pairs(mesh.triangles)
        key = metric_box_key(mesh.node_keys, triangle)
        push!(get!(grouped, key, Int[]), index)
    end
    return [grouped[key] for key in sort!(collect(keys(grouped)))]
end

"""
Refine the metric-box groups selected by `ranges` in `mesh`.

`ranges` indexes the ordered result of `metric_box_groups`. Return
`node_keys`, `triangles`, and the doubled lattice `scale`.
"""
function refine_selected_groups(mesh, ranges)
    groups = metric_box_groups(mesh)
    marked = Set{Tuple{Int,Int}}()
    for group_index in Iterators.flatten(ranges)
        for triangle_index in groups[group_index]
            union!(marked, triangle_edges(mesh.triangles[triangle_index]))
        end
    end
    node_keys = [(2 * x, 2 * y) for (x, y) in mesh.node_keys]
    close_edge_marking!(mesh.triangles, marked)
    midpoint_ids = insert_midpoints!(node_keys, marked)
    triangles = refine_marked_triangles(
        node_keys, mesh.triangles, marked, midpoint_ids,
    )
    return (; node_keys, triangles, scale = 2 * mesh.scale)
end

"""Return the fixed lattice mesh as `node_keys`, `triangles`, and `scale`."""
function fixed_lattice_mesh()
    mesh = base_sector_mesh(800)
    node_keys, triangles, scale = mesh.node_keys, mesh.triangles, mesh.scale
    for pass in 1:3
        node_keys, triangles, scale = refine_boundary_once(
            node_keys, triangles, scale, 800, 1, pass,
        )
    end
    for pass in 1:3
        node_keys, triangles, scale = refine_corners_once(
            node_keys, triangles, scale, 800, 1, pass,
        )
    end
    mesh = (; node_keys, triangles, scale)
    for ranges in SELECTIVE_GROUP_RANGES
        mesh = refine_selected_groups(mesh, ranges)
    end
    return mesh
end

"""Write `matrix` under `name` to MAT-v4 stream `io`; return nothing."""
function write_mat4_matrix(io, name, matrix)
    rows, columns = size(matrix)
    write(io, Int32(0), Int32(rows), Int32(columns), Int32(0))
    write(io, Int32(ncodeunits(name) + 1), codeunits(name), UInt8(0))
    write(io, vec(Float64.(matrix)))
    return nothing
end

"""Write `mesh` to the MATLAB version-4 file at `path`; return nothing."""
function write_triangulation(path, mesh)
    node_keys = stack(mesh.node_keys)
    triangles = stack(mesh.triangles)
    lattice_size = maximum(first, mesh.node_keys)
    open(path, "w") do io
        write_mat4_matrix(io, "node_keys", node_keys)
        write_mat4_matrix(io, "triangles", triangles)
        write_mat4_matrix(io, "lattice_size", reshape([lattice_size], 1, 1))
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    mesh = fixed_lattice_mesh()
    write_triangulation(joinpath(@__DIR__, "triangulation.mat"), mesh)
    println(
        "wrote $(length(mesh.node_keys)) nodes and " *
        "$(length(mesh.triangles)) triangles",
    )
end
