using IntervalArithmetic

"""Return the base-grid index of the lattice point `(i, -j)`."""
lattice_index(i::Integer, j::Integer) = i * (i + 1) ÷ 2 + j + 1

"""Return the signed double area of a triangle in integer lattice coordinates."""
function lattice_double_area(node_keys, tri)
    x1, y1 = node_keys[tri[1]]
    x2, y2 = node_keys[tri[2]]
    x3, y3 = node_keys[tri[3]]
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
end

"""Orient a nondegenerate lattice triangle counterclockwise."""
function positive_triangle(node_keys, tri)
    area = lattice_double_area(node_keys, tri)
    iszero(area) && error("Degenerate triangle $tri")
    return area > 0 ? tri : (tri[3], tri[2], tri[1])
end

"""Return the three canonical undirected edges of a triangle."""
function triangle_edges(tri)
    return (
        minmax(tri[1], tri[2]),
        minmax(tri[2], tri[3]),
        minmax(tri[3], tri[1]),
    )
end

"""Construct the unrefined triangular lattice on the fundamental sector."""
function base_sector_mesh(delta, N::Integer)
    h = (1 - delta) // N
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
    return (; node_keys, triangles, h, scale = 1)
end

"""Mark every edge of triangles contained in the requested boundary band."""
function boundary_patch_edges(node_keys, triangles, inner_x)
    marked = Set{Tuple{Int,Int}}()
    for tri in triangles
        all(vertex -> node_keys[vertex][1] >= inner_x, tri) || continue
        union!(marked, triangle_edges(tri))
    end
    isempty(marked) && error("The boundary refinement patch is empty")
    return marked
end

"""Mark every edge of triangles contained in either physical corner patch."""
function corner_patch_edges(node_keys, triangles, inner_x, transverse_width)
    marked = Set{Tuple{Int,Int}}()
    for tri in triangles
        all(vertex -> node_keys[vertex][1] >= inner_x, tri) || continue
        top = all(vertex -> -node_keys[vertex][2] <= transverse_width, tri)
        bottom = all(vertex -> sum(node_keys[vertex]) <= transverse_width, tri)
        (top || bottom) || continue
        union!(marked, triangle_edges(tri))
    end
    isempty(marked) && error("The corner refinement patches are empty")
    return marked
end

"""Complete an edge marking so no triangle has exactly two marked edges."""
function close_edge_marking!(triangles, marked)
    changed = true
    while changed
        changed = false
        for tri in triangles
            edges = triangle_edges(tri)
            count(edge -> edge in marked, edges) == 2 || continue
            old_length = length(marked)
            union!(marked, edges)
            changed |= length(marked) > old_length
        end
    end
    return marked
end

"""Insert the midpoint of every marked edge on the common dyadic lattice."""
function insert_midpoints!(node_keys, marked)
    existing = Dict(key => index for (index, key) in pairs(node_keys))
    midpoint_ids = Dict{Tuple{Int,Int},Int}()
    for edge in sort!(collect(marked))
        u, v = edge
        sums = node_keys[u] .+ node_keys[v]
        all(iseven, sums) || error("Marked edge $edge is not dyadically divisible")
        key = (sums[1] ÷ 2, sums[2] ÷ 2)
        haskey(existing, key) && error("Marked edge $edge already has a midpoint vertex")
        push!(node_keys, key)
        midpoint_ids[edge] = length(node_keys)
        existing[key] = length(node_keys)
    end
    return midpoint_ids
end

"""Return the two green-refinement children associated with one split edge."""
function green_children(node_keys, tri, edge, midpoint)
    u, v = edge
    opposite = only(vertex for vertex in tri if vertex != u && vertex != v)
    return (
        positive_triangle(node_keys, (u, midpoint, opposite)),
        positive_triangle(node_keys, (midpoint, v, opposite)),
    )
end

"""Return the four red-refinement children of a triangle."""
function red_children(node_keys, tri, midpoint_ids)
    a, b, c = tri
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

"""Apply the red-green templates selected by a closed edge marking."""
function refine_marked_triangles(node_keys, triangles, marked, midpoint_ids)
    refined = Tuple{Int,Int,Int}[]
    for tri in triangles
        edges = triangle_edges(tri)
        split = [edge for edge in edges if edge in marked]
        if isempty(split)
            push!(refined, tri)
        elseif length(split) == 1
            append!(refined, green_children(node_keys, tri, only(split), midpoint_ids[only(split)]))
        elseif length(split) == 3
            append!(refined, red_children(node_keys, tri, midpoint_ids))
        else
            error("Edge closure left a triangle with two marked edges")
        end
    end
    return refined
end

"""Refine one graded layer next to the physical boundary."""
function refine_boundary_once(node_keys, triangles, scale, N, layers, pass)
    node_keys = [(2 * x, 2 * y) for (x, y) in node_keys]
    scale *= 2
    closure_layers = min(N, layers + pass - 1)
    marked = boundary_patch_edges(node_keys, triangles, (N - closure_layers) * scale)
    close_edge_marking!(triangles, marked)
    midpoint_ids = insert_midpoints!(node_keys, marked)
    triangles = refine_marked_triangles(node_keys, triangles, marked, midpoint_ids)
    return node_keys, triangles, scale
end

"""Refine one graded layer in the two physical corner patches."""
function refine_corners_once(node_keys, triangles, scale, N, layers, pass)
    node_keys = [(2 * x, 2 * y) for (x, y) in node_keys]
    scale *= 2
    closure_layers = min(N, layers + pass - 1)
    width = closure_layers * scale
    marked = corner_patch_edges(node_keys, triangles, (N - closure_layers) * scale, width)
    close_edge_marking!(triangles, marked)
    midpoint_ids = insert_midpoints!(node_keys, marked)
    triangles = refine_marked_triangles(node_keys, triangles, marked, midpoint_ids)
    return node_keys, triangles, scale
end

"""Convert exact lattice keys to outward-rounded physical coordinates."""
function interval_nodes(node_keys, h, scale)
    nodes = Matrix{Interval{BigFloat}}(undef, length(node_keys), 2)
    for (index, key) in pairs(node_keys)
        x = key[1] * h / scale
        y = key[2] * h / scale
        nodes[index, 1] = interval(BigFloat(x, RoundDown), BigFloat(x, RoundUp))
        nodes[index, 2] = interval(BigFloat(y, RoundDown), BigFloat(y, RoundUp))
    end
    return nodes
end

"""Validate the exact parameters of a sector triangulation."""
function validate_mesh_parameters(
    delta,
    N,
    boundary_refinements,
    boundary_layers,
    corner_refinements,
    corner_layers,
)
    0 < delta < 1 || throw(ArgumentError("delta must lie in (0,1)"))
    N > 0 || throw(ArgumentError("N must be positive"))
    boundary_refinements >= 0 || throw(ArgumentError("boundary_refinements must be nonnegative"))
    corner_refinements >= 0 || throw(ArgumentError("corner_refinements must be nonnegative"))
    1 <= boundary_layers <= N || throw(ArgumentError("boundary_layers must lie in 1:N"))
    1 <= corner_layers <= N || throw(ArgumentError("corner_layers must lie in 1:N"))
end

"""Construct the conforming graded triangulation of the lower fundamental sector."""
function sector_mesh(
    delta,
    N,
    boundary_refinements,
    boundary_layers,
    corner_refinements,
    corner_layers,
)
    validate_mesh_parameters(
        delta,
        N,
        boundary_refinements,
        boundary_layers,
        corner_refinements,
        corner_layers,
    )
    mesh = base_sector_mesh(delta, N)
    node_keys, triangles, scale = mesh.node_keys, mesh.triangles, mesh.scale
    for pass in 1:boundary_refinements
        node_keys, triangles, scale = refine_boundary_once(
            node_keys,
            triangles,
            scale,
            N,
            boundary_layers,
            pass,
        )
    end
    for pass in 1:corner_refinements
        node_keys, triangles, scale = refine_corners_once(
            node_keys,
            triangles,
            scale,
            N,
            corner_layers,
            pass,
        )
    end
    nodes = interval_nodes(node_keys, mesh.h, scale)
    return (; nodes, triangles, node_keys, h = mesh.h, scale)
end
