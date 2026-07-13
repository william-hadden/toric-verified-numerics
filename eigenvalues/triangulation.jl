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

"""
Construct the uniform base mesh of the lower fundamental sector.

`delta` is the exact rational inset parameter, so the sector has vertices
`(0, 0)`, `(1 - delta, 0)`, and `(1 - delta, -(1 - delta))`. `N` is the
positive number of base-grid intervals along each edge. Return integer lattice
`node_keys`, counterclockwise `triangles`. The physical coordinates for an
integer lattice point k=(k1,k2) is p=(h/scale)*k.
"""
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

"""
Return all edges of triangles wholly inside a physical-boundary band.

For triangulations of the fundamental sector, these are some triangles on
the right side of the triangulation. x=N is the right end of the triangulation.
(`node_keys`, `triangles`) is the triangulation to apply this to.
`inner_x` is the inclusive inner edge of the band in the same
lattice units. The returned set contains the canonical vertex-index pair for
every edge of each triangle whose vertices all satisfy `x >= inner_x`.
E.g. inner_x=N gives all triangles for which all vertices have x>=N, of which
there are none.
"""
function boundary_patch_edges(node_keys, triangles, inner_x)
    marked = Set{Tuple{Int,Int}}()
    for tri in triangles
        all(vertex -> node_keys[vertex][1] >= inner_x, tri) || continue
        union!(marked, triangle_edges(tri))
    end
    isempty(marked) && error("The boundary refinement patch is empty")
    return marked
end

"""
Return all edges of triangles wholly inside either boundary-corner patch.

`node_keys` gives integer lattice coordinates, `triangles` gives vertex-index
triples, and `inner_x` is the inclusive inner edge of the boundary band.
`transverse_width`, in the same lattice units, bounds `-y` near the upper
corner and `x + y` near the lower corner. The result is a set of canonical
vertex-index pairs.
"""
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

"""Promote every two-edge marking to three, leaving only green or red templates."""
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
        haskey(existing, key) && error("Marked edge $edge already has a midpoint vertex")
        push!(node_keys, key)
        midpoint_ids[edge] = length(node_keys)
        existing[key] = length(node_keys)
    end
    return midpoint_ids
end

"""
Apply standard green refinement across one split edge.

Join the edge midpoint to the opposite vertex, bisecting `tri` into two
counterclockwise children.
"""
function green_children(node_keys, tri, edge, midpoint)
    u, v = edge
    opposite = only(vertex for vertex in tri if vertex != u && vertex != v)
    return (
        positive_triangle(node_keys, (u, midpoint, opposite)),
        positive_triangle(node_keys, (midpoint, v, opposite)),
    )
end

"""
Apply standard red refinement to a triangle.

Bisect all three edges and join their midpoints, producing four
counterclockwise children.
"""
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

"""
Apply the standard red-green templates selected by a closed edge marking.

A triangle with one marked edge receives green refinement into two children
by bisecting along the marked edge;
a triangle with all three edges marked receives red refinement into four.
(Red/green is standard FEM terminology, see Larson-Bengzon: "The Finite Element
Method: Theory, Implementation, and Applications", Figure 4.6 on p.102.)
"""
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

"""
Apply one red-green refinement pass next to the physical boundary.

`node_keys` and `triangles` describe the current mesh, while `scale` is its
integer-lattice denominator: a key `p` represents `h * p / scale`, where `h`
is the base spacing returned by `base_sector_mesh`. `N` is the base resolution,
`layers` is the boundary-band width on the first pass in base-grid layers, and
`pass` is the one-based pass number. This pass marks triangles wholly inside a
band of `min(N, layers + pass - 1)` layers; red-green closure may also bisect
adjacent triangles to preserve conformity. Return the updated
`(node_keys, triangles, scale)`.
"""
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

"""
Apply one red-green refinement pass in the two physical corner patches.

`node_keys` and `triangles` describe the current mesh. A key `p` represents
`h * p / scale`, where `h` is the base spacing, `scale` is the current dyadic
denominator, and `N` is the base resolution. `layers` is each patch's width on
the first pass in base-grid layers, and `pass` is the one-based pass number.
For `k = min(N, layers + pass - 1)`, marked triangles lie inside
`x >= (N - k)h` and either `-y <= kh` or `x + y <= kh`; red-green closure may
also bisect neighboring triangles. Return the updated
`(node_keys, triangles, scale)`.
"""
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

"""
Enclose the exact physical coordinates in `BigFloat` intervals.

`node_keys` contains integer coordinate pairs, `h` is the exact physical
spacing of the base lattice, and `scale` is the current dyadic lattice
denominator. Row `i` of the returned matrix encloses the exact point
`h * node_keys[i] / scale` at the active `BigFloat` precision.
"""
function interval_nodes(node_keys, h, scale)
    nodes = Matrix{Interval{BigFloat}}(undef, length(node_keys), 2)
    for (index, key) in pairs(node_keys)
        x = key[1] * h / scale
        y = key[2] * h / scale
        nodes[index, 1] = interval(BigFloat, x)
        nodes[index, 2] = interval(BigFloat, y)
    end
    return nodes
end

"""
Construct the conforming graded mesh of the lower fundamental sector.

`delta` and `N` define the uniform base mesh. `boundary_refinements` and
`corner_refinements` are the numbers of graded refinement passes, while
`boundary_layers` and `corner_layers` give the corresponding patch widths on
the first pass. Return interval coordinates `nodes`, connectivity `triangles`,
exact integer `node_keys`, base spacing `h`, and dyadic lattice `scale`.
"""
function sector_mesh(
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
