using IntervalArithmetic

const IMPROVED_DEFAULT_DELTA = 1 // 20000
const IMPROVED_DEFAULT_N = 14

_improved_edge_key(i::Integer, j::Integer) = i < j ? (i, j) : (j, i)

function _improved_index(i::Integer, j::Integer)
    return i * (i + 1) ÷ 2 + j + 1
end

function _improved_double_area(node_keys, tri)
    x1, y1 = node_keys[tri[1]]
    x2, y2 = node_keys[tri[2]]
    x3, y3 = node_keys[tri[3]]
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
end

function _improved_right_angle_position(node_keys, tri)
    for position in 1:3
        q = node_keys[tri[position]]
        p = node_keys[tri[mod1(position - 1, 3)]]
        r = node_keys[tri[mod1(position + 1, 3)]]
        first_vector = (p[1] - q[1], p[2] - q[2])
        second_vector = (r[1] - q[1], r[2] - q[2])
        dot_product =
            first_vector[1] * second_vector[1] +
            first_vector[2] * second_vector[2]
        iszero(dot_product) && return position
    end
    return nothing
end

function _improved_positive_triangle(node_keys, tri)
    double_area = _improved_double_area(node_keys, tri)
    iszero(double_area) && error("Degenerate triangle $tri")
    return double_area > 0 ? tri : (tri[3], tri[2], tri[1])
end

function _improved_triangle_edges(tri)
    return (
        _improved_edge_key(tri[1], tri[2]),
        _improved_edge_key(tri[2], tri[3]),
        _improved_edge_key(tri[3], tri[1]),
    )
end

function _improved_boundary_refinement_edges(node_keys, triangles, band_inner_x)
    edges = Set{Tuple{Int,Int}}()
    for tri in triangles
        # Red-refine every current triangle contained in the fixed band of
        # base-grid cells next to the physical boundary.  The band interface
        # is itself a base-grid line, so no current triangle straddles it.
        all(vertex -> node_keys[vertex][1] >= band_inner_x, tri) || continue
        # Marking all three edges is affine-equivariant, and hence preserves
        # the non-Euclidean D6 lattice symmetries.
        union!(edges, _improved_triangle_edges(tri))
    end
    isempty(edges) && error("The triangulation has no triangle in the boundary band")
    return edges
end

"""
Complete an edge marking so every triangle has zero, one, or three marked
edges.  A triangle with two marked edges is promoted to a red refinement.

This removes the ambiguous two-edge green template while keeping the closure
local: a triangle with one marked edge is allowed to be a green transition.
"""
function _improved_red_green_closure!(triangles, marked_edges)
    changed = true
    while changed
        changed = false
        for tri in triangles
            edges = _improved_triangle_edges(tri)
            marked_count = count(edge -> edge in marked_edges, edges)
            marked_count == 2 || continue
            for edge in edges
                if !(edge in marked_edges)
                    push!(marked_edges, edge)
                    changed = true
                end
            end
        end
    end
    return marked_edges
end

function _improved_add_midpoints!(node_keys, marked_edges)
    existing_nodes = Dict(key => index for (index, key) in pairs(node_keys))
    midpoint_ids = Dict{Tuple{Int,Int},Int}()
    for edge in sort!(collect(marked_edges))
        u, v = edge
        sums = (
            node_keys[u][1] + node_keys[v][1],
            node_keys[u][2] + node_keys[v][2],
        )
        all(iseven, sums) || error("Marked edge $edge is not on the dyadic lattice")
        midpoint_key = (sums[1] ÷ 2, sums[2] ÷ 2)
        haskey(existing_nodes, midpoint_key) && error(
            "The midpoint of marked edge $edge is already a vertex; " *
            "the input mesh has a hanging node",
        )
        push!(node_keys, midpoint_key)
        midpoint_ids[edge] = length(node_keys)
        existing_nodes[midpoint_key] = length(node_keys)
    end
    return midpoint_ids
end

function _improved_refine_marked_triangles(node_keys, triangles, marked_edges, midpoint_ids)
    refined = Tuple{Int,Int,Int}[]
    for tri in triangles
        edges = _improved_triangle_edges(tri)
        marked = findall(edge -> edge in marked_edges, edges)
        if isempty(marked)
            push!(refined, tri)
        elseif length(marked) == 1
            edge = edges[only(marked)]
            u, v = edge
            midpoint = midpoint_ids[edge]
            opposite = only(vertex for vertex in tri if vertex != u && vertex != v)
            push!(
                refined,
                _improved_positive_triangle(node_keys, (u, midpoint, opposite)),
                _improved_positive_triangle(node_keys, (midpoint, v, opposite)),
            )
        elseif length(marked) == 3
            a, b, c = tri
            midpoint_ab = midpoint_ids[_improved_edge_key(a, b)]
            midpoint_bc = midpoint_ids[_improved_edge_key(b, c)]
            midpoint_ca = midpoint_ids[_improved_edge_key(c, a)]
            push!(
                refined,
                _improved_positive_triangle(node_keys, (a, midpoint_ab, midpoint_ca)),
                _improved_positive_triangle(node_keys, (midpoint_ab, b, midpoint_bc)),
                _improved_positive_triangle(node_keys, (midpoint_ca, midpoint_bc, c)),
                _improved_positive_triangle(
                    node_keys,
                    (midpoint_ab, midpoint_bc, midpoint_ca),
                ),
            )
        else
            error("Red-green closure left a triangle with two marked edges")
        end
    end
    return refined
end

function _improved_refine_boundary_once(
    node_keys,
    triangles,
    scale,
    N,
    boundary_layers,
    refinement_pass,
)
    # Move the whole current mesh to the next dyadic lattice before creating
    # midpoints.  This keeps all symmetry keys integral at every level.
    node_keys = [(2 * key[1], 2 * key[2]) for key in node_keys]
    scale *= 2
    # Repeatedly bisecting the same one-edge green transition would create
    # progressively thinner triangles.  Expand the closure by one base-grid
    # layer per pass, giving successive layers one fewer red generations while
    # refining the requested fixed core band on every pass.
    closure_layers = min(N, boundary_layers + refinement_pass - 1)
    band_inner_x = (N - closure_layers) * scale

    marked_edges = _improved_boundary_refinement_edges(
        node_keys,
        triangles,
        band_inner_x,
    )
    _improved_red_green_closure!(triangles, marked_edges)
    midpoint_ids = _improved_add_midpoints!(node_keys, marked_edges)
    triangles = _improved_refine_marked_triangles(
        node_keys,
        triangles,
        marked_edges,
        midpoint_ids,
    )
    return node_keys, triangles, scale
end

function _improved_corner_refinement_edges(
    node_keys,
    triangles,
    N,
    scale,
    closure_layers,
)
    outer_inner_x = (N - closure_layers) * scale
    transverse_outer = closure_layers * scale
    edges = Set{Tuple{Int,Int}}()

    for tri in triangles
        in_outer_band = all(
            vertex -> node_keys[vertex][1] >= outer_inner_x,
            tri,
        )
        in_outer_band || continue

        # These two predicates are exchanged by
        # sector_reflect_key((x,y)) = (x,-x-y).
        in_top_corner = all(
            vertex -> -node_keys[vertex][2] <= transverse_outer,
            tri,
        )
        in_bottom_corner = all(
            vertex -> sum(node_keys[vertex]) <= transverse_outer,
            tri,
        )
        (in_top_corner || in_bottom_corner) || continue
        union!(edges, _improved_triangle_edges(tri))
    end

    isempty(edges) && error("The triangulation has no triangle in the corner patches")
    return edges
end

function _improved_refine_corners_once(
    node_keys,
    triangles,
    scale,
    N,
    corner_layers,
    corner_pass,
)
    node_keys = [(2 * key[1], 2 * key[2]) for key in node_keys]
    scale *= 2

    # The requested corner patches are refined on every pass.  Growing by one
    # base-grid layer in both defining directions gives successive surrounding
    # layers one fewer generations and prevents repeated green-transition
    # bisections from creating slivers.
    closure_layers = min(N, corner_layers + corner_pass - 1)
    marked_edges = _improved_corner_refinement_edges(
        node_keys,
        triangles,
        N,
        scale,
        closure_layers,
    )
    _improved_red_green_closure!(triangles, marked_edges)
    midpoint_ids = _improved_add_midpoints!(node_keys, marked_edges)
    triangles = _improved_refine_marked_triangles(
        node_keys,
        triangles,
        marked_edges,
        midpoint_ids,
    )
    return node_keys, triangles, scale
end

function _improved_interval_nodes(node_keys, h, scale)
    enclose(x) = interval(BigFloat(x, RoundDown), BigFloat(x, RoundUp))
    nodes = Matrix{Interval{BigFloat}}(undef, length(node_keys), 2)
    for (index, key) in pairs(node_keys)
        nodes[index, 1] = enclose(key[1] * h / scale)
        nodes[index, 2] = enclose(key[2] * h / scale)
    end
    return nodes
end

"""
Return a conforming, boundary-refined triangulation of the lower fundamental
triangle of `P_delta`.

At `refinement_level = 0`, the node ordering, interval coordinates, lattice
keys, triangle ordering, and `h` value exactly reproduce
`assemble_matrices.jl:pdelta_mesh_data`.

For each positive refinement level, `boundary_layers` selects a fixed band of
the original base-grid cells,

    x >= (1 - delta) - boundary_layers * h.

Every current triangle inside that complete band is red-refined on every pass.
Neighbouring triangles with one split edge receive the usual two-child green
transition.  A triangle with two split edges is promoted to a red refinement,
so no ambiguous transition template occurs.  This rule is combinatorial and
therefore preserves the D6 lattice symmetries.  `boundary_layers = 1` agrees
with the previous first-pass physical-edge marking after its conformity
closure.  To prevent repeated green bisections from producing slivers, pass
`r` also red-refines a graded closure extending through `r - 1` additional
base-grid layers.  Thus adjacent layers differ by at most one red generation.

The red children remain right triangles.  A green transition across a leg of a
right triangle necessarily contains one non-right child; keeping every child
right would force another red refinement and ultimately global propagation.
Consequently, refined callers must use general affine-triangle integration and
must not rely on the old `tri[2]` right-angle convention.  Every returned
triangle is nevertheless positively oriented.

The `node_keys` for a level-`r` mesh are integer coordinates on the common
dyadic lattice.  With boundary level `r` and corner level `s`, their common
scale is `2^(r+s)`.  Multiplying all keys by this common scale does not alter
the existing D6 quotient operations.

After the boundary passes, `corner_refinement_level` additional passes may be
applied to the two patches where the physical facet meets a symmetry edge.  A
fixed corner core consists of the last `corner_layers` base layers in `x` and
either the first `corner_layers` layers in `-y` or in `x+y`:

    x >= (1 - delta) - corner_layers * h,
    -y <= corner_layers * h  or  x+y <= corner_layers * h.

The two transverse conditions are exchanged by `sector_reflect_key`.  Corner
pass `s` adds `s-1` base layers to both inequalities as a shape-regular graded
closure, while the fixed core is refined on every pass.

A tempting alternative is to apply midpoint 1-to-4 (red) refinement only to
the boundary triangles.  That splits all three of their edges, leaving hanging
midpoints in neighbouring coarse triangles.  Closing the mesh using red
refinement alone propagates across the connected triangulation and becomes a
global refinement.  The green transition used here is what keeps refinement
local and conforming.
"""
function improved_pdelta_mesh_data(;
    delta = IMPROVED_DEFAULT_DELTA,
    N::Integer = IMPROVED_DEFAULT_N,
    refinement_level::Integer = 0,
    boundary_layers::Integer = 1,
    corner_refinement_level::Integer = 0,
    corner_layers::Integer = 1,
)
    N > 0 || throw(ArgumentError("N must be positive"))
    refinement_level >= 0 || throw(ArgumentError("refinement_level must be nonnegative"))
    1 <= boundary_layers <= N || throw(ArgumentError("boundary_layers must lie in 1:N"))
    corner_refinement_level >= 0 ||
        throw(ArgumentError("corner_refinement_level must be nonnegative"))
    1 <= corner_layers <= N || throw(ArgumentError("corner_layers must lie in 1:N"))
    0 <= delta < 1 || throw(ArgumentError("delta must lie in [0, 1)"))

    h = (1 - delta) // N
    node_count = (N + 1) * (N + 2) ÷ 2
    node_keys = Vector{Tuple{Int,Int}}(undef, node_count)
    for i in 0:N, j in 0:i
        node_keys[_improved_index(i, j)] = (i, -j)
    end

    triangles = Tuple{Int,Int,Int}[]
    sizehint!(triangles, N^2)
    for i in 0:(N - 1)
        for j in 0:i
            push!(
                triangles,
                (
                    _improved_index(i + 1, j + 1),
                    _improved_index(i + 1, j),
                    _improved_index(i, j),
                ),
            )
        end
        for j in 0:(i - 1)
            push!(
                triangles,
                (
                    _improved_index(i, j),
                    _improved_index(i, j + 1),
                    _improved_index(i + 1, j + 1),
                ),
            )
        end
    end

    scale = 1
    for refinement_pass in 1:refinement_level
        node_keys, triangles, scale = _improved_refine_boundary_once(
            node_keys,
            triangles,
            scale,
            N,
            boundary_layers,
            refinement_pass,
        )
    end

    for corner_pass in 1:corner_refinement_level
        node_keys, triangles, scale = _improved_refine_corners_once(
            node_keys,
            triangles,
            scale,
            N,
            corner_layers,
            corner_pass,
        )
    end

    nodes = _improved_interval_nodes(node_keys, h, scale)
    return (; nodes, triangles, node_keys, h)
end

boundary_refined_pdelta_mesh_data(; kwargs...) = improved_pdelta_mesh_data(; kwargs...)
