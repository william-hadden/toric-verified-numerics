using IntervalArithmetic
using Logging

Logging.disable_logging(Logging.Info)
setprecision(BigFloat, 100)

include(joinpath(@__DIR__, "inverse_metric_box_bounds_v2.jl"))
include(joinpath(@__DIR__, "util", "local_polynomials.jl"))

const DEFAULT_DELTA = 1 // 20000
const DEFAULT_N = 14

"""
Flips the sign of the xy and yx components of a 2x2 tensor. This is the correct
transformation if transforming from [0,1]x[-1,0] to [0,1]x[0,1].
"""
physical_metric_components(U) = (; xx = U.xx, xy = -U.xy, yx = -U.yx, yy = U.yy)

"""
Return a hardcoded interval triangulation of P_delta.

Here P is the lower-square fundamental triangle with vertices (0, 0), (1, 0),
and (1, -1). The default inset is delta = 1/20000, and the default mesh uses
N = 14 uniform lattice steps along each edge, giving 196 triangles.

The returned `nodes` is an n x 2 matrix of interval coordinates, and
`triangles` is a vector of positively oriented triples of node indices whose
second vertex is the right-angle vertex. All geometric coordinates are intervals
so later mesh-generation and inset operations can preserve outward containment
without changing the assembly API.
"""
function pdelta_mesh_data(; delta = DEFAULT_DELTA, N::Integer = DEFAULT_N)
    I(x) = interval(BigFloat(x, RoundDown), BigFloat(x, RoundUp))
    h = (1 - delta) // N
    index(i, j) = i * (i + 1) ÷ 2 + j + 1

    nodes = Matrix{Interval{BigFloat}}(undef, (N + 1) * (N + 2) ÷ 2, 2)
    node_keys = Vector{Tuple{Int, Int}}(undef, size(nodes, 1))
    for i in 0:N, j in 0:i
        key = (i, -j)
        node_keys[index(i, j)] = key
        nodes[index(i, j), 1] = I(key[1] * h)
        nodes[index(i, j), 2] = I(key[2] * h)
    end

    triangles = Tuple{Int, Int, Int}[]
    for i in 0:(N - 1)
        for j in 0:i
            push!(triangles, (index(i + 1, j + 1), index(i + 1, j), index(i, j)))
        end
        for j in 0:(i - 1)
            push!(triangles, (index(i, j), index(i, j + 1), index(i + 1, j + 1)))
        end
    end

    return (; nodes, triangles, node_keys, h)
end

"""
Return the monomial moment of degree n on [-1, 1]:
the integral from -1 to 1 of x^n dx.
"""
J_moment(n::Integer) = iseven(n) ? interval(BigFloat(2) / BigFloat(n + 1)) : interval(BigFloat(0))

"""
Return the `x^p y^q` moment on one half of `[-1,1]^2`. upper == true
computes on the half `x + y >= 0`. upper == false computes on the half
`x + y <= 0`.
"""
function half_square_moment(p::Integer, q::Integer, upper::Bool)
    if upper
        return (J_moment(p) - interval(BigFloat((-1)^(q + 1))) * J_moment(p + q + 1)) / interval(BigFloat(q + 1))
    end

    return interval(BigFloat((-1)^(q + 1))) * (J_moment(p + q + 1) - J_moment(p)) / interval(BigFloat(q + 1))
end

"""
For this structured triangulation, return whether the right-angle vertex lies
above the midpoint of the opposite edge.
"""
function triangle_is_upper_half(nodes, tri)
    right_angle_y = nodes[tri[2], 2]
    opposite_midpoint_y = (nodes[tri[1], 2] + nodes[tri[3], 2]) / interval(BigFloat(2))
    offset = right_angle_y - opposite_midpoint_y

    inf(offset) > 0 && return true
    sup(offset) < 0 && return false
    error("Could not determine which half-square contains the triangle")
end

"""
Integrate a local polynomial (in the standard monomial basis, not Chebyshev) 
over one mesh triangle.
"""
function integrate_local_polynomial_on_triangle(P, nodes, tri, xbox, ybox)
    rx = interval((sup(xbox) - inf(xbox)) / 2)
    ry = interval((sup(ybox) - inf(ybox)) / 2)
    upper = triangle_is_upper_half(nodes, tri)
    total = interval(BigFloat(0))

    for j in axes(P, 2), i in axes(P, 1)
        total += P[i, j] * half_square_moment(i - 1, j - 1, upper)
    end

    return rx * ry * total
end

"""
Compute interval integrals of the inverse metric components over one triangle.
"""
function metric_integral_for_single_triangle(oracle, nodes, tri; deg::Integer = 8, terms::Integer = 8)
    xs = nodes[collect(tri), 1]
    ys = nodes[collect(tri), 2]
    xbox = hull(hull(xs[1], xs[2]), xs[3])
    ybox = hull(hull(ys[1], ys[2]), ys[3])
    xcheb = exact(2) * xbox - exact(1)
    ycheb = exact(2) * ybox + exact(1)

    C = local_metric_component_polynomials(oracle, xcheb, ycheb; deg)
    invD = reciprocal_polynomial_neumann(C.D; deg, terms)

    return physical_metric_components((
        xx = integrate_local_polynomial_on_triangle(poly_mul(C.A11, invD), nodes, tri, xbox, ybox),
        xy = integrate_local_polynomial_on_triangle(poly_mul(C.A12, invD), nodes, tri, xbox, ybox),
        yx = integrate_local_polynomial_on_triangle(poly_mul(C.A12, invD), nodes, tri, xbox, ybox),
        yy = integrate_local_polynomial_on_triangle(poly_mul(C.A22, invD), nodes, tri, xbox, ybox),
    ))
end

"""
Compute inverse-metric component integrals for every triangle in a P_delta mesh.
"""
function metric_integrals_on_pdelta_triangulation(nodes, triangles; deg::Integer = 8, terms::Integer = 8)
    oracle = InverseMetricBoxOracleV2()
    return [metric_integral_for_single_triangle(oracle, nodes, tri; deg, terms) for tri in triangles]
end

"""
Return the signed double area of a triangle.
"""
function triangle_double_area(nodes, tri)
    x1, y1 = nodes[tri[1], 1], nodes[tri[1], 2]
    x2, y2 = nodes[tri[2], 1], nodes[tri[2], 2]
    x3, y3 = nodes[tri[3], 1], nodes[tri[3], 2]
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
end

"""
Return the three constant Crouzeix-Raviart basis gradients on a triangle.
"""
function cr_gradients(nodes, tri)
    x1, y1 = nodes[tri[1], 1], nodes[tri[1], 2]
    x2, y2 = nodes[tri[2], 1], nodes[tri[2], 2]
    x3, y3 = nodes[tri[3], 1], nodes[tri[3], 2]
    detJ = triangle_double_area(nodes, tri)

    return -interval(BigFloat(2)) * [
        y2 - y3  x3 - x2
        y3 - y1  x1 - x3
        y1 - y2  x2 - x1
    ] / detJ
end

"""
Return the local diagonal CR mass matrix on a triangle. It is a diagonal
matrix.
"""
function local_cr_mass_matrix(nodes, tri)
    area = triangle_double_area(nodes, tri) / interval(BigFloat(2))
    M = fill(zero(area), 3, 3)
    for a in 1:3
        M[a, a] = area / interval(BigFloat(3))
    end
    return M
end

"""
Return the local CR stiffness matrix from preintegrated metric components.
"""
function local_cr_stiffness_matrix_from_integrals(nodes, tri, U)
    G = cr_gradients(nodes, tri)
    K = fill(zero(U.xx), 3, 3)

    for a in 1:3, b in 1:3
        K[a, b] =
            G[a, 1] * U.xx * G[b, 1] +
            G[a, 1] * U.xy * G[b, 2] +
            G[a, 2] * U.yx * G[b, 1] +
            G[a, 2] * U.yy * G[b, 2]
    end

    return K
end

"""
Return the three opposite edges used as local CR degrees of freedom.
"""
local_cr_edges(tri) = ((tri[2], tri[3]), (tri[3], tri[1]), (tri[1], tri[2]))

"""
Return a canonical ordered key for an undirected edge.
"""
edge_key(i, j) = i < j ? (i, j) : (j, i)

"""
Build global CR edge degrees of freedom for a triangle list.
"""
function build_cr_edges(triangles)
    edge_ids = Dict{Tuple{Int, Int}, Int}()
    edges = Tuple{Int, Int}[]
    triangle_edges = Tuple{Int, Int, Int}[]

    function edge_id(edge)
        key = edge_key(edge...)
        if haskey(edge_ids, key)
            return edge_ids[key]
        end
        push!(edges, key)
        edge_ids[key] = length(edges)
        return length(edges)
    end

    for tri in triangles
        push!(triangle_edges, ntuple(a -> edge_id(local_cr_edges(tri)[a]), 3))
    end

    return edges, triangle_edges
end

"""
Add a dense local matrix block B into a sparse dictionary matrix A.
It will be added at keys dof.
"""
function addblock!(A, dofs, B)
    for a in 1:3, b in 1:3
        iszero(B[a, b]) && continue
        key = (dofs[a], dofs[b])
        A[key] = get(A, key, zero(B[a, b])) + B[a, b]
    end
end

"""
Rotate an integer lattice key by `k` units of the D6 rotation.
This rotates fundamental domains within the standard hexagon with
vertices (-1,1),(-1,0),(0,-1),(1,-1),(1,0),(0,1). Here, (a,b) are
integers identifying lattice points. (a,b) encodes the lattice point
(a*h,b*h), see the construction in pdelta_mesh_data.
"""
rotkey((a, b), k) = k == 0 ? (a, b) : rotkey((-b, a + b), k - 1)

"""
Reflect a sector edge key across the lower-square diagonal.
"""
sector_reflect_key((a, b)) = (a, -a - b)

"""
Return exact doubled midpoint lattice coordinates for CR edge degrees of freedom.

`edges` is a list of vertex-index pairs `(i, j)`. `vertex_keys[v]` is the
integer lattice coordinate `(a, b)` of vertex `v`, representing the physical
point `(a*h, b*h)`. For each edge, this returns `vertex_keys[i] + vertex_keys[j]`,
which represents twice the edge midpoint and avoids half-integer coordinates.

Example: if an edge joins vertices with keys `(3, -1)` and `(4, -2)`, the
returned key is `(7, -3)`, representing midpoint `(7h/2, -3h/2)`.
"""
function edge_key_sums(edges, vertex_keys)
    return [
        (vertex_keys[i][1] + vertex_keys[j][1], vertex_keys[i][2] + vertex_keys[j][2])
        for (i, j) in edges
    ]
end

"""
Return the quotient degree-of-freedom index for each original index.

The original indices are `1:n`. Each pair `(i, j)` in `pairs` declares that
indices `i` and `j` represent the same quotient degree of freedom. The returned
vector `q` has length `n`, and `q[i]` is the new consecutive quotient index for
old index `i`.

Example: if `n = 9` and `pairs = [(2, 5), (5, 9), (3, 7)]`, then indices
`2, 5, 9` are identified, and indices `3, 7` are identified. A possible return
value is `[1, 2, 3, 4, 2, 5, 3, 6, 2]`.
"""
function quotient_map(n::Integer, pairs)
    parent = collect(1:n)

    function find_root(i)
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end
        return i
    end

    for (i, j) in pairs
        ri, rj = find_root(i), find_root(j)
        ri == rj || (parent[rj] = ri)
    end

    root_ids = Dict{Int, Int}()
    return [get!(root_ids, find_root(i), length(root_ids) + 1) for i in 1:n]
end

"""
Compute the D6-invariant quotient map for sector CR edge degrees of freedom.
"""
function d6_sector_edge_quotient(edges, vertex_keys)
    sums = edge_key_sums(edges, vertex_keys)
    edge_id = Dict(sums[i] => i for i in eachindex(sums))
    pairs = Tuple{Int, Int}[]

    for i in eachindex(sums)
        reflected = sector_reflect_key(sums[i])
        haskey(edge_id, reflected) && push!(pairs, (i, edge_id[reflected]))

        if sums[i][2] == -sums[i][1]
            rotated = rotkey(sums[i], 1)
            haskey(edge_id, rotated) && push!(pairs, (i, edge_id[rotated]))
        end
    end

    return quotient_map(length(edges), pairs)
end

"""
Assemble D6-invariant CR mass and stiffness matrices on the fundamental
domain triangle P_delta.
"""
function assemble_d6_invariant_cr_matrices(nodes, triangles, metric_integrals; delta = DEFAULT_DELTA, N::Integer = DEFAULT_N)
    pmesh = pdelta_mesh_data(; delta, N)
    edges, triangle_edges = build_cr_edges(triangles)
    quotient = d6_sector_edge_quotient(edges, pmesh.node_keys)
    local_mass = [local_cr_mass_matrix(nodes, tri) for tri in triangles]
    local_stiffness = [
        local_cr_stiffness_matrix_from_integrals(nodes, triangles[t], metric_integrals[t])
        for t in eachindex(triangles)
    ]
    mass = Dict{Tuple{Int, Int}, Interval{BigFloat}}()
    stiffness = Dict{Tuple{Int, Int}, Interval{BigFloat}}()

    for t in eachindex(triangles)
        dofs = ntuple(a -> quotient[triangle_edges[t][a]], 3)
        addblock!(mass, dofs, local_mass[t])
        addblock!(stiffness, dofs, local_stiffness[t])
    end

    return (; matrix_size = maximum(quotient), mass, stiffness)
end

"""
Write MATLAB v4 variables from name-value entries.
"""
function mat4write(path, entries)
    open(path, "w") do io
        for (name, value) in entries
            A = value isa Number ? reshape([Float64(value)], 1, 1) : reshape(Float64.(value), :, 1)
            write(io, Int32(0), Int32(size(A, 1)), Int32(size(A, 2)), Int32(0), Int32(length(name) + 1))
            write(io, codeunits(name))
            write(io, UInt8(0))
            write(io, vec(A))
        end
    end
end

"""
Save a sparse interval matrix as MATLAB v4 endpoint arrays.
"""
function savematrix(path, A, n)
    ij = sort!(collect(keys(A)))
    mat4write(path, [
        "n" => n,
        "i" => first.(ij),
        "j" => last.(ij),
        "lo" => [Float64(inf(A[key]), RoundDown) for key in ij],
        "hi" => [Float64(sup(A[key]), RoundUp) for key in ij],
    ])
end

"""
Widen paired entries so a dictionary matrix is exactly symmetric.
"""
function symmetrize_entries(A)
    S = copy(A)
    for (i, j) in collect(keys(A))
        haskey(A, (j, i)) || continue
        value = interval(min(inf(S[(i, j)]), inf(S[(j, i)])), max(sup(S[(i, j)]), sup(S[(j, i)])))
        S[(i, j)] = value
        S[(j, i)] = value
    end
    return S
end

"""
Write stiffness and mass matrices for MATLAB/INTLAB.
"""
function write_matlab_matrices(assembly; dir = @__DIR__)
    n = assembly.matrix_size
    savematrix(joinpath(dir, "stiff_matrix.mat"), symmetrize_entries(assembly.stiffness), n)
    savematrix(joinpath(dir, "mass_matrix.mat"), symmetrize_entries(assembly.mass), n)
end

"""
Assemble the D6-invariant CR eigenvalue problem and write MATLAB input files.
"""
function main(; delta = DEFAULT_DELTA, N::Integer = DEFAULT_N)
    pmesh = pdelta_mesh_data(; delta, N)
    nodes, triangles = pmesh.nodes, pmesh.triangles
    metric_integrals = metric_integrals_on_pdelta_triangulation(nodes, triangles)
    assembly = assemble_d6_invariant_cr_matrices(nodes, triangles, metric_integrals; delta, N)
    write_matlab_matrices(assembly)
    return assembly
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
