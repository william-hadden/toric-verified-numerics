using IntervalArithmetic
using Logging

Logging.disable_logging(Logging.Info)
setprecision(BigFloat, 100)

include(joinpath(@__DIR__, "inverse_metric_box_bounds_v2.jl"))

hex_rotation_matrix(k) = [0 -1; 1 1]^mod(k, 6)

function rotate_hex_point((x, y), k)
    A = hex_rotation_matrix(k)
    return (A[1, 1] * x + A[1, 2] * y, A[2, 1] * x + A[2, 2] * y)
end

function rotate_hex_nodes(nodes, k)
    rotated = similar(nodes)
    for i in axes(nodes, 1)
        rotated[i, 1], rotated[i, 2] = rotate_hex_point((nodes[i, 1], nodes[i, 2]), k)
    end
    return rotated
end

function rotate_contravariant_tensor(U, k)
    A = hex_rotation_matrix(k)
    V = A * [U.xx U.xy; U.yx U.yy] * transpose(A)
    return (; xx = V[1, 1], xy = V[1, 2], yx = V[2, 1], yy = V[2, 2])
end

physical_metric_components(U) = (; xx = U.xx, xy = -U.xy, yx = -U.yx, yy = U.yy)

"""
Return a hardcoded interval triangulation of P_delta.

Here P is the lower-square fundamental triangle with vertices (0, 0), (1, 0),
and (1, -1). For now delta is fixed to 0.1, so P_delta has vertices (0, 0),
(0.9, 0), and (0.9, -0.9). The hardcoded mesh uses N = 14 uniform lattice
steps along each edge, giving 196 triangles.

The returned `nodes` is an n x 2 matrix of interval coordinates, and
`triangles` is a vector of triples of node indices. All geometric coordinates
are intervals so later mesh-generation and inset operations can preserve
outward containment without changing the assembly API.
"""
function pdelta_mesh_data()
    N = 14
    I(x) = interval(BigFloat(x, RoundDown), BigFloat(x, RoundUp))
    h = 9 // (10 * N)
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
            push!(triangles, (index(i, j), index(i + 1, j + 1), index(i + 1, j)))
        end
        for j in 0:(i - 1)
            push!(triangles, (index(i, j), index(i, j + 1), index(i + 1, j + 1)))
        end
    end

    return (; nodes, triangles, node_keys, h)
end

function get_pdelta_triangulation()
    mesh = pdelta_mesh_data()
    return mesh.nodes, mesh.triangles
end

function qdelta_triangulation()
    pmesh = pdelta_mesh_data()
    I(x) = interval(BigFloat(x, RoundDown), BigFloat(x, RoundUp))
    vertex_ids = Dict{Tuple{Int, Int}, Int}()
    vertex_keys = Tuple{Int, Int}[]

    function vertex_id(key)
        if haskey(vertex_ids, key)
            return vertex_ids[key]
        end
        push!(vertex_keys, key)
        vertex_ids[key] = length(vertex_keys)
        return length(vertex_keys)
    end

    triangles = Tuple{Int, Int, Int}[]
    source_triangles = Int[]
    for k in 0:5, (t, tri) in pairs(pmesh.triangles)
        ids = ntuple(a -> vertex_id(rotate_hex_point(pmesh.node_keys[tri[a]], k)), 3)
        push!(triangles, ids)
        push!(source_triangles, t)
    end

    nodes = Matrix{Interval{BigFloat}}(undef, length(vertex_keys), 2)
    for (i, key) in pairs(vertex_keys)
        nodes[i, 1] = I(key[1] * pmesh.h)
        nodes[i, 2] = I(key[2] * pmesh.h)
    end

    return (; nodes, triangles, source_triangles, vertex_keys)
end

"""
Compute interval bounds for the inverse metric on one triangle of P_delta.

The triangle is enclosed in an axis-aligned interval box in the lower-square
coordinates, and `inverse_metric_bounds_on_lower_square_box_v2` is called on that
box. The return value contains only the four interval tensor entries needed by
stiffness assembly: `xx`, `xy`, `yx`, and `yy`.
"""
function metric_bound_for_single_triangle(oracle, nodes, tri)
    xs = nodes[collect(tri), 1]
    ys = nodes[collect(tri), 2]
    xbox = hull(hull(xs[1], xs[2]), xs[3])
    ybox = hull(hull(ys[1], ys[2]), ys[3])

    return physical_metric_components(inverse_metric_bounds_on_lower_square_box_v2(oracle, xbox, ybox))
end

"""
Compute inverse-metric interval bounds for every triangle in a P_delta mesh.

`nodes` and `triangles` are the output of `get_pdelta_triangulation`. The
inverse-metric oracle is built once and reused for all triangles. The returned
vector has the same order as `triangles`; entry `k` contains interval bounds
`xx`, `xy`, `yx`, and `yy` for `triangles[k]`.
"""
function metric_bounds_on_pdelta_triangulation(nodes, triangles)
    oracle = InverseMetricBoxOracleV2()
    return [metric_bound_for_single_triangle(oracle, nodes, tri) for tri in triangles]
end

function poly_trim_with_tail(A, deg::Integer)
    T = eltype(A)
    B = zeros(T, min(size(A, 1), deg + 1), min(size(A, 2), deg + 1))
    tail = zero(abs(A[1, 1]))

    for j in axes(A, 2), i in axes(A, 1)
        if i <= size(B, 1) && j <= size(B, 2) && (i - 1) + (j - 1) <= deg
            B[i, j] += A[i, j]
        else
            tail += abs(A[i, j])
        end
    end

    B[1, 1] += symmetric_interval(tail)
    return B
end

function poly_add(A, B)
    T = promote_type(eltype(A), eltype(B))
    C = zeros(T, max(size(A, 1), size(B, 1)), max(size(A, 2), size(B, 2)))
    C[1:size(A, 1), 1:size(A, 2)] .+= A
    C[1:size(B, 1), 1:size(B, 2)] .+= B
    return C
end

poly_sub(A, B) = poly_add(A, -B)
poly_scale(A, c) = c .* A
poly_exact_zero(x) = isequal_interval(x, zero(x))

function poly_mul(A, B)
    T = promote_type(eltype(A), eltype(B))
    C = zeros(T, size(A, 1) + size(B, 1) - 1, size(A, 2) + size(B, 2) - 1)

    for j in axes(A, 2), i in axes(A, 1)
        poly_exact_zero(A[i, j]) && continue
        for l in axes(B, 2), k in axes(B, 1)
            poly_exact_zero(B[k, l]) && continue
            C[i + k - 1, j + l - 1] += A[i, j] * B[k, l]
        end
    end

    return C
end

poly_mul_trunc(A, B, deg::Integer) = poly_trim_with_tail(poly_mul(A, B), deg)

function local_power_with_tail(trunc, xcheb, ycheb, deg::Integer)
    P = local_power_coeffs_cheb_2d(trunc.coeffs, xcheb, ycheb)
    P[1, 1] += symmetric_interval(trunc.tail)
    return poly_trim_with_tail(P, deg)
end

function local_power_trimmed(coeffs, xcheb, ycheb, deg::Integer)
    return poly_trim_with_tail(local_power_coeffs_cheb_2d(coeffs, xcheb, ycheb), deg)
end

function local_metric_component_polynomials(oracle, xcheb, ycheb; deg::Integer = 8)
    v = oracle.canonical

    h11 = local_power_with_tail(oracle.h11, xcheb, ycheb, deg)
    h12 = local_power_with_tail(oracle.h12, xcheb, ycheb, deg)
    h22 = local_power_with_tail(oracle.h22, xcheb, ycheb, deg)

    lprod = local_power_trimmed(v.lprod, xcheb, ycheb, deg)
    v11 = local_power_trimmed(v.lprod_v11, xcheb, ycheb, deg)
    v12 = local_power_trimmed(v.lprod_v12, xcheb, ycheb, deg)
    v22 = local_power_trimmed(v.lprod_v22, xcheb, ycheb, deg)
    B = local_power_trimmed(v.B, xcheb, ycheb, deg)

    A11 = poly_trim_with_tail(poly_add(v22, poly_mul_trunc(lprod, h22, deg)), deg)
    A12 = poly_scale(poly_trim_with_tail(poly_add(v12, poly_mul_trunc(lprod, h12, deg)), deg), -interval(BigFloat(1)))
    A22 = poly_trim_with_tail(poly_add(v11, poly_mul_trunc(lprod, h11, deg)), deg)

    D = poly_add(B, poly_mul_trunc(v22, h11, deg))
    D = poly_add(D, poly_mul_trunc(v11, h22, deg))
    D = poly_sub(D, poly_scale(poly_mul_trunc(v12, h12, deg), interval(BigFloat(2))))
    D = poly_add(D, poly_mul_trunc(lprod, poly_sub(poly_mul_trunc(h11, h22, deg), poly_mul_trunc(h12, h12, deg)), deg))
    D = poly_trim_with_tail(D, deg)

    return (; A11, A12, A22, D)
end

poly_abs_bound(A) = sum(abs(A[i, j]) for j in axes(A, 2), i in axes(A, 1))

function reciprocal_polynomial_neumann(D; deg::Integer = 8, terms::Integer = 8)
    d0 = (inf(D[1, 1]) + sup(D[1, 1])) / 2
    R = copy(D)
    R[1, 1] -= interval(d0)

    ratio = sup(poly_abs_bound(R)) / abs(d0)
    ratio < 1 || error("Neumann series does not contract: ratio = $ratio")

    T = eltype(D)
    P = zeros(T, 1, 1)
    Rpow = zeros(T, 1, 1)
    Rpow[1, 1] = interval(BigFloat(1))

    for k in 0:terms
        P = poly_add(P, poly_scale(Rpow, interval((-1)^k) / interval(d0)^(k + 1)))
        Rpow = poly_mul_trunc(Rpow, R, deg)
    end

    remainder = (ratio^(terms + 1)) / (abs(d0) * (1 - ratio))
    P[1, 1] += symmetric_interval(interval(BigFloat(remainder)))
    return poly_trim_with_tail(P, deg)
end

J_moment(n::Integer) = iseven(n) ? interval(BigFloat(2) / BigFloat(n + 1)) : interval(BigFloat(0))

function half_square_moment(p::Integer, q::Integer, upper::Bool)
    if upper
        return (J_moment(p) - interval(BigFloat((-1)^(q + 1))) * J_moment(p + q + 1)) / interval(BigFloat(q + 1))
    end

    return interval(BigFloat((-1)^(q + 1))) * (J_moment(p + q + 1) - J_moment(p)) / interval(BigFloat(q + 1))
end

function triangle_is_upper_half(nodes, tri, xbox, ybox)
    xc = (inf(xbox) + sup(xbox)) / 2
    yc = (inf(ybox) + sup(ybox)) / 2
    rx = (sup(xbox) - inf(xbox)) / 2
    ry = (sup(ybox) - inf(ybox)) / 2
    vals = [
        ((inf(nodes[i, 1]) + sup(nodes[i, 1])) / 2 - xc) / rx +
        ((inf(nodes[i, 2]) + sup(nodes[i, 2])) / 2 - yc) / ry
        for i in tri
    ]
    return maximum(vals) > -minimum(vals)
end

function integrate_local_polynomial_on_triangle(P, nodes, tri, xbox, ybox)
    rx = interval((sup(xbox) - inf(xbox)) / 2)
    ry = interval((sup(ybox) - inf(ybox)) / 2)
    upper = triangle_is_upper_half(nodes, tri, xbox, ybox)
    total = zero(P[1, 1])

    for j in axes(P, 2), i in axes(P, 1)
        total += P[i, j] * half_square_moment(i - 1, j - 1, upper)
    end

    return rx * ry * total
end

function metric_integral_for_single_triangle(oracle, nodes, tri; deg::Integer = 8, terms::Integer = 8)
    xs = nodes[collect(tri), 1]
    ys = nodes[collect(tri), 2]
    xbox = hull(hull(xs[1], xs[2]), xs[3])
    ybox = hull(hull(ys[1], ys[2]), ys[3])
    xcheb = interval_constant(2) * xbox - interval_constant(1)
    ycheb = interval_constant(2) * ybox + interval_constant(1)

    C = local_metric_component_polynomials(oracle, xcheb, ycheb; deg)
    invD = reciprocal_polynomial_neumann(C.D; deg, terms)

    return physical_metric_components((
        xx = integrate_local_polynomial_on_triangle(poly_mul(C.A11, invD), nodes, tri, xbox, ybox),
        xy = integrate_local_polynomial_on_triangle(poly_mul(C.A12, invD), nodes, tri, xbox, ybox),
        yx = integrate_local_polynomial_on_triangle(poly_mul(C.A12, invD), nodes, tri, xbox, ybox),
        yy = integrate_local_polynomial_on_triangle(poly_mul(C.A22, invD), nodes, tri, xbox, ybox),
    ))
end

function metric_integrals_on_pdelta_triangulation(nodes, triangles; deg::Integer = 8, terms::Integer = 8)
    oracle = InverseMetricBoxOracleV2()
    return [metric_integral_for_single_triangle(oracle, nodes, tri; deg, terms) for tri in triangles]
end

function triangle_double_area(nodes, tri)
    x1, y1 = nodes[tri[1], 1], nodes[tri[1], 2]
    x2, y2 = nodes[tri[2], 1], nodes[tri[2], 2]
    x3, y3 = nodes[tri[3], 1], nodes[tri[3], 2]
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
end

triangle_area(nodes, tri) = triangle_double_area(nodes, tri) / interval(BigFloat(2))

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

function local_cr_mass_matrix(nodes, tri)
    area = triangle_area(nodes, tri)
    M = fill(zero(area), 3, 3)
    for a in 1:3
        M[a, a] = area / interval(BigFloat(3))
    end
    return M
end

function local_cr_stiffness_matrix(nodes, tri, U)
    area = triangle_area(nodes, tri)
    G = cr_gradients(nodes, tri)
    K = fill(zero(area), 3, 3)

    for a in 1:3, b in 1:3
        K[a, b] = area * (
            G[a, 1] * U.xx * G[b, 1] +
            G[a, 1] * U.xy * G[b, 2] +
            G[a, 2] * U.yx * G[b, 1] +
            G[a, 2] * U.yy * G[b, 2]
        )
    end

    return K
end

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

local_cr_edges(tri) = ((tri[2], tri[3]), (tri[3], tri[1]), (tri[1], tri[2]))
edge_key(i, j) = i < j ? (i, j) : (j, i)

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

function addblock!(A, dofs, B)
    for a in 1:3, b in 1:3
        iszero(B[a, b]) && continue
        key = (dofs[a], dofs[b])
        A[key] = get(A, key, zero(B[a, b])) + B[a, b]
    end
end

function assemble_qdelta_cr_matrices(nodes, triangles, metric_integrals)
    qmesh = qdelta_triangulation()
    edges, triangle_edges = build_cr_edges(qmesh.triangles)
    local_mass = [local_cr_mass_matrix(nodes, tri) for tri in triangles]
    local_stiffness = [
        local_cr_stiffness_matrix_from_integrals(nodes, triangles[t], metric_integrals[t])
        for t in eachindex(triangles)
    ]
    mass = Dict{Tuple{Int, Int}, Interval{BigFloat}}()
    stiffness = Dict{Tuple{Int, Int}, Interval{BigFloat}}()
    element_mass = Matrix{Interval{BigFloat}}[]
    element_stiffness = Matrix{Interval{BigFloat}}[]

    for t in eachindex(qmesh.triangles)
        source = qmesh.source_triangles[t]
        dofs = triangle_edges[t]
        M = local_mass[source]
        K = local_stiffness[source]
        push!(element_mass, M)
        push!(element_stiffness, K)
        addblock!(mass, dofs, M)
        addblock!(stiffness, dofs, K)
    end

    return (;
        nodes = qmesh.nodes,
        triangles = qmesh.triangles,
        source_triangles = qmesh.source_triangles,
        edges,
        triangle_edges,
        mass,
        stiffness,
        local_mass,
        local_stiffness,
        element_mass,
        element_stiffness,
    )
end

float_down(x) = Float64(x, RoundDown)
float_up(x) = Float64(x, RoundUp)

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

function savematrix(path, A, n)
    ij = sort!(collect(keys(A)))
    mat4write(path, [
        "n" => n,
        "i" => first.(ij),
        "j" => last.(ij),
        "lo" => [float_down(inf(A[key])) for key in ij],
        "hi" => [float_up(sup(A[key])) for key in ij],
    ])
end

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

function write_matlab_matrices(assembly; dir = @__DIR__)
    n = length(assembly.edges)
    savematrix(joinpath(dir, "stiff_matrix.mat"), symmetrize_entries(assembly.stiffness), n)
    savematrix(joinpath(dir, "mass_matrix.mat"), symmetrize_entries(assembly.mass), n)
end

"""
Run the current Step 1 prototype.

This constructs the hardcoded P_delta triangulation and computes
triangle-wise inverse-metric bounds in memory. Nothing is written to disk.
"""
function main()
    nodes, triangles = get_pdelta_triangulation()
    println("P_delta triangulation: $(size(nodes, 1)) nodes, $(length(triangles)) triangles")

    elapsed = @elapsed metric_integrals = metric_integrals_on_pdelta_triangulation(nodes, triangles)
    println("metric integrals computed in $(round(elapsed; digits = 3)) seconds")

    println("max width int_xx <= $(maximum(sup(U.xx) - inf(U.xx) for U in metric_integrals))")
    println("max width int_xy <= $(maximum(sup(U.xy) - inf(U.xy) for U in metric_integrals))")
    println("max width int_yx <= $(maximum(sup(U.yx) - inf(U.yx) for U in metric_integrals))")
    println("max width int_yy <= $(maximum(sup(U.yy) - inf(U.yy) for U in metric_integrals))")

    assembly = assemble_qdelta_cr_matrices(nodes, triangles, metric_integrals)
    println("Q_delta triangulation: $(size(assembly.nodes, 1)) nodes, $(length(assembly.triangles)) triangles")
    println("CR matrices: $(length(assembly.edges)) x $(length(assembly.edges))")
    println("mass stored entries: $(length(assembly.mass))")
    println("stiffness stored entries: $(length(assembly.stiffness))")

    write_matlab_matrices(assembly)
    println("wrote ", joinpath(@__DIR__, "stiff_matrix.mat"))
    println("wrote ", joinpath(@__DIR__, "mass_matrix.mat"))

    return (; nodes, triangles, metric_integrals, assembly)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
