using IntervalArithmetic
using Logging

Logging.disable_logging(Logging.Info)
setprecision(BigFloat, 100)

include(joinpath(@__DIR__, "inverse_metric_box_bounds_v2.jl"))

const DEFAULT_DELTA = 1 // 20000
const DEFAULT_N = 14

physical_metric_components(U) = (; xx = U.xx, xy = -U.xy, yx = -U.yx, yy = U.yy)

"""
Return a hardcoded interval triangulation of P_delta.

Here P is the lower-square fundamental triangle with vertices (0, 0), (1, 0),
and (1, -1). The default inset is delta = 1/20000, and the default mesh uses
N = 14 uniform lattice steps along each edge, giving 196 triangles.

The returned `nodes` is an n x 2 matrix of interval coordinates, and
`triangles` is a vector of triples of node indices. All geometric coordinates
are intervals so later mesh-generation and inset operations can preserve
outward containment without changing the assembly API.
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
            push!(triangles, (index(i, j), index(i + 1, j + 1), index(i + 1, j)))
        end
        for j in 0:(i - 1)
            push!(triangles, (index(i, j), index(i, j + 1), index(i + 1, j + 1)))
        end
    end

    return (; nodes, triangles, node_keys, h)
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

rotkey((a, b), k) = k == 0 ? (a, b) : rotkey((-b, a + b), k - 1)
sector_reflect_key((a, b)) = (a, -a - b)

function edge_key_sums(edges, vertex_keys)
    return [
        (vertex_keys[i][1] + vertex_keys[j][1], vertex_keys[i][2] + vertex_keys[j][2])
        for (i, j) in edges
    ]
end

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
