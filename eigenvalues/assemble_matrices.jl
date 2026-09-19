using IntervalArithmetic

# Select the BigFloat-compatible interval matrix product without per-call fallback.
IntervalArithmetic.configure(; matmul = :slow)

include(joinpath(@__DIR__, "inverse_metric_bounds.jl"))

"""Return twice the signed physical area using interval-enclosed coordinates."""
function twice_signed_triangle_area(nodes, tri)
    x1, y1 = nodes[tri[1], 1], nodes[tri[1], 2]
    x2, y2 = nodes[tri[2], 1], nodes[tri[2], 2]
    x3, y3 = nodes[tri[3], 1], nodes[tri[3], 2]
    return (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
end

"""Return the three constant Crouzeix--Raviart basis gradients."""
function cr_gradients(nodes, tri)
    x1, y1 = nodes[tri[1], 1], nodes[tri[1], 2]
    x2, y2 = nodes[tri[2], 1], nodes[tri[2], 2]
    x3, y3 = nodes[tri[3], 1], nodes[tri[3], 2]
    determinant = twice_signed_triangle_area(nodes, tri)
    inf(determinant) > 0 || error("Triangle $tri is not positively oriented")
    return -interval(BigFloat(2)) * [
        y2 - y3  x3 - x2
        y3 - y1  x1 - x3
        y1 - y2  x2 - x1
    ] / determinant
end

"""Return the local diagonal Crouzeix--Raviart mass matrix, i.e. the mass
matrix for CR elements on one single triangle."""
function local_cr_mass_matrix(nodes, tri)
    area = twice_signed_triangle_area(nodes, tri) / interval(BigFloat(2))
    mass = fill(zero(area), 3, 3)
    for index in 1:3
        mass[index, index] = area / interval(BigFloat(3))
    end
    return mass
end

"""Return the local stiffness matrix for a constant inner product `B`."""
function local_cr_stiffness_matrix(nodes, tri, B)
    area = twice_signed_triangle_area(nodes, tri) / interval(BigFloat(2))
    gradients = cr_gradients(nodes, tri)
    coefficient = interval.(B)
    return area * (gradients * coefficient * transpose(gradients))
end

"""
Return the canonical mesh edges associated with the local CR basis functions.

Entry `i` joins the two vertices other than `tri[i]`, matching gradient row `i`.
"""
function cr_basis_edges(tri)
    a, b, c = tri
    return (minmax(b, c), minmax(c, a), minmax(a, b))
end

"""Build the global edge degrees of freedom for a triangle list."""
function build_cr_edges(triangles)
    edge_ids = Dict{Tuple{Int,Int},Int}()
    edges = Tuple{Int,Int}[]
    triangle_edge_ids = Tuple{Int,Int,Int}[]
    for tri in triangles
        local_edges = cr_basis_edges(tri)
        ids = ntuple(3) do position
            edge = local_edges[position]
            get!(edge_ids, edge) do
                push!(edges, edge)
                length(edges)
            end
        end
        push!(triangle_edge_ids, ids)
    end
    return edges, triangle_edge_ids
end

"""Add one dense local interval block to a sparse dictionary matrix."""
function add_block!(matrix, dofs, block)
    for a in 1:3, b in 1:3
        value = block[a, b]
        isequal_interval(value, zero(value)) && continue
        key = (dofs[a], dofs[b])
        matrix[key] = get(matrix, key, zero(value)) + value
    end
    return matrix
end

"""Apply the D6 reflection exchanging the two sides of the fundamental sector
that are not a hexagon boundary."""
sector_reflect_key((a, b)) = (a, -a - b)

"""
Return twice each edge midpoint in exact lattice coordinates.

For edge `(i, j)`, the corresponding result is
`node_keys[i] + node_keys[j]`. Doubled coordinates avoid half-integers.
"""
function edge_midpoint_keys(edges, node_keys)
    return [
        (node_keys[i][1] + node_keys[j][1], node_keys[i][2] + node_keys[j][2])
        for (i, j) in edges
    ]
end

"""
Map each sector CR edge to its D6-invariant degree of freedom.

The smaller of a midpoint key and its reflection is their common orbit key.
"""
function d6_edge_quotient(edges, node_keys)
    orbit_ids = Dict{Tuple{Int,Int},Int}()
    return map(edge_midpoint_keys(edges, node_keys)) do midpoint
        orbit_key = min(midpoint, sector_reflect_key(midpoint))
        get!(orbit_ids, orbit_key, length(orbit_ids) + 1)
    end
end

"""Return a rigorous upper bound for the longest side of a triangle."""
function triangle_diameter_upper(nodes, tri)
    diameter = BigFloat(0)
    for (a, b) in ((1, 2), (2, 3), (3, 1))
        dx = nodes[tri[a], 1] - nodes[tri[b], 1]
        dy = nodes[tri[a], 2] - nodes[tri[b], 2]
        diameter = max(diameter, sup(sqrt(dx^2 + dy^2)))
    end
    isfinite(diameter) && diameter > 0 || error("Triangle $tri has invalid diameter")
    return diameter
end

"""Return the rigorous local Liu constant `0.1893 h_T/sqrt(alpha_T)`."""
function local_liu_constant(diameter_upper, alpha_lower)
    alpha_lower > 0 || error("The eigenvalue lower bound must be positive")
    factor = interval(BigFloat(1893)) / interval(BigFloat(10000))
    value = factor * interval(diameter_upper) / sqrt(interval(alpha_lower))
    upper = sup(value)
    isfinite(upper) && upper > 0 || error("The local Liu constant is invalid")
    return upper
end

"""Certify one element until its Liu constant is at most `liu_constant_target`."""
function certify_element(oracle, mesh, tri, degree, terms, bisection_steps, liu_constant_target)
    diameter = triangle_diameter_upper(mesh.nodes, tri)
    min_depth = 0
    while true
        bound = inverse_metric_lower_bound(
            oracle,
            mesh.nodes,
            tri,
            min_depth,
            degree,
            terms,
            bisection_steps,
        )
        liu_constant = local_liu_constant(diameter, bound.alpha)
        if liu_constant <= liu_constant_target
            return (; B = bound.B, alpha = bound.alpha, liu_constant)
        end

        beta = minimum(symmetric_eigenvalue_lower_bound(leaf.G) for leaf in bound.leaves)
        scalar_liu_constant = local_liu_constant(diameter, beta)
        if scalar_liu_constant <= liu_constant_target
            return (; B = BigFloat[beta 0; 0 beta],
                    alpha = beta,
                    liu_constant = scalar_liu_constant)
        end
        min_depth += 1
    end
end

"""
Certify every element of `mesh` with `oracle`.

`degree`, `terms`, and `bisection_steps` control the inverse-metric enclosure;
`liu_constant_target` bounds each local Liu constant, and `progress_interval`
is the threaded batch size. Return one certificate per triangle in mesh order.
"""
function certify_elements(
    oracle,
    mesh,
    degree,
    terms,
    bisection_steps,
    liu_constant_target,
    progress_interval,
)
    liu_constant_threshold = inf(interval(BigFloat, liu_constant_target))
    triangle_count = length(mesh.triangles)
    certificates = Vector{Any}(undef, triangle_count)
    for first_index in 1:progress_interval:triangle_count
        last_index = min(first_index + progress_interval - 1, triangle_count)
        Threads.@threads for index in first_index:last_index
            certificates[index] = certify_element(
                oracle,
                mesh,
                mesh.triangles[index],
                degree,
                terms,
                bisection_steps,
                liu_constant_threshold,
            )
        end
        println("certified $last_index / $triangle_count elements")
        flush(stdout)
    end
    return certificates
end

"""Assemble the D6-quotient CR mass and certified comparison stiffness matrices."""
function assemble_matrices(mesh, certificates)
    length(certificates) == length(mesh.triangles) ||
        throw(DimensionMismatch("Expected one certificate per triangle"))
    edges, triangle_edge_ids = build_cr_edges(mesh.triangles)
    quotient = d6_edge_quotient(edges, mesh.node_keys)
    mass = Dict{Tuple{Int,Int},Interval{BigFloat}}()
    stiffness = Dict{Tuple{Int,Int},Interval{BigFloat}}()
    for index in eachindex(mesh.triangles)
        tri = mesh.triangles[index]
        dofs = ntuple(position -> quotient[triangle_edge_ids[index][position]], 3)
        add_block!(mass, dofs, local_cr_mass_matrix(mesh.nodes, tri))
        local_stiffness = local_cr_stiffness_matrix(
            mesh.nodes,
            tri,
            certificates[index].B,
        )
        add_block!(stiffness, dofs, local_stiffness)
    end
    liu_constant = maximum(certificate.liu_constant for certificate in certificates)
    return (; matrix_size = maximum(quotient), mass, stiffness, liu_constant)
end

"""Write named numeric arrays in the MATLAB version-4 binary format."""
function write_mat4(path, entries)
    open(path, "w") do io
        for (name, value) in entries
            array = if value isa Number
                reshape([Float64(value)], 1, 1)
            elseif value isa AbstractVector
                reshape(Float64.(value), :, 1)
            else
                Float64.(value)
            end
            write(io, Int32(0), Int32(size(array, 1)), Int32(size(array, 2)))
            write(io, Int32(0), Int32(length(name) + 1))
            write(io, codeunits(name))
            write(io, UInt8(0))
            write(io, vec(array))
        end
    end
end

"""Save a sparse interval matrix as outward-rounded MATLAB endpoint arrays."""
function save_interval_matrix(path, matrix, size_)
    entries = sort!(collect(keys(matrix)))
    write_mat4(path, [
        "n" => size_,
        "i" => first.(entries),
        "j" => last.(entries),
        "lo" => [Float64(inf(matrix[key]), RoundDown) for key in entries],
        "hi" => [Float64(sup(matrix[key]), RoundUp) for key in entries],
    ])
end

"""Write the fixed stiffness and mass inputs ingested by MATLAB and INTLAB."""
function write_matlab_matrices(assembly, directory)
    save_interval_matrix(
        joinpath(directory, "stiff_matrix.mat"),
        assembly.stiffness,
        assembly.matrix_size,
    )
    save_interval_matrix(
        joinpath(directory, "mass_matrix.mat"),
        assembly.mass,
        assembly.matrix_size,
    )
end
