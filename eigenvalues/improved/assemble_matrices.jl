using IntervalArithmetic
using LinearAlgebra

# The production assembly supplies the well-tested CR geometry, D6 quotient,
# sparse-dictionary assembly, and outward-rounded MATLAB writer.  The improved
# pipeline replaces only the mesh source and the local stiffness coefficient.
if !isdefined(@__MODULE__, :pdelta_mesh_data)
    include(joinpath(@__DIR__, "..", "assemble_matrices.jl"))
end
if !isdefined(@__MODULE__, :improved_pdelta_mesh_data)
    include(joinpath(@__DIR__, "triangulation.jl"))
end
if !isdefined(@__MODULE__, :improved_inverse_metric_lower_bound_for_triangle)
    include(joinpath(@__DIR__, "inverse_metric_box_bounds_v2.jl"))
end

const IMPROVED_LIU_CONSTANT = 1893 // 10000

"""Return a rigorous upper bound for the Euclidean diameter of one triangle."""
function improved_triangle_diameter_upper(nodes, tri)
    diameter = BigFloat(0)
    for (a, b) in ((1, 2), (2, 3), (3, 1))
        dx = nodes[tri[a], 1] - nodes[tri[b], 1]
        dy = nodes[tri[a], 2] - nodes[tri[b], 2]
        distance = sqrt(dx^2 + dy^2)
        diameter = max(diameter, sup(distance))
    end
    isfinite(diameter) && diameter > 0 || error(
        "Could not certify a positive finite diameter for triangle $tri",
    )
    return diameter
end

"""
Return a rigorous upper bound for the local Liu interpolation constant

    0.1893 * h_T / sqrt(alpha_T),

where `diameter_upper >= h_T` and `alpha_lower <= lambda_min(B_T)`.
"""
function improved_local_liu_constant_upper(diameter_upper, alpha_lower)
    alpha_lower > 0 || error("The lower ellipticity bound must be positive")
    factor = interval(BigFloat(numerator(IMPROVED_LIU_CONSTANT))) /
        interval(BigFloat(denominator(IMPROVED_LIU_CONSTANT)))
    value = factor * interval(BigFloat(diameter_upper)) /
        sqrt(interval(BigFloat(alpha_lower)))
    upper = sup(value)
    isfinite(upper) && upper > 0 || error("The local Liu constant is not positive and finite")
    return upper
end

"""Return the constant-coefficient local CR stiffness matrix for `B`."""
function improved_local_cr_stiffness_matrix(nodes, tri, B)
    size(B) == (2, 2) || throw(ArgumentError("B must be a 2x2 matrix"))
    area = triangle_double_area(nodes, tri) / interval(BigFloat(2))
    inf(area) > 0 || error("Triangle $tri is not certified positively oriented")

    # Reuse the production local formula by supplying the exact integral
    # \int_T B dx = |T| B as an outward-rounded interval matrix.
    integrated = (;
        xx = area * interval(B[1, 1]),
        xy = area * interval(B[1, 2]),
        yx = area * interval(B[2, 1]),
        yy = area * interval(B[2, 2]),
    )
    return local_cr_stiffness_matrix_from_integrals(nodes, tri, integrated)
end

"""
Add a local interval block without asking for an indeterminate interval Boolean.

The production helper uses `iszero`, which is appropriate for its historical
entries but throws when outward rounding encloses a theoretical zero by a tiny
interval such as `[-eps,eps]`.  Only an interval exactly equal to zero may be
skipped here.
"""
function improved_addblock!(global_matrix, dofs, local_matrix)
    for a in 1:3, b in 1:3
        value = local_matrix[a, b]
        isequal_interval(value, zero(value)) && continue
        key = (dofs[a], dofs[b])
        global_matrix[key] = get(global_matrix, key, zero(value)) + value
    end
    return global_matrix
end

"""
Construct and certify the comparison matrix `B_T` on every element.

If the coarse pointwise metric box is not uniformly SPD, the optional adaptive
Cartesian subcover may split only that enclosure.  Failure after the configured
subdivision depth is deliberately fatal: it signals that the geometric mesh or
the pointwise enclosure must be refined before Liu's comparison can be used.
"""
function improved_element_certificates(
    mesh;
    deg::Integer = 8,
    terms::Integer = 8,
    bisection_iterations::Integer = 80,
    max_subdivision_depth::Integer = 0,
    quality_liu_target = nothing,
    max_quality_subdivision_depth::Integer = 0,
    progress_every::Integer = 100,
    parallel::Bool = false,
)
    max_quality_subdivision_depth >= 0 || throw(
        ArgumentError("max_quality_subdivision_depth must be nonnegative"),
    )
    if !isnothing(quality_liu_target)
        quality_liu_target > 0 || throw(ArgumentError("quality_liu_target must be positive"))
        max_quality_subdivision_depth > 0 || throw(ArgumentError(
            "max_quality_subdivision_depth must be positive when quality_liu_target is set",
        ))
    end
    oracle = InverseMetricBoxOracleV2()
    certificates = Vector{Any}(undef, length(mesh.triangles))

    function certify_one!(index)
        tri = mesh.triangles[index]
        diameter_upper = improved_triangle_diameter_upper(mesh.nodes, tri)
        target = if isnothing(quality_liu_target)
            nothing
        elseif quality_liu_target isa Rational
            inf(
                interval(BigFloat(numerator(quality_liu_target))) /
                interval(BigFloat(denominator(quality_liu_target))),
            )
        else
            BigFloat(quality_liu_target, RoundDown)
        end
        last_quality_depth = isnothing(target) ? 0 : max_quality_subdivision_depth
        best_certificate = nothing
        best_liu_constant = BigFloat(Inf)
        best_quality_depth = 0
        last_exception_message = nothing
        available_subdivision_depth = max(
            max_subdivision_depth,
            max_quality_subdivision_depth,
        )

        for quality_depth in 0:last_quality_depth
            certificate = try
                improved_inverse_metric_lower_bound_for_triangle(
                    oracle,
                    mesh.nodes,
                    tri;
                    deg,
                    terms,
                    bisection_iterations,
                    max_subdivision_depth = available_subdivision_depth,
                    minimum_subdivision_depth = quality_depth,
                )
            catch exception
                last_exception_message = sprint(showerror, exception)
                continue
            end

            liu_constant = improved_local_liu_constant_upper(
                diameter_upper,
                certificate.alpha,
            )
            if liu_constant < best_liu_constant
                best_certificate = certificate
                best_liu_constant = liu_constant
                best_quality_depth = quality_depth
            end
            (isnothing(target) || best_liu_constant <= target) && break
        end

        isnothing(best_certificate) && error(
            "Failed to construct B_T for triangle $index = $tri through " *
            "quality subdivision depth $last_quality_depth: " *
            something(last_exception_message, "no valid cover was returned"),
        )

        if !isnothing(target) && best_liu_constant > target
            error(
                "Triangle $index = $tri cannot meet quality_liu_target=$target " *
                "through minimum subdivision depth $max_quality_subdivision_depth; " *
                "best certified local Liu constant is $best_liu_constant. " *
                "Refine the geometric mesh or increase the quality depth." *
                (isnothing(last_exception_message) ? "" :
                 " The last attempted cover failed with: $last_exception_message"),
            )
        end

        certificates[index] = merge(
            best_certificate,
            (;
                triangle_index = index,
                diameter_upper,
                liu_constant_upper = best_liu_constant,
                quality_minimum_subdivision_depth = best_quality_depth,
            ),
        )
        return nothing
    end

    if parallel && Threads.nthreads() > 1
        completed = Threads.Atomic{Int}(0)
        next_report = Threads.Atomic{Int}(progress_every > 0 ? progress_every : typemax(Int))
        Threads.@threads for index in eachindex(mesh.triangles)
            certify_one!(index)
            done = Threads.atomic_add!(completed, 1) + 1
            if progress_every > 0
                target = next_report[]
                if done >= target && Threads.atomic_cas!(next_report, target, target + progress_every) == target
                    println("certified B_T on $done / $(length(mesh.triangles)) triangles")
                    flush(stdout)
                end
            end
        end
    else
        for index in eachindex(mesh.triangles)
            certify_one!(index)
            if progress_every > 0 &&
               (index % progress_every == 0 || index == length(mesh.triangles))
                println("certified B_T on $index / $(length(mesh.triangles)) triangles")
                flush(stdout)
            end
        end
    end

    return certificates
end

"""
Assemble the D6-invariant CR mass matrix and the stiffness matrix for the
piecewise-constant certified coefficient `B_h`.
"""
function assemble_improved_d6_invariant_cr_matrices(mesh, certificates)
    length(certificates) == length(mesh.triangles) || throw(
        DimensionMismatch("There must be one B_T certificate per triangle"),
    )

    edges, triangle_edges = build_cr_edges(mesh.triangles)
    quotient = d6_sector_edge_quotient(edges, mesh.node_keys)
    local_mass = [local_cr_mass_matrix(mesh.nodes, tri) for tri in mesh.triangles]
    local_stiffness = [
        improved_local_cr_stiffness_matrix(
            mesh.nodes,
            mesh.triangles[index],
            certificates[index].B,
        )
        for index in eachindex(mesh.triangles)
    ]

    mass = Dict{Tuple{Int,Int},Interval{BigFloat}}()
    stiffness = Dict{Tuple{Int,Int},Interval{BigFloat}}()
    for index in eachindex(mesh.triangles)
        dofs = ntuple(a -> quotient[triangle_edges[index][a]], 3)
        improved_addblock!(mass, dofs, local_mass[index])
        improved_addblock!(stiffness, dofs, local_stiffness[index])
    end

    liu_constant_upper = maximum(certificate.liu_constant_upper for certificate in certificates)
    worst_liu_triangle = argmax([certificate.liu_constant_upper for certificate in certificates])
    return (;
        matrix_size = maximum(quotient),
        mass,
        stiffness,
        edges,
        triangle_edges,
        quotient,
        liu_constant_upper,
        worst_liu_triangle,
        certificates,
    )
end

"""Generate a mesh, certify every B_T, and assemble the comparison problem."""
function assemble_improved_problem(;
    delta = IMPROVED_DEFAULT_DELTA,
    N::Integer = IMPROVED_DEFAULT_N,
    refinement_level::Integer = 0,
    boundary_layers::Integer = 1,
    corner_refinement_level::Integer = 0,
    corner_layers::Integer = 1,
    deg::Integer = 8,
    terms::Integer = 8,
    bisection_iterations::Integer = 80,
    max_subdivision_depth::Integer = 0,
    quality_liu_target = nothing,
    max_quality_subdivision_depth::Integer = 0,
    progress_every::Integer = 100,
    parallel::Bool = false,
)
    mesh = improved_pdelta_mesh_data(
        ;
        delta,
        N,
        refinement_level,
        boundary_layers,
        corner_refinement_level,
        corner_layers,
    )
    certificates = improved_element_certificates(
        mesh;
        deg,
        terms,
        bisection_iterations,
        max_subdivision_depth,
        quality_liu_target,
        max_quality_subdivision_depth,
        progress_every,
        parallel,
    )
    assembly = assemble_improved_d6_invariant_cr_matrices(mesh, certificates)
    return (; mesh, assembly)
end

"""
Apply Liu's comparison monotonically to a verified lower endpoint of the FEM
eigenvalue.  The return value is a rigorous lower endpoint for the smooth
inset eigenvalue with the true inverse metric.
"""
function improved_liu_lower_bound(lambda_fem_lower, liu_constant_upper)
    lambda_fem_lower > 0 || error("The verified FEM eigenvalue lower bound must be positive")
    liu_constant_upper > 0 || error("The Liu constant upper bound must be positive")
    lambda = interval(BigFloat(lambda_fem_lower))
    constant = interval(BigFloat(liu_constant_upper))
    value = lambda / (interval(BigFloat(1)) + constant^2 * lambda)
    lower = inf(value)
    lower > 0 || error("Liu's transformed lower bound is not positive")
    return lower
end

"""Return a non-rigorous midpoint eigenvalue for diagnostics only."""
function improved_midpoint_eigenvalue(assembly)
    n = assembly.matrix_size
    K = zeros(Float64, n, n)
    M = zeros(Float64, n, n)
    for ((i, j), value) in symmetrize_entries(assembly.stiffness)
        K[i, j] = Float64((inf(value) + sup(value)) / BigFloat(2))
    end
    for ((i, j), value) in symmetrize_entries(assembly.mass)
        M[i, j] = Float64((inf(value) + sup(value)) / BigFloat(2))
    end
    return eigen(Symmetric(K), Symmetric(M)).values[2]
end
