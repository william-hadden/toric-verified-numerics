using IntervalArithmetic

if !isdefined(@__MODULE__, :InverseMetricBoxOracleV2)
    include(joinpath(@__DIR__, "..", "inverse_metric_box_bounds_v2.jl"))
end
if !isdefined(@__MODULE__, :local_metric_component_polynomials)
    include(joinpath(@__DIR__, "..", "util", "local_polynomials.jl"))
end

"""
    improved_polynomial_range(P)

Enclose a local power-basis polynomial on `[-1,1]^2`.  `P[p+1,q+1]`
is the interval coefficient of `x^p y^q`.  The constant coefficient is kept,
and the absolute values of all nonconstant coefficients are added as a
symmetric remainder.  The returned interval therefore contains every value
of the polynomial on the box, including all coefficient uncertainty.
"""
function improved_polynomial_range(P::AbstractMatrix)
    isempty(P) && throw(ArgumentError("A polynomial coefficient array must not be empty"))

    tail = zero(abs(P[1, 1]))
    for j in axes(P, 2), i in axes(P, 1)
        i == 1 && j == 1 && continue
        tail += abs(P[i, j])
    end

    radius = sup(tail)
    isfinite(radius) || error("The polynomial range has a non-finite remainder: $tail")
    return P[1, 1] + interval(-radius, radius)
end

_improved_interval_midpoint(x::Interval) = (inf(x) + sup(x)) / BigFloat(2)

function _improved_point_interval(x::BigFloat)
    return interval(x)
end

function _improved_interval_hull(values)
    isempty(values) && throw(ArgumentError("Cannot form the hull of an empty collection"))
    result = first(values)
    for value in Iterators.drop(values, 1)
        result = hull(result, value)
    end
    return result
end

function _improved_triangle_cartesian_hull(nodes, tri)
    length(tri) == 3 || throw(ArgumentError("A triangle must have exactly three vertices"))
    length(unique(tri)) == 3 || throw(ArgumentError("A triangle must have three distinct vertices"))
    all(index -> 1 <= index <= size(nodes, 1), tri) ||
        throw(BoundsError(nodes, collect(tri)))

    xbox = _improved_interval_hull([nodes[index, 1] for index in tri])
    ybox = _improved_interval_hull([nodes[index, 2] for index in tri])
    return (; xbox, ybox)
end

function _improved_physical_metric_components(U)
    # The stored Chebyshev coefficients use the reflected upper-square
    # coordinate.  Returning to the physical lower square reverses exactly one
    # coordinate, and hence changes the sign of the mixed components.
    return (; xx = U.xx, xy = -U.xy, yx = -U.yx, yy = U.yy)
end

function _improved_neumann_diagnostics(D, terms::Integer)
    d0 = (inf(D[1, 1]) + sup(D[1, 1])) / BigFloat(2)
    iszero(d0) && error("The local denominator polynomial has zero midpoint")

    remainder_polynomial = copy(D)
    remainder_polynomial[1, 1] -= interval(d0)
    ratio = sup(poly_abs_bound(remainder_polynomial)) / abs(d0)
    ratio < 1 || error(
        "The reciprocal Neumann series does not contract: ratio = $ratio. " *
        "Refine the mesh or change the local-polynomial parameters.",
    )
    remainder = ratio^(terms + 1) / (abs(d0) * (1 - ratio))
    return (; center = d0, ratio, remainder)
end

"""
    improved_inverse_metric_box_on_cartesian_box(
        oracle, xbox, ybox; deg=8, terms=8,
    )

Return a rigorous pointwise interval enclosure of the physical inverse metric
on the Cartesian box `xbox x ybox`.  The enclosure is obtained from the local
power-polynomial representation and its certified reciprocal Neumann series;
it is not a direct interval quotient.  Each polynomial is ranged by its
interval constant coefficient plus the symmetric sum of the absolute values
of all nonconstant coefficients.

The returned `G` is a symmetric `2x2` interval matrix.  Additional fields
record the local Chebyshev box, Neumann diagnostics, and the component boxes.
"""
function improved_inverse_metric_box_on_cartesian_box(
    oracle::InverseMetricBoxOracleV2,
    xbox::Interval,
    ybox::Interval;
    deg::Integer = 8,
    terms::Integer = 8,
)
    deg >= 0 || throw(ArgumentError("deg must be nonnegative"))
    terms >= 0 || throw(ArgumentError("terms must be nonnegative"))
    isfinite(inf(xbox)) && isfinite(sup(xbox)) ||
        throw(ArgumentError("xbox must be finite"))
    isfinite(inf(ybox)) && isfinite(sup(ybox)) ||
        throw(ArgumentError("ybox must be finite"))

    xcheb = interval(BigFloat(2)) * xbox - interval(BigFloat(1))
    ycheb = interval(BigFloat(2)) * ybox + interval(BigFloat(1))
    components = local_metric_component_polynomials(oracle, xcheb, ycheb; deg)
    neumann = _improved_neumann_diagnostics(components.D, terms)
    inverse_denominator = reciprocal_polynomial_neumann(
        components.D;
        deg,
        terms,
    )

    canonical_polynomials = (;
        xx = poly_mul(components.A11, inverse_denominator),
        xy = poly_mul(components.A12, inverse_denominator),
        yx = poly_mul(components.A12, inverse_denominator),
        yy = poly_mul(components.A22, inverse_denominator),
    )
    canonical_boxes = (;
        xx = improved_polynomial_range(canonical_polynomials.xx),
        xy = improved_polynomial_range(canonical_polynomials.xy),
        yx = improved_polynomial_range(canonical_polynomials.yx),
        yy = improved_polynomial_range(canonical_polynomials.yy),
    )
    physical_boxes = _improved_physical_metric_components(canonical_boxes)
    mixed_box = hull(physical_boxes.xy, physical_boxes.yx)
    G = [physical_boxes.xx mixed_box; mixed_box physical_boxes.yy]

    return (;
        xbox,
        ybox,
        xcheb,
        ycheb,
        G,
        neumann,
        canonical_component_boxes = canonical_boxes,
        physical_component_boxes = physical_boxes,
    )
end

"""
    improved_spd_certificate_2x2(A)

Certify that every symmetric point matrix represented by the symmetric `2x2`
interval matrix `A` is positive definite.  The test is exact for the interval
box: both diagonal lower endpoints must be positive and the lower endpoint of
`A[1,1]*A[2,2] - A[1,2]^2` must be positive.

Besides the Boolean result, return the rigorous diagonal and determinant
margins and a lower bound for the smallest eigenvalue of every represented
matrix.
"""
function improved_spd_certificate_2x2(A::AbstractMatrix{<:Interval})
    size(A) == (2, 2) || throw(ArgumentError("Expected a 2x2 interval matrix"))

    off_diagonal = hull(A[1, 2], A[2, 1])
    symmetric_box = [A[1, 1] off_diagonal; off_diagonal A[2, 2]]
    diagonal_lower = (inf(symmetric_box[1, 1]), inf(symmetric_box[2, 2]))
    determinant_range =
        symmetric_box[1, 1] * symmetric_box[2, 2] - off_diagonal^2
    determinant_lower = inf(determinant_range)
    spd = diagonal_lower[1] > 0 && diagonal_lower[2] > 0 && determinant_lower > 0

    lambda_lower = if spd
        a = interval(diagonal_lower[1])
        c = interval(diagonal_lower[2])
        b = interval(sup(abs(off_diagonal)))
        inf((a + c - sqrt((a - c)^2 + interval(BigFloat(4)) * b^2)) /
            interval(BigFloat(2)))
    else
        BigFloat(-Inf)
    end

    return (;
        spd,
        diagonal_lower,
        diagonal_margin = min(diagonal_lower...),
        determinant_range,
        determinant_lower,
        lambda_lower,
        symmetric_box,
    )
end

function _improved_point_matrix_intervals(A::AbstractMatrix{BigFloat})
    size(A) == (2, 2) || throw(ArgumentError("Expected a 2x2 point matrix"))
    return [
        _improved_point_interval(A[1, 1]) _improved_point_interval(A[1, 2])
        _improved_point_interval(A[2, 1]) _improved_point_interval(A[2, 2])
    ]
end

function _improved_midpoint_matrix(G::AbstractMatrix{<:Interval})
    off_diagonal = hull(G[1, 2], G[2, 1])
    c12 = _improved_interval_midpoint(off_diagonal)
    return BigFloat[
        _improved_interval_midpoint(G[1, 1]) c12
        c12 _improved_interval_midpoint(G[2, 2])
    ]
end

function _improved_symmetric_matrix_hull(matrices)
    isempty(matrices) && throw(ArgumentError("At least one interval matrix is required"))
    first_matrix = first(matrices)
    size(first_matrix) == (2, 2) || throw(ArgumentError("Expected 2x2 interval matrices"))

    diagonal_11 = first_matrix[1, 1]
    diagonal_22 = first_matrix[2, 2]
    off_diagonal = hull(first_matrix[1, 2], first_matrix[2, 1])
    for matrix in Iterators.drop(matrices, 1)
        size(matrix) == (2, 2) || throw(ArgumentError("Expected 2x2 interval matrices"))
        diagonal_11 = hull(diagonal_11, matrix[1, 1])
        diagonal_22 = hull(diagonal_22, matrix[2, 2])
        off_diagonal = hull(off_diagonal, matrix[1, 2])
        off_diagonal = hull(off_diagonal, matrix[2, 1])
    end
    return [diagonal_11 off_diagonal; off_diagonal diagonal_22]
end

function _improved_scaled_midpoint_matrix(C::AbstractMatrix{BigFloat}, theta::BigFloat)
    b11 = theta * C[1, 1]
    b12 = theta * C[1, 2]
    b22 = theta * C[2, 2]
    return BigFloat[b11 b12; b12 b22]
end

function _improved_residual_certificate(G, B)
    residual = G - _improved_point_matrix_intervals(B)
    return improved_spd_certificate_2x2(residual)
end

function _improved_split_interval(box::Interval)
    lower = inf(box)
    upper = sup(box)
    lower < upper || error("Cannot subdivide a zero-width interval $box")
    midpoint = (lower + upper) / BigFloat(2)
    lower < midpoint < upper || error(
        "The active BigFloat precision cannot split interval $box any further",
    )
    return interval(lower, midpoint), interval(midpoint, upper)
end

function _improved_adaptive_metric_cover(
    oracle,
    root_evaluation;
    deg::Integer,
    terms::Integer,
    max_subdivision_depth::Integer,
    minimum_subdivision_depth::Integer = 0,
)
    max_subdivision_depth >= 0 ||
        throw(ArgumentError("max_subdivision_depth must be nonnegative"))
    0 <= minimum_subdivision_depth <= max_subdivision_depth ||
        throw(ArgumentError(
            "minimum_subdivision_depth must lie in 0:max_subdivision_depth",
        ))
    leaves = Any[]

    function visit(evaluation, depth)
        G_certificate = improved_spd_certificate_2x2(evaluation.G)
        if G_certificate.spd && depth >= minimum_subdivision_depth
            push!(
                leaves,
                (;
                    xbox = evaluation.xbox,
                    ybox = evaluation.ybox,
                    depth,
                    G = evaluation.G,
                    G_certificate,
                    neumann = evaluation.neumann,
                ),
            )
            return
        end

        if depth >= max_subdivision_depth
            G_certificate.spd && error(
                "The requested minimum metric-box subdivision depth " *
                "$minimum_subdivision_depth exceeds the available maximum depth " *
                "$max_subdivision_depth.",
            )
            error(
                "The pointwise inverse-metric interval is not uniformly SPD on " *
                "leaf box $(evaluation.xbox) x $(evaluation.ybox) at maximum " *
                "subdivision depth $max_subdivision_depth: diagonal lower bounds = " *
                "$(G_certificate.diagonal_lower), determinant lower bound = " *
                "$(G_certificate.determinant_lower). Increase max_subdivision_depth, " *
                "refine the geometric mesh, or tighten deg/terms.",
            )
        end

        xchildren = _improved_split_interval(evaluation.xbox)
        ychildren = _improved_split_interval(evaluation.ybox)
        for child_ybox in ychildren, child_xbox in xchildren
            child = improved_inverse_metric_box_on_cartesian_box(
                oracle,
                child_xbox,
                child_ybox;
                deg,
                terms,
            )
            visit(child, depth + 1)
        end
    end

    visit(root_evaluation, 0)
    return leaves
end

function _improved_residual_certificates(G_leaves, B)
    return [_improved_residual_certificate(G, B) for G in G_leaves]
end

_improved_all_spd(certificates) = all(certificate -> certificate.spd, certificates)

function _improved_bisect_lower_matrix(G_leaves, C; iterations::Integer)
    iterations > 0 || throw(ArgumentError("bisection_iterations must be positive"))
    isempty(G_leaves) && throw(ArgumentError("At least one metric leaf is required"))

    zero_theta = BigFloat(0)
    one_theta = BigFloat(1)
    zero_matrix = _improved_scaled_midpoint_matrix(C, zero_theta)
    zero_certificates = _improved_residual_certificates(G_leaves, zero_matrix)
    _improved_all_spd(zero_certificates) || error(
        "An adaptive-cover leaf is not uniformly SPD before subtraction; " *
        "this indicates an internal cover-construction error.",
    )

    one_matrix = _improved_scaled_midpoint_matrix(C, one_theta)
    one_certificates = _improved_residual_certificates(G_leaves, one_matrix)
    if _improved_all_spd(one_certificates)
        return (;
            theta = one_theta,
            theta_upper = one_theta,
            B = one_matrix,
            residual_certificates = one_certificates,
            completed_iterations = 0,
        )
    end

    lower = zero_theta
    upper = one_theta
    best_matrix = zero_matrix
    best_certificates = zero_certificates
    completed_iterations = 0

    for iteration in 1:iterations
        candidate = (lower + upper) / BigFloat(2)
        (candidate == lower || candidate == upper) && break

        candidate_matrix = _improved_scaled_midpoint_matrix(C, candidate)
        candidate_certificates = _improved_residual_certificates(
            G_leaves,
            candidate_matrix,
        )
        if _improved_all_spd(candidate_certificates)
            lower = candidate
            best_matrix = candidate_matrix
            best_certificates = candidate_certificates
        else
            upper = candidate
        end
        completed_iterations = iteration
    end

    lower > 0 || error(
        "No positive certified scaling was found in $completed_iterations bisection " *
        "iterations. Increase bisection_iterations or refine the mesh.",
    )
    return (;
        theta = lower,
        theta_upper = upper,
        B = best_matrix,
        residual_certificates = best_certificates,
        completed_iterations,
    )
end

"""
    improved_inverse_metric_lower_bound_for_triangle(
        oracle, nodes, tri;
        deg=8, terms=8, bisection_iterations=80,
        max_subdivision_depth=0, minimum_subdivision_depth=0,
    )

Construct a rigorous constant Loewner lower bound for the inverse metric on the
Cartesian hull of one triangle.

The inverse metric is first represented by the same local-polynomial Neumann
enclosure used by the eigenvalue assembly.  Each resulting polynomial is
ranged on `[-1,1]^2` by retaining its interval constant coefficient and adding
the symmetric sum of the absolute nonconstant coefficients.  The mixed entry
is then sign-corrected for the physical lower square.

The Cartesian box is first evaluated coarsely.  If that interval family is not
uniformly SPD and `max_subdivision_depth > 0`, the box is split into four equal
child boxes, recursively and only where necessary.  The resulting SPD leaf
matrices cover the whole Cartesian hull.  Their componentwise symmetric hull
is the returned matrix `G`, and `C` is the actual symmetric point midpoint of
this cover hull.  `C` must itself certify as SPD.  The original one-box
evaluation is retained separately as `diagnostics.coarse_G`.

The routine bisects for the largest certified `theta` on its dyadic search grid
such that every leaf family in the cover remains SPD after subtracting the
same matrix `B = theta*C`.  `B` is formed from the actual rounded `BigFloat`
entries and is recertified, as is every leaf residual.  The default subdivision
depth is zero and therefore preserves the original one-box behavior.

Returns `G`, `C`, `theta`, `B`, a rigorous lower bound `alpha` for
`lambda_min(B)`, the minimum rigorous `spd_margin` over all leaf residuals,
and detailed cover diagnostics.  If subdivision was used, the componentwise
hull can contain artificial combinations drawn from different leaves, so no
claim is made that `G-B` itself is SPD; the rigorous Loewner claim is carried
by the complete leaf cover.

`minimum_subdivision_depth` may force one or more uniform initial subdivision
levels even when the parent interval is already SPD.  This removes interval
dependency from poorly conditioned but formally positive boxes.  It does not
weaken the certificate: the same rounded `B` is still recertified against every
leaf of the complete cover.
"""
function improved_inverse_metric_lower_bound_for_triangle(
    oracle::InverseMetricBoxOracleV2,
    nodes,
    tri;
    deg::Integer = 8,
    terms::Integer = 8,
    bisection_iterations::Integer = 80,
    max_subdivision_depth::Integer = 0,
    minimum_subdivision_depth::Integer = 0,
)
    deg >= 0 || throw(ArgumentError("deg must be nonnegative"))
    terms >= 0 || throw(ArgumentError("terms must be nonnegative"))
    max_subdivision_depth >= 0 ||
        throw(ArgumentError("max_subdivision_depth must be nonnegative"))
    0 <= minimum_subdivision_depth <= max_subdivision_depth ||
        throw(ArgumentError(
            "minimum_subdivision_depth must lie in 0:max_subdivision_depth",
        ))

    hull_box = _improved_triangle_cartesian_hull(nodes, tri)
    coarse = improved_inverse_metric_box_on_cartesian_box(
        oracle,
        hull_box.xbox,
        hull_box.ybox;
        deg,
        terms,
    )
    coarse_G = coarse.G
    coarse_G_certificate = improved_spd_certificate_2x2(coarse_G)

    leaves = try
        _improved_adaptive_metric_cover(
            oracle,
            coarse;
            deg,
            terms,
            max_subdivision_depth,
            minimum_subdivision_depth,
        )
    catch exception
        error("Failed to cover triangle $tri by uniformly SPD metric boxes: " *
              sprint(showerror, exception))
    end
    leaf_matrices = [leaf.G for leaf in leaves]
    G = _improved_symmetric_matrix_hull(leaf_matrices)
    G_certificate = improved_spd_certificate_2x2(G)
    C = _improved_midpoint_matrix(G)
    C_certificate = improved_spd_certificate_2x2(_improved_point_matrix_intervals(C))
    C_certificate.spd || error(
        "The symmetric midpoint C of the adaptive-cover inverse-metric hull is " *
        "not certified SPD on triangle $tri: diagonal lower bounds = " *
        "$(C_certificate.diagonal_lower), determinant lower bound = " *
        "$(C_certificate.determinant_lower). The constrained form B=theta*C " *
        "cannot produce an SPD lower matrix.",
    )
    search = _improved_bisect_lower_matrix(
        leaf_matrices,
        C;
        iterations = bisection_iterations,
    )
    B = search.B

    # Recreate point intervals from the actual, rounded BigFloat entries stored
    # in B.  No symbolic theta*C identity is used in either final certificate.
    B_certificate = improved_spd_certificate_2x2(_improved_point_matrix_intervals(B))
    B_certificate.spd || error(
        "The rounded lower matrix B is not certified SPD: " *
        "diagonal lower bounds = $(B_certificate.diagonal_lower), " *
        "determinant lower bound = $(B_certificate.determinant_lower).",
    )
    residual_certificates = _improved_residual_certificates(leaf_matrices, B)
    for (leaf_index, residual_certificate) in pairs(residual_certificates)
        residual_certificate.spd || error(
            "The rounded lower matrix B failed final Loewner recertification " *
            "on adaptive-cover leaf $leaf_index: diagonal lower bounds = " *
            "$(residual_certificate.diagonal_lower), determinant lower bound = " *
            "$(residual_certificate.determinant_lower).",
        )
    end

    alpha = B_certificate.lambda_lower
    spd_margin = minimum(
        residual_certificate.lambda_lower
        for residual_certificate in residual_certificates
    )
    alpha > 0 || error("The certified lower eigenvalue of B is not positive: $alpha")
    spd_margin > 0 || error(
        "The minimum certified SPD margin of the adaptive-cover residuals is " *
        "not positive: $spd_margin",
    )

    cover_used = length(leaves) > 1
    leaf_diagnostics = [
        (;
            xbox = leaf.xbox,
            ybox = leaf.ybox,
            depth = leaf.depth,
            G = leaf.G,
            G_certificate = leaf.G_certificate,
            neumann = leaf.neumann,
            residual_certificate = residual_certificates[index],
        )
        for (index, leaf) in pairs(leaves)
    ]

    diagnostics = (;
        cartesian_hull = hull_box,
        chebyshev_hull = (; x = coarse.xcheb, y = coarse.ycheb),
        polynomial_parameters = (;
            deg,
            terms,
            bisection_iterations,
            max_subdivision_depth,
            minimum_subdivision_depth,
        ),
        neumann = coarse.neumann,
        canonical_component_boxes = coarse.canonical_component_boxes,
        physical_component_boxes = coarse.physical_component_boxes,
        coarse_G,
        coarse_G_certificate,
        cover_G = G,
        cover_G_certificate = G_certificate,
        G_certificate,
        C_certificate,
        B_certificate,
        cover_used,
        leaf_count = length(leaves),
        max_leaf_depth = maximum(leaf.depth for leaf in leaves),
        leaves = leaf_diagnostics,
        residual_certificate = cover_used ? nothing : only(residual_certificates),
        residual_certificates,
        min_residual_spd_margin = spd_margin,
        theta_bracket = (search.theta, search.theta_upper),
        completed_bisection_iterations = search.completed_iterations,
    )

    return (;
        G,
        G_box = G,
        C,
        theta = search.theta,
        B,
        alpha,
        spd_margin,
        diagnostics,
    )
end

"""
    improved_inverse_metric_lower_bound_for_triangle(nodes, tri; oracle, kwargs...)

Convenience method which constructs an `InverseMetricBoxOracleV2` unless one is
provided.  Reuse an explicit oracle when processing many triangles.
"""
function improved_inverse_metric_lower_bound_for_triangle(
    nodes,
    tri;
    oracle = InverseMetricBoxOracleV2(),
    kwargs...,
)
    return improved_inverse_metric_lower_bound_for_triangle(
        oracle,
        nodes,
        tri;
        kwargs...,
    )
end
