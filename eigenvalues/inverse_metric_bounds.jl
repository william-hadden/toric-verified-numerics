using IntervalArithmetic

include(joinpath(@__DIR__, "..", "bound_residual", "util", "io.jl"))
include(joinpath(@__DIR__, "..", "bound_residual", "util", "chebyshev_algebra.jl"))
include(joinpath(@__DIR__, "..", "bound_residual", "util", "problem.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "io.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "progress.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "interval_helpers.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "chebyshev_interval_evaluation.jl"))
include(joinpath(@__DIR__, "util", "local_polynomials.jl"))

"""Cached coefficient data used to enclose the inverse metric on local boxes."""
struct InverseMetricOracle
    h11
    h12
    h22
    canonical
end

"""Load the approximate metric and truncate its Hessian at `coefficient_degree`."""
function InverseMetricOracle(coefficient_degree::Integer)
    coefficient_degree >= 0 || throw(ArgumentError("coefficient_degree must be nonnegative"))
    coefficients = load_metric_coeffs_csv(METRIC_U0_PATH)
    derivatives = compute_second_derivatives(coefficients)
    return InverseMetricOracle(
        truncate_coeffs_with_tail(derivatives.uxx, coefficient_degree),
        truncate_coeffs_with_tail(derivatives.uxy, coefficient_degree),
        truncate_coeffs_with_tail(derivatives.uyy, coefficient_degree),
        factored_canonical_metric_coeffs(),
    )
end

"""Enclose a local power polynomial on the square `[-1,1]^2`."""
function polynomial_range(P)
    tail = zero(abs(P[1, 1]))
    for j in axes(P, 2), i in axes(P, 1)
        i == 1 && j == 1 && continue
        tail += abs(P[i, j])
    end
    radius = sup(tail)
    isfinite(radius) || error("The polynomial range has a non-finite radius")
    return P[1, 1] + interval(-radius, radius)
end

"""Return the contraction ratio for the reciprocal Neumann expansion."""
function reciprocal_contraction_ratio(D)
    center = (inf(D[1, 1]) + sup(D[1, 1])) / BigFloat(2)
    iszero(center) && return BigFloat(Inf)
    remainder = copy(D)
    remainder[1, 1] -= interval(center)
    return sup(poly_abs_bound(remainder) / interval(abs(center)))
end

"""Enclose the physical inverse metric on one Cartesian box, if it contracts."""
function inverse_metric_box(oracle, xbox, ybox, degree, neumann_terms)
    xcheb = interval(BigFloat(2)) * xbox - interval(BigFloat(1))
    ycheb = interval(BigFloat(2)) * ybox + interval(BigFloat(1))
    components = local_metric_component_polynomials(oracle, xcheb, ycheb; deg = degree)
    ratio = reciprocal_contraction_ratio(components.D)
    ratio < 1 || return nothing
    inverse_denominator = reciprocal_polynomial_neumann(
        components.D;
        deg = degree,
        terms = neumann_terms,
    )
    xx = polynomial_range(poly_mul(components.A11, inverse_denominator))
    xy = -polynomial_range(poly_mul(components.A12, inverse_denominator))
    yy = polynomial_range(poly_mul(components.A22, inverse_denominator))
    return (; xbox, ybox, G = [xx xy; xy yy], ratio)
end

"""Certify positive definiteness of every matrix in a symmetric 2-by-2 box."""
function metric_spd_certificate(A)
    size(A) == (2, 2) || throw(ArgumentError("Expected a 2-by-2 matrix"))
    off_diagonal = hull(A[1, 2], A[2, 1])
    diagonal_lower = (inf(A[1, 1]), inf(A[2, 2]))
    determinant_lower = inf(A[1, 1] * A[2, 2] - off_diagonal^2)
    spd = minimum(diagonal_lower) > 0 && determinant_lower > 0
    lambda_lower = BigFloat(-Inf)
    if spd
        a, c = interval(diagonal_lower[1]), interval(diagonal_lower[2])
        b = interval(sup(abs(off_diagonal)))
        discriminant = sqrt((a - c)^2 + interval(BigFloat(4)) * b^2)
        lambda_lower = inf((a + c - discriminant) / interval(BigFloat(2)))
    end
    return (; spd, diagonal_lower, determinant_lower, lambda_lower)
end

"""Return the Cartesian hull of one mesh triangle."""
function triangle_cartesian_hull(nodes, tri)
    xbox = nodes[tri[1], 1]
    ybox = nodes[tri[1], 2]
    for vertex in tri[2:3]
        xbox = hull(xbox, nodes[vertex, 1])
        ybox = hull(ybox, nodes[vertex, 2])
    end
    return (; xbox, ybox)
end

"""Split a nondegenerate interval into two outward-rounded halves."""
function split_interval(box)
    lower, upper = inf(box), sup(box)
    lower < upper || error("Cannot subdivide the zero-width interval $box")
    midpoint = (lower + upper) / BigFloat(2)
    lower < midpoint < upper || error("The active precision cannot split $box")
    return interval(lower, midpoint), interval(midpoint, upper)
end

"""Recursively append SPD metric boxes at or below the forced cover depth."""
function cover_metric_box!(leaves, oracle, xbox, ybox, depth, forced_depth, degree, terms)
    evaluation = inverse_metric_box(oracle, xbox, ybox, degree, terms)
    certificate = isnothing(evaluation) ? nothing : metric_spd_certificate(evaluation.G)
    if !isnothing(certificate) && certificate.spd && depth >= forced_depth
        push!(leaves, (;
            xbox = evaluation.xbox,
            ybox = evaluation.ybox,
            G = evaluation.G,
            depth,
        ))
        return leaves
    end
    for ychild in split_interval(ybox), xchild in split_interval(xbox)
        cover_metric_box!(
            leaves,
            oracle,
            xchild,
            ychild,
            depth + 1,
            forced_depth,
            degree,
            terms,
        )
    end
    return leaves
end

"""Cover a triangle's Cartesian hull by certified SPD inverse-metric boxes."""
function metric_cover(oracle, nodes, tri, forced_depth, degree, terms)
    forced_depth >= 0 || throw(ArgumentError("forced_depth must be nonnegative"))
    box = triangle_cartesian_hull(nodes, tri)
    leaves = NamedTuple[]
    cover_metric_box!(leaves, oracle, box.xbox, box.ybox, 0, forced_depth, degree, terms)
    return leaves
end

"""Return the symmetric componentwise hull of certified metric matrices."""
function metric_matrix_hull(leaves)
    isempty(leaves) && error("A metric cover must have at least one leaf")
    xx, yy = leaves[1].G[1, 1], leaves[1].G[2, 2]
    xy = hull(leaves[1].G[1, 2], leaves[1].G[2, 1])
    for leaf in leaves[2:end]
        xx = hull(xx, leaf.G[1, 1])
        yy = hull(yy, leaf.G[2, 2])
        xy = hull(xy, leaf.G[1, 2])
        xy = hull(xy, leaf.G[2, 1])
    end
    return [xx xy; xy yy]
end

"""Return the symmetric point midpoint of a 2-by-2 interval matrix."""
function midpoint_matrix(G)
    two = BigFloat(2)
    xx = (inf(G[1, 1]) + sup(G[1, 1])) / two
    yy = (inf(G[2, 2]) + sup(G[2, 2])) / two
    mixed_box = hull(G[1, 2], G[2, 1])
    mixed = (inf(mixed_box) + sup(mixed_box)) / two
    return BigFloat[xx mixed; mixed yy]
end

"""Embed a point matrix as a matrix of point intervals."""
function point_interval_matrix(A)
    size(A) == (2, 2) || throw(ArgumentError("Expected a 2-by-2 matrix"))
    return [interval(A[1, 1]) interval(A[1, 2]); interval(A[2, 1]) interval(A[2, 2])]
end

"""Certify every leaf residual after subtracting a constant point matrix."""
function residual_certificates(leaves, B)
    point_B = point_interval_matrix(B)
    return [metric_spd_certificate(leaf.G - point_B) for leaf in leaves]
end

"""Find a certified dyadic scaling of the cover midpoint by bisection."""
function bisect_lower_matrix(leaves, midpoint, steps)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    lower, upper = BigFloat(0), BigFloat(1)
    best = zeros(BigFloat, 2, 2)
    all(certificate -> certificate.spd, residual_certificates(leaves, midpoint)) &&
        return midpoint
    for _ in 1:steps
        theta = (lower + upper) / BigFloat(2)
        candidate = theta * midpoint
        if all(certificate -> certificate.spd, residual_certificates(leaves, candidate))
            lower, best = theta, candidate
        else
            upper = theta
        end
    end
    lower > 0 || error("Bisection did not find a positive metric lower matrix")
    return best
end

"""Construct and recertify a constant Loewner lower matrix on one element."""
function inverse_metric_lower_bound(
    oracle,
    nodes,
    tri,
    forced_depth,
    degree,
    terms,
    bisection_steps,
)
    leaves = metric_cover(oracle, nodes, tri, forced_depth, degree, terms)
    midpoint = midpoint_matrix(metric_matrix_hull(leaves))
    metric_spd_certificate(point_interval_matrix(midpoint)).spd ||
        error("The metric-cover midpoint is not positive definite on triangle $tri")
    B = bisect_lower_matrix(leaves, midpoint, bisection_steps)
    B_certificate = metric_spd_certificate(point_interval_matrix(B))
    B_certificate.spd || error("The rounded lower matrix is not positive definite")
    residuals = residual_certificates(leaves, B)
    all(certificate -> certificate.spd, residuals) ||
        error("The rounded lower matrix failed leaf-wise recertification")
    B_certificate.lambda_lower > 0 || error("The certified ellipticity is not positive")
    return (; B, alpha = B_certificate.lambda_lower, leaves)
end
