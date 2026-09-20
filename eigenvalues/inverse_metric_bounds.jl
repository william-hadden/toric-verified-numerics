using IntervalArithmetic

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_truncation.jl"))
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
    coefficients = load_rational_coeffs_csv(U0_PATH)
    # derivatives = compute_second_derivatives(coefficients)
    derivatives = build_derivative_pack(coefficients)
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

"""Enclose the physical inverse metric on one Cartesian box, if it contracts."""
function inverse_metric_box(oracle, xbox, ybox, degree, neumann_terms)
    xcheb = interval(BigFloat(2)) * xbox - interval(BigFloat(1))
    ycheb = interval(BigFloat(2)) * ybox + interval(BigFloat(1))
    components = local_metric_component_polynomials(oracle, xcheb, ycheb; deg = degree)
    reciprocal = reciprocal_polynomial_neumann(
        components.D;
        deg = degree,
        terms = neumann_terms,
    )
    if isnothing(reciprocal)
        return nothing
    end
    inverse_denominator, ratio = reciprocal.polynomial, reciprocal.ratio
    xx = polynomial_range(poly_mul(components.A11, inverse_denominator))
    xy = -polynomial_range(poly_mul(components.A12, inverse_denominator))
    yy = polynomial_range(poly_mul(components.A22, inverse_denominator))
    return (; xbox, ybox, G = [xx xy; xy yy], ratio)
end

"""
Return a rigorous lower bound for the smallest symmetric eigenvalue in `A`.

Use the lower endpoints of the diagonal entries and the largest possible
absolute off-diagonal entry in the explicit eigenvalue formula for a symmetric
2-by-2 matrix. A positive result certifies every symmetric matrix in `A` as
positive definite.
"""
function symmetric_eigenvalue_lower_bound(A)
    a = interval(inf(A[1, 1]))
    c = interval(inf(A[2, 2]))
    b = interval(sup(abs(hull(A[1, 2], A[2, 1]))))
    discriminant = sqrt((a - c)^2 + interval(BigFloat(4)) * b^2)
    return inf((a + c - discriminant) / interval(BigFloat(2))) # smaller value in quadratic formula
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
    midpoint = (lower + upper) / BigFloat(2)
    lower < midpoint < upper || error("The active precision cannot split $box")
    return interval(lower, midpoint), interval(midpoint, upper)
end

"""Recursively cover a box by certified leaves no shallower than `min_depth`."""
function cover_metric_box!(leaves, oracle, xbox, ybox, depth, min_depth, degree, terms)
    evaluation = inverse_metric_box(oracle, xbox, ybox, degree, terms)
    certified = !isnothing(evaluation) && symmetric_eigenvalue_lower_bound(evaluation.G) > 0
    if certified && depth >= min_depth
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
            min_depth,
            degree,
            terms,
        )
    end
    return leaves
end

"""Cover a triangle by certified metric boxes of depth at least `min_depth`."""
function metric_cover(oracle, nodes, tri, min_depth, degree, terms)
    box = triangle_cartesian_hull(nodes, tri)
    leaves = NamedTuple[]
    cover_metric_box!(leaves, oracle, box.xbox, box.ybox, 0, min_depth, degree, terms)
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
    xx = (inf(G[1, 1]) + sup(G[1, 1])) / BigFloat(2)
    yy = (inf(G[2, 2]) + sup(G[2, 2])) / BigFloat(2)
    mixed_box = hull(G[1, 2], G[2, 1])
    mixed = (inf(mixed_box) + sup(mixed_box)) / BigFloat(2)
    return BigFloat[xx mixed; mixed yy]
end

"""Return smallest-eigenvalue lower bounds for all leaf residuals `G - B`."""
function residual_eigenvalue_lower_bounds(leaves, B)
    point_B = interval.(B)
    return [symmetric_eigenvalue_lower_bound(leaf.G - point_B) for leaf in leaves]
end

"""
Search for a certified lower matrix of the form `B = theta * midpoint`.

The scalar `theta` is searched in `[0, 1]`. A candidate is accepted when every
interval residual `leaf.G - B` has a strictly positive eigenvalue lower bound;
this certifies that `B` lies below the inverse metric throughout the triangle
in the Loewner order. The search retains the largest certified value of `theta`
encountered during `steps` bisections and does not subdivide the metric boxes.
"""
function search_lower_matrix_scaling(leaves, midpoint, steps)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    lower, upper = BigFloat(0), BigFloat(1)
    best = zeros(BigFloat, 2, 2)
    if all(bound -> bound > 0, residual_eigenvalue_lower_bounds(leaves, midpoint))
        return midpoint
    end
    for _ in 1:steps
        theta = (lower + upper) / BigFloat(2)
        candidate = theta * midpoint
        if all(bound -> bound > 0, residual_eigenvalue_lower_bounds(leaves, candidate))
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
    min_depth,
    degree,
    terms,
    bisection_steps,
)
    leaves = metric_cover(oracle, nodes, tri, min_depth, degree, terms)

    if min_depth >= 5
        lambda = minimum(symmetric_eigenvalue_lower_bound(leaf.G) for leaf in leaves)
        lambda > 0 || error("The identity lower matrix is not positive definite on triangle $tri")

        # Use a tiny strict shrink so the residual check can prove G - B > 0.
        lambda = prevfloat(lambda)

        B = [
            lambda        zero(lambda)
            zero(lambda) lambda
        ]

        alpha = symmetric_eigenvalue_lower_bound(interval.(B))
        alpha > 0 || error("The identity lower matrix has nonpositive alpha")

        residuals = residual_eigenvalue_lower_bounds(leaves, B)
        all(bound -> bound > 0, residuals) ||
            error("The identity lower matrix failed leaf-wise recertification")

        return (; B, alpha, leaves)
    end

    midpoint = midpoint_matrix(metric_matrix_hull(leaves))
    symmetric_eigenvalue_lower_bound(interval.(midpoint)) > 0 ||
        error("The metric-cover midpoint is not positive definite on triangle $tri")
    B = search_lower_matrix_scaling(leaves, midpoint, bisection_steps)
    alpha = symmetric_eigenvalue_lower_bound(interval.(B))
    alpha > 0 || error("The rounded lower matrix is not positive definite")
    residuals = residual_eigenvalue_lower_bounds(leaves, B)
    all(bound -> bound > 0, residuals) ||
        error("The rounded lower matrix failed leaf-wise recertification")
    return (; B, alpha, leaves)
end
