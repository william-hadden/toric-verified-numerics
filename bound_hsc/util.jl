using IntervalArithmetic

struct PolytopeBox
    xlo::Rational{BigInt}
    xhi::Rational{BigInt}
    ylo::Rational{BigInt}
    yhi::Rational{BigInt}

    function PolytopeBox(xlo, xhi, ylo, yhi)
        xlo = bigq(xlo)
        xhi = bigq(xhi)
        ylo = bigq(ylo)
        yhi = bigq(yhi)

        xlo < xhi || error("Need xlo < xhi")
        ylo < yhi || error("Need ylo < yhi")

        new(xlo, xhi, ylo, yhi)
    end
end

function refine_boxes(boxes::AbstractVector{PolytopeBox})
    refined = PolytopeBox[]
    half = big(1) // big(2)
    for B in boxes
        xm = (B.xlo + B.xhi) * half
        ym = (B.ylo + B.yhi) * half
        append!(refined, (
            PolytopeBox(B.xlo, xm, B.ylo, ym),
            PolytopeBox(xm, B.xhi, B.ylo, ym),
            PolytopeBox(B.xlo, xm, ym, B.yhi),
            PolytopeBox(xm, B.xhi, ym, B.yhi),
        ))
    end
    return refined
end

function evaluate_geometry_on_box(derivatives, X, Y)
    require_guaranteed((X, Y), "HSC box coordinates")
    D = evaluate_hsc_coeffs_on_box(
        derivatives.D,
        X,
        Y,
    )

    # Division is only rigorous if D is bounded away from zero.
    require_guaranteed(D, "inverse-metric denominator")
    inf(D) > 0 || return nothing

    deriv_num = derivatives.deriv_num

    uinv = [
        evaluate_hsc_coeffs_on_box(
            deriv_num[(i, j, (0, 0))],
            X,
            Y,
        ) / D
        for i in 1:2, j in 1:2
    ]

    d2uinv = [
        evaluate_hsc_coeffs_on_box(
            deriv_num[
                (
                    i,
                    j,
                    derivative_exponent_multiindex_2d(a, b),
                )
            ],
            X,
            Y,
        ) / D^3
        for i in 1:2, j in 1:2,
            a in 1:2, b in 1:2
    ]

    require_guaranteed(uinv, "inverse metric")
    require_guaranteed(d2uinv, "second inverse-metric derivatives")
    return (; uinv, d2uinv)
end

"""Evaluate a global `[0,1]^2` Chebyshev series by recentering it on a box."""
function evaluate_hsc_coeffs_on_box(coeffs, X, Y)
    if inf(X) == sup(X) && inf(Y) == sup(Y)
        return evaluate_coeffs_at_point(coeffs, X, Y)
    end
    TX = exact(2) * X - exact(1)
    TY = exact(1) - exact(2) * Y
    return local_coeff_sum_centered_enclosure_cheb_2d(coeffs, TX, TY)
end

function hsc_enclosure_sum(terms, pdeg::Integer)
    pdeg > 0 || error("HSC coefficient truncation requires pdeg > 0")
    isempty(terms) && return truncated_coeff_enclosure(
        cheb_constant(interval_constant(0)),
        pdeg,
    )
    return foldl((left, right) -> enclosure_add(left, right, pdeg), terms)
end

function hsc_derivative_enclosures(derivatives, pdeg::Integer)
    pdeg > 0 || error("HSC coefficient truncation requires pdeg > 0")
    from_pack = haskey(derivatives, :pack)
    source = from_pack ? derivatives.pack.data : derivatives.deriv_num
    return Dict(
        key => from_pack ?
            truncate_enclosure_coeffs(value.coeffs, value.tail, pdeg) :
            truncated_coeff_enclosure(value, pdeg)
        for (key, value) in source
    )
end

"""
Build the direction-independent coefficient-space HSC contractions.

If `u^{ij}_{,ab} = N^{ij}_{ab}/D0^3`, then
`|Riem|^2 = R2_num/D0^6` with all tensor contractions performed before
interval evaluation.
"""
function prepare_hsc_base_coeffs(derivatives; pdeg::Integer = derivatives.pdeg)
    data = hsc_derivative_enclosures(derivatives, pdeg)
    D0 = derivatives.D

    terms = Any[]
    for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        ab = derivative_exponent_multiindex_2d(a, b)
        cd = derivative_exponent_multiindex_2d(c, d)
        push!(
            terms,
            enclosure_mul(data[(a, b, cd)], data[(c, d, ab)], pdeg),
        )
    end

    R2_num = enclosure_to_coeffs(hsc_enclosure_sum(terms, pdeg))
    return (; D0, R2_num, pdeg)
end

"""
Build the coefficient-space numerator for Q in a fixed constant direction.

Writing `u^{ij}=A^{ij}/D0`, define `S = A^{ij}xi_i xi_j`.  The full
curvature contraction is `Q = P/(D0^3*S^2)`.
"""
function prepare_hsc_direction_coeffs(
    derivatives,
    base,
    ξ;
    pdeg::Integer = base.pdeg,
)
    data = hsc_derivative_enclosures(derivatives, pdeg)

    scaled(coeffs, c) = enclosure_scale(coeffs, interval_constant(c), pdeg)
    A(i, j) = data[(i, j, (0, 0))]

    S_enclosure = hsc_enclosure_sum(
        [scaled(A(i, j), ξ[i] * ξ[j]) for i in 1:2, j in 1:2],
        pdeg,
    )
    W = [
        hsc_enclosure_sum([scaled(A(a, i), ξ[i]) for i in 1:2], pdeg)
        for a in 1:2
    ]
    H = [
        hsc_enclosure_sum(
            [
                scaled(
                    data[(j, l, derivative_exponent_multiindex_2d(a, b))],
                    ξ[j] * ξ[l],
                )
                for j in 1:2, l in 1:2
            ],
            pdeg,
        )
        for a in 1:2, b in 1:2
    ]

    P_terms = Any[]
    for a in 1:2, b in 1:2
        WW = enclosure_mul(W[a], W[b], pdeg)
        push!(P_terms, enclosure_mul(WW, H[a, b], pdeg))
    end
    P_enclosure = enclosure_scale(
        hsc_enclosure_sum(P_terms, pdeg),
        -interval_half(),
        pdeg,
    )

    return (;
        base...,
        S = enclosure_to_coeffs(S_enclosure),
        P = enclosure_to_coeffs(P_enclosure),
        ξ,
    )
end

function compact_curvature_ranges_on_box(prepared, B::PolytopeBox)
    X, Y = box_intervals(B)
    D0 = evaluate_hsc_coeffs_on_box(prepared.D0, X, Y)
    inf(D0) > 0 || return nothing

    S = evaluate_hsc_coeffs_on_box(prepared.S, X, Y)
    inf(S) > 0 || return nothing

    P = evaluate_hsc_coeffs_on_box(prepared.P, X, Y)
    R2_num = evaluate_hsc_coeffs_on_box(prepared.R2_num, X, Y)

    require_guaranteed(D0, "compact inverse-metric denominator")
    require_guaranteed(S, "compact direction numerator")
    require_guaranteed(P, "compact Q numerator")
    require_guaranteed(R2_num, "compact Riemann-squared numerator")

    Q = P / (D0^3 * S^2)
    R2 = R2_num / D0^6
    return require_guaranteed((; Q, R2), "compact curvature ranges")
end

"""
Enclose Q₁(V) on a box.

Here V is the g₁-unit normalization of the real direction represented
by ξ. Returns `nothing` if positivity of the squared length cannot be
certified on the box.
"""
function compute_Q_rigorously(geometry, ξ, X, Y)
    geometry === nothing && return nothing
    uinv = geometry.uinv
    d2uinv = geometry.d2uinv

    ξbox = ntuple(i -> eval_component(ξ[i], X, Y), 2)

    # D = u^{ij} ξᵢ ξⱼ = |W|²_g.
    D = sum(
        uinv[i, j] * ξbox[i] * ξbox[j]
        for i in 1:2, j in 1:2
    )

    require_guaranteed(D, "direction squared length")

    # Interval dependency may prevent positivity on a large box.
    # The caller should then subdivide.
    if inf(D) <= 0
        return nothing
    end

    # wₐ = Σᵢ u^{ai} ξᵢ
    w = [
        sum(
            uinv[a, i] * ξbox[i]
            for i in 1:2
        )
        for a in 1:2
    ]

    # hₐᵦ = Σⱼₗ u^{jl}_{ab} ξⱼ ξₗ
    h = [
        sum(
            d2uinv[j, l, a, b] * ξbox[j] * ξbox[l]
            for j in 1:2, l in 1:2
        )
        for a in 1:2, b in 1:2
    ]

    numerator = -sum(
        w[a] * w[b] * h[a, b]
        for a in 1:2, b in 1:2
    ) / exact(2)

    # This is Q₁(V), because V = W/sqrt(D) is unit.
    Q = numerator / D^2
    return require_guaranteed(Q, "holomorphic sectional curvature term")
end


"""Enclose `Q₁(V)` and `|Riem₁|²` on a polytope box."""
function curvature_ranges_on_box(derivatives, ξ, B::PolytopeBox; prepared = nothing)
    prepared === nothing || return compact_curvature_ranges_on_box(prepared, B)

    X, Y = box_intervals(B)
    geometry = evaluate_geometry_on_box(derivatives, X, Y)
    geometry === nothing && return nothing

    Q = compute_Q_rigorously(geometry, ξ, X, Y)
    Q === nothing && return nothing

    R2 = require_guaranteed(
        sum(
            geometry.d2uinv[a, b, c, d] * geometry.d2uinv[c, d, a, b]
            for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        ),
        "squared Riemann norm",
    )

    return (; Q, R2)
end

function diagnose_hsc_box_failure(derivatives, ξ, B::PolytopeBox)
    X, Y = box_intervals(B)
    inverse_denominator = evaluate_hsc_coeffs_on_box(derivatives.D, X, Y)

    if inf(inverse_denominator) <= 0
        return (
            reason = :inverse_denominator_not_certified_positive,
            box = B,
            X,
            Y,
            inverse_denominator,
        )
    end

    geometry = evaluate_geometry_on_box(derivatives, X, Y)
    geometry === nothing && return (
        reason = :geometry_not_certified,
        box = B,
        X,
        Y,
        inverse_denominator,
    )

    ξbox = ntuple(i -> eval_component(ξ[i], X, Y), 2)
    direction_squared = sum(
        geometry.uinv[i, j] * ξbox[i] * ξbox[j]
        for i in 1:2, j in 1:2
    )
    second_derivative_abs_upper = maximum(
        sup(abs(entry)) for entry in geometry.d2uinv
    )

    return (
        reason = inf(direction_squared) <= 0 ?
                 :direction_squared_not_certified_positive :
                 :curvature_range_not_certified,
        box = B,
        X,
        Y,
        inverse_denominator,
        direction_squared,
        inverse_metric = geometry.uinv,
        second_derivative_abs_upper,
    )
end


function boxes_have_disjoint_interiors(boxes::AbstractVector{PolytopeBox})
    for i in eachindex(boxes), j in (i + 1):lastindex(boxes)
        A, B = boxes[i], boxes[j]
        x_overlap = max(A.xlo, B.xlo) < min(A.xhi, B.xhi)
        y_overlap = max(A.ylo, B.ylo) < min(A.yhi, B.yhi)
        x_overlap && y_overlap && return false
    end
    return true
end

"""
Bound all terms in Proposition 5.6 over a fixed union of boxes.

The boxes must have disjoint interiors. Their union represents the
base region B, while the set in the manifold is B × T².
"""
function bound_proposition_5_6_margin(
    derivatives,
    boxes::AbstractVector{PolytopeBox},
    ξ,
    eta,
    rho,
    ;
    box_progress = nothing,
    prepared = nothing,
    pdeg::Integer = derivatives.pdeg,
)
    isempty(boxes) && error("U must contain at least one box")
    boxes_have_disjoint_interiors(boxes) ||
        error("HSC boxes must have disjoint interiors")

    η = as_big_interval(eta)
    ρ = as_big_interval(rho)

    inf(η) >= 0 || error("Need η ≥ 0")
    sup(η) < 1 || error("Need η < 1")
    inf(ρ) >= 0 || error("Need ρ ≥ 0")

    if prepared === nothing && haskey(derivatives, :pack)
        base = prepare_hsc_base_coeffs(derivatives; pdeg)
        prepared = prepare_hsc_direction_coeffs(derivatives, base, ξ; pdeg)
    end

    Q_base_integral = interval(BigFloat, 0)
    R2_base_integral = interval(BigFloat, 0)
    base_area = interval(BigFloat, 0)

    last_box_report = time_ns()
    for (box_index, B) in pairs(boxes)
        if box_progress !== nothing &&
           (time_ns() - last_box_report) / 1.0e9 >= 30
            box_progress((index = box_index, total = length(boxes)))
            last_box_report = time_ns()
        end
        ranges = curvature_ranges_on_box(derivatives, ξ, B; prepared)

        if ranges === nothing
            return (
                valid = false,
                certified = false,
                reason = :direction_not_certified,
                failing_box = B,
                diagnostic = diagnose_hsc_box_failure(derivatives, ξ, B),
            )
        end

        area = exact_big_interval((B.xhi - B.xlo) * (B.yhi - B.ylo))

        Q_base_integral += area * ranges.Q
        R2_base_integral += area * ranges.R2
        base_area += area
    end

    # Rigorous enclosure of 4π².
    torus_factor =
        exact(4) * interval(BigFloat, π)^2

    Q_integral =
        torus_factor * Q_base_integral

    R2_integral =
        torus_factor * R2_base_integral

    volume_U =
        torus_factor * base_area

    # The exact integral of |Riem|² is nonnegative.
    # A negative lower endpoint can arise from interval overestimation.
    R2_upper = sup(R2_integral)

    R2_upper >= 0 || error(
        "Upper bound for ∫|Riem|² is negative; check the curvature formula"
    )

    R2_nonnegative =
        interval(BigFloat, 0, R2_upper)

    Rnorm =
        sqrt(R2_nonnegative)

    correction =
        sqrt(volume_U) *
        (η * Rnorm + (one(η) + η) * ρ)

    # Desired inequality is:
    #
    # Q_integral + correction < 0.
    margin =
        Q_integral + correction

    require_guaranteed(
        (Q_integral, R2_integral, Rnorm, volume_U, correction, margin),
        "Proposition 5.6 bound",
    )

    return (
        valid = true,
        certified = sup(margin) < 0,
        margin = margin,
        margin_upper = sup(margin),
        Q_integral = Q_integral,
        R2_integral = R2_integral,
        Rnorm = Rnorm,
        volume = volume_U,
        correction = correction,
        rhs = -correction,
        boxes = length(boxes),
    )
end

bigq(x::Integer) =
    BigInt(x) // BigInt(1)

bigq(x::Rational) =
    BigInt(numerator(x)) // BigInt(denominator(x))

function canonical_direction(p::Integer, q::Integer)
    p == 0 && q == 0 && return nothing

    p = BigInt(p)
    q = BigInt(q)

    g = gcd(abs(p), abs(q))
    p ÷= g
    q ÷= g

    if p < 0 || (p == 0 && q < 0)
        p = -p
        q = -q
    end

    return (bigq(p), bigq(q))
end

"""
Generate all primitive real projective directions (p,q), up to sign,
with max(|p|,|q|) ≤ order.
"""
function primitive_directions(order::Integer)
    order >= 1 || error("Need direction order ≥ 1")

    directions = Set{
        Tuple{Rational{BigInt},Rational{BigInt}}
    }()

    for p in -order:order, q in -order:order
        ξ = canonical_direction(p, q)
        ξ === nothing || push!(directions, ξ)
    end

    return sort!(collect(directions))
end

function rational_gridpoint(lo, hi, k::Integer, n::Integer)
    0 <= k <= n || error("Grid index outside range")
    λ = BigInt(k) // BigInt(n)
    return lo + (hi - lo) * λ
end

function cell_centre(lo, hi, k::Integer, n::Integer)
    # Cell k, where k = 0,...,n-1.
    λ = BigInt(2k + 1) // BigInt(2n)
    return lo + (hi - lo) * λ
end

"""
Find pointwise negative-curvature seeds on a coarse grid.

The scores are heuristic only.
"""
function generate_hsc_seeds(
    derivatives,
    domain::PolytopeBox;
    grid::Integer = 12,
    direction_order::Integer = 5,
    directions_per_cell::Integer = 2,
    max_seeds::Integer = 100,
    seed_cutoff::Real = 0,
    progress = nothing,
)
    grid >= 1 || error("Need grid >= 1")
    directions_per_cell >= 1 || error("Need directions_per_cell >= 1")
    max_seeds >= 1 || error("Need max_seeds >= 1")

    directions = primitive_directions(direction_order)
    seeds = NamedTuple[]

    for iy in 0:(grid - 1), ix in 0:(grid - 1)
        x = cell_centre(domain.xlo, domain.xhi, ix, grid)
        y = cell_centre(domain.ylo, domain.yhi, iy, grid)

        X = exact_big_interval(x)
        Y = exact_big_interval(y)

        geometry =
            evaluate_geometry_on_box(derivatives, X, Y)

        local_seeds = NamedTuple[]

        for ξ in directions
            Q = compute_Q_rigorously(geometry, ξ, X, Y)
            if Q !== nothing
                # Midpoint is used only to rank candidates.
                score = Float64((inf(Q) + sup(Q)) / exact(2))

                if score < seed_cutoff
                    push!(
                        local_seeds,
                        (
                            ix = ix,
                            iy = iy,
                            ξ = ξ,
                            point_Q = Q,
                            score = score,
                        ),
                    )
                end
            end

            progress === nothing || advance_progress!(progress)
        end

        sort!(local_seeds; by = seed -> seed.score)

        nkeep = min(directions_per_cell, length(local_seeds))
        append!(seeds, local_seeds[1:nkeep])
    end

    sort!(seeds; by = seed -> seed.score)

    if length(seeds) > max_seeds
        resize!(seeds, max_seeds)
    end

    return seeds
end

"""
Create a grid-aligned rectangle around a seed cell.

rx and ry are the number of additional cells included on each side.
"""
function candidate_box_from_seed(
    domain::PolytopeBox,
    ix::Integer,
    iy::Integer,
    grid::Integer,
    rx::Integer,
    ry::Integer,
)
    ixlo = max(0, ix - rx)
    ixhi = min(grid, ix + rx + 1)
    iylo = max(0, iy - ry)
    iyhi = min(grid, iy + ry + 1)

    return PolytopeBox(
        rational_gridpoint(
            domain.xlo, domain.xhi, ixlo, grid
        ),
        rational_gridpoint(
            domain.xlo, domain.xhi, ixhi, grid
        ),
        rational_gridpoint(
            domain.ylo, domain.yhi, iylo, grid
        ),
        rational_gridpoint(
            domain.ylo, domain.yhi, iyhi, grid
        ),
    )
end

"""
Generate rectangular candidate regions from pointwise seeds.
"""
function generate_hsc_candidates(
    derivatives,
    domain::PolytopeBox;
    grid::Integer = 12,
    direction_order::Integer = 5,
    directions_per_cell::Integer = 2,
    max_seeds::Integer = 100,
    seed_cutoff::Real = 0,
    progress = nothing,

    # Include small, medium, large, and elongated regions.
    shapes = (
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (1, 2),
        (2, 1),
        (2, 4),
        (4, 2),
    ),
)
    seeds = generate_hsc_seeds(
        derivatives,
        domain;
        grid = grid,
        direction_order = direction_order,
        directions_per_cell = directions_per_cell,
        max_seeds = max_seeds,
        seed_cutoff = seed_cutoff,
        progress = progress,
    )

    candidates = NamedTuple[]
    seen = Set()

    for seed in seeds, (rx, ry) in shapes
        B = candidate_box_from_seed(
            domain,
            seed.ix,
            seed.iy,
            grid,
            rx,
            ry,
        )

        key = (
            B.xlo, B.xhi,
            B.ylo, B.yhi,
            seed.ξ,
        )

        key in seen && continue
        push!(seen, key)

        push!(
            candidates,
            (
                region = PolytopeBox[B],
                ξ = seed.ξ,
                seed_score = seed.score,
                seed_cell = (seed.ix, seed.iy),
                shape = (rx, ry),
            ),
        )
    end

    sort!(candidates; by = c -> c.seed_score)
    return candidates
end

function refine_region(
    boxes::AbstractVector{PolytopeBox},
    depth::Integer,
)
    refined = collect(boxes)

    for _ in 1:depth
        refined = refine_boxes(refined)
    end

    return refined
end

"""
Refine promising candidates in parallel as a beam search.

`beam_width` controls how many valid candidates survive each round.
`invalid_keep` retains some candidates whose direction positivity was
not certified on a coarse box; subdivision may resolve that.
"""
function find_compatible_U_V(
    derivatives,
    eta,
    rho,
    candidates;
    pdeg::Integer = derivatives.pdeg,
    screen_depth::Integer = 0,
    maxdepth::Integer = 7,
    beam_width::Integer = 20,
    invalid_keep::Integer = 5,
    progress = nothing,
)
    isempty(candidates) && return (
        found = false,
        reason = :no_candidates,
        best = nothing,
        search_is_exhaustive = false,
    )

    progress_message!(
        progress,
        "Step 6: building direction-independent coefficient contractions",
    )
    base_coeffs = prepare_hsc_base_coeffs(derivatives; pdeg)
    progress_message!(progress, "Step 6: coefficient contractions ready")
    direction_coeffs = Dict{Any, Any}()
    prepared_direction(ξ) = get!(direction_coeffs, ξ) do
        prepare_hsc_direction_coeffs(derivatives, base_coeffs, ξ; pdeg)
    end

    states = [
        (
            candidate = candidate,
            boxes = refine_region(
                candidate.region,
                screen_depth,
            ),
        )
        for candidate in candidates
    ]

    best = nothing
    best_unresolved = nothing

    format_box(B::PolytopeBox) =
        "[$(B.xlo), $(B.xhi)] × [$(B.ylo), $(B.yhi)]"

    function valid_candidate_message(item)
        candidate = item.candidate
        proof = item.proof
        return "best valid candidate: direction $(candidate.ξ), seed cell " *
               "$(candidate.seed_cell), shape $(candidate.shape), " *
               "$(length(item.boxes)) boxes, upper margin $(proof.margin_upper), " *
               "Q integral $(proof.Q_integral), correction $(proof.correction)"
    end

    function unresolved_candidate_message(item)
        candidate = item.candidate
        diagnostic = item.proof.diagnostic
        message = "best unresolved candidate: direction $(candidate.ξ), seed cell " *
                  "$(candidate.seed_cell), shape $(candidate.shape), seed score " *
                  "$(candidate.seed_score), failing box $(format_box(diagnostic.box)), " *
                  "reason $(diagnostic.reason), inverse denominator " *
                  "$(diagnostic.inverse_denominator)"

        if haskey(diagnostic, :direction_squared)
            message *= ", direction squared $(diagnostic.direction_squared), " *
                       "inverse metric $(diagnostic.inverse_metric), " *
                       "max |second inverse derivative| " *
                       "$(diagnostic.second_derivative_abs_upper)"
        end
        return message
    end

    for depth in screen_depth:maxdepth
        evaluated = NamedTuple[]
        total_boxes = sum(length(state.boxes) for state in states)
        progress_message!(
            progress,
            "Step 6 depth $depth/$maxdepth: evaluating $(length(states)) " *
            "candidates across $total_boxes boxes",
        )

        for (index, state) in pairs(states)
            if depth >= 3 || index == 1 || index % 10 == 0
                progress_message!(
                    progress,
                    "Step 6 depth $depth/$maxdepth: starting candidate " *
                    "$index/$(length(states)) with $(length(state.boxes)) boxes, " *
                    "direction $(state.candidate.ξ)",
                )
            end
            proof = bound_proposition_5_6_margin(
                derivatives,
                state.boxes,
                state.candidate.ξ,
                eta,
                rho;
                prepared = prepared_direction(state.candidate.ξ),
                pdeg,
                box_progress = event -> progress_message!(
                    progress,
                    "Step 6 depth $depth/$maxdepth, candidate " *
                    "$index/$(length(states)): box $(event.index)/$(event.total), " *
                    "direction $(state.candidate.ξ)",
                ),
            )

            item = (
                candidate = state.candidate,
                boxes = state.boxes,
                proof = proof,
                depth = depth,
            )

            if proof.valid && proof.certified
                return (
                    found = true,
                    candidate = state.candidate,
                    proof = proof,
                    depth = depth,
                    boxes = state.boxes,
                )
            end

            if proof.valid
                if best === nothing || proof.margin_upper < best.proof.margin_upper
                    best = item
                end
            end

            push!(evaluated, item)

            if progress !== nothing
                event = (
                    stage = :candidate_refinement,
                    depth = depth,
                    index = index,
                    total = length(states),
                    best_margin = best === nothing ?
                        Inf : best.proof.margin_upper,
                )

                if applicable(progress, event)
                    progress(event)
                else
                    advance_progress!(progress)
                end
            end
        end

        valid = [
            item for item in evaluated
            if item.proof.valid
        ]

        invalid = [
            item for item in evaluated
            if !item.proof.valid
        ]

        sort!(
            valid;
            by = item -> item.proof.margin_upper,
        )

        sort!(
            invalid;
            by = item -> item.candidate.seed_score,
        )

        isempty(invalid) || (best_unresolved = first(invalid))

        nkeep_valid =
            min(beam_width, length(valid))

        nkeep_invalid =
            min(invalid_keep, length(invalid))

        candidate_details = if best !== nothing
            valid_candidate_message(best)
        elseif !isempty(invalid)
            unresolved_candidate_message(first(invalid))
        else
            "no valid or unresolved candidate remains"
        end
        progress_message!(
            progress,
            "Step 6 depth $depth complete: $(length(valid)) valid, " *
            "$(length(invalid)) unresolved; $candidate_details",
        )

        depth == maxdepth && break

        survivors = NamedTuple[]

        append!(survivors,valid[1:nkeep_valid],)

        append!(survivors,invalid[1:nkeep_invalid],)

        isempty(survivors) && break

        states = [
            (
                candidate = item.candidate,
                boxes = refine_boxes(item.boxes),
            )
            for item in survivors
        ]
    end

    return (
        found = false,
        reason = :heuristic_search_exhausted,
        best = best,
        best_unresolved,
        # Directions, seed cells, region shapes, and beam survivors are all
        # finite heuristic samples.
        search_is_exhaustive = false,
    )
end
