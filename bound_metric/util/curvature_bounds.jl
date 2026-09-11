function truncated_coeff_enclosure(coeffs::AbstractMatrix{<:Number}, pdeg::Integer = 20)
    trunc = truncate_coeffs_with_tail(coeffs, pdeg)
    return (; coeffs = trunc.coeffs, tail = trunc.tail, pdeg = trunc.pdeg)
end

function truncate_enclosure_coeffs(coeffs::AbstractMatrix{<:Number}, tail, pdeg::Integer = 20)
    trunc = truncate_coeffs_with_tail(coeffs, pdeg)
    
    old_tail = abs(tail)
    new_tail = abs(trunc.tail)

    return (; coeffs = trunc.coeffs, tail = old_tail + new_tail, pdeg = trunc.pdeg)
end

"""
Add two truncated Chebyshev enclosures in coefficient space.

Both inputs are expected to have the form `(; coeffs, tail, pdeg)`. The retained
coefficient arrays are added, the existing tail bounds are summed, and the
result is truncated again to `pdeg`, with any newly discarded coefficients added
to the returned tail.
"""
function enclosure_add(f, g, pdeg::Integer = 20)
    return truncate_enclosure_coeffs(cheb_add(f.coeffs, g.coeffs), f.tail + g.tail, pdeg)
end

function enclosure_sub(f, g, pdeg::Integer = 20)
    return truncate_enclosure_coeffs(cheb_sub(f.coeffs, g.coeffs), f.tail + g.tail, pdeg)
end

"""
Scale a truncated Chebyshev enclosure by a scalar.

`f` is an enclosure `(; coeffs, tail, pdeg)`. The retained coefficients are
scaled by `c`, while the tail bound is scaled by `abs(c)` (or the supremum of
`abs(c)` when `c` is an interval). The result is truncated again to `pdeg`,
adding any newly discarded coefficient mass to the returned tail.
"""
function enclosure_scale(f, c, pdeg::Integer = 20)
    c_abs = interval(sup(abs(c))) 
    return truncate_enclosure_coeffs(cheb_scale(f.coeffs, c), c_abs * f.tail, pdeg)
end

"""
Multiply two truncated Chebyshev enclosures in coefficient space.

The retained coefficient arrays are multiplied with rigorous Chebyshev
arithmetic. The returned tail bounds all products involving at least one
discarded part:

    f_tail * ||g_coeffs|| + g_tail * ||f_coeffs|| + f_tail * g_tail.

The product is then truncated to `pdeg`, adding the new truncation tail to the
returned enclosure. If `progress` is supplied, it is advanced after the retained
coefficient multiplication.
"""
function enclosure_mul(f, g, pdeg::Integer = 20; progress = nothing)
    product_coeffs = cheb_mul_fast(f.coeffs, g.coeffs)
    advance_progress!(progress)
    g_coeffs_interval = intervalize_coefficients(g.coeffs)
    f_coeffs_interval = intervalize_coefficients(f.coeffs)
    product_tail =
        f.tail * interval(sup(chebyshev_coeff_sup_bound(g_coeffs_interval))) +
        g.tail * interval(sup(chebyshev_coeff_sup_bound(f_coeffs_interval))) +
        f.tail * g.tail

    return truncate_enclosure_coeffs(product_coeffs, product_tail, pdeg)
end

"""
Convert a truncated Chebyshev enclosure into a coefficient matrix.

The enclosure `f` has the form `(; coeffs, tail, pdeg)`. This helper returns a
plain coefficient array by widening the constant coefficient by `[-tail, tail]`,
so evaluating the returned series still rigorously encloses the original
function represented by `f`.
"""
function enclosure_to_coeffs(f)
    return inflate_constant_mode!(f.coeffs, f.tail)
end


"""
Compute the derviatve u^{ij}_I from u^{ij} = A/D (where A is a S^2(R) matrix) using the quotient rule to split into rational 
function. Then compute sums and products in numerator using pdeg truncation.
where I is some multiindex 
I = (i_1, i_2, ..., i_k), where each index takes value 1 or 2 corresponding to 
x and y derivatives
"""
# ------------------------------------------------------------
# Exact derivative cache
# ------------------------------------------------------------

function cached_exact_derivative!(cache::Dict{Tuple, Any}, coeffs::AbstractMatrix, I)
    key = Tuple(I)
    if !haskey(cache, key)
        derivative = coeffs
        for a in key
            a == 1 && (derivative = differentiate_coeffs_x(derivative); continue)
            a == 2 && (derivative = differentiate_coeffs_y(derivative); continue)
            error("Multiindex entries must be 1 or 2")
        end
        cache[key] = derivative
    end
    return cache[key]
end


# ------------------------------------------------------------
# Formal numerator expression
#
# A term represents:
#
#     coeff * ∂_{A_I} A * ∏_J ∂_J D
#
# D_Is is a vector of multiindices J.
# J = () means undifferentiated D.
# ------------------------------------------------------------

function normalize_terms(terms)
    acc = Dict{Any, BigInt}()

    for t in terms
        D_key = Tuple(sort([Tuple(J) for J in t.D_Is]))
        key = (Tuple(t.A_I), D_key)
        acc[key] = get(acc, key, big(0)) + t.coeff
    end

    out = []

    for ((A_I, D_key), coeff) in acc
        coeff == 0 && continue
        push!(out, (;
            coeff,
            A_I = Tuple(A_I),
            D_Is = Tuple[collect(D_key)...],
        ))
    end

    return out
end

function differentiate_formal_term(t, a::Integer)
    out = []

    push!(out, (;
        coeff = t.coeff,
        A_I = (t.A_I..., a),
        D_Is = Tuple[t.D_Is...],
    ))

    for j in eachindex(t.D_Is)
        new_D_Is = Tuple[t.D_Is...]
        new_D_Is[j] = (new_D_Is[j]..., a)

        push!(out, (;
            coeff = t.coeff,
            A_I = t.A_I,
            D_Is = new_D_Is,
        ))
    end

    return out
end

function quotient_numerator_formal_terms(I)
    # N_empty = A
    terms = [(; coeff = big(1), A_I = (), D_Is = Tuple[])]

    for (r, a) in enumerate(I)
        a == 1 || a == 2 || error("Multiindex entries must be 1 or 2")

        new_terms = []

        for t in terms
            # D * ∂_a N
            for dt in differentiate_formal_term(t, a)
                push!(new_terms, (;
                    coeff = dt.coeff,
                    A_I = dt.A_I,
                    D_Is = [dt.D_Is... , ()],
                ))
            end

            # - r * N * D_a
            push!(new_terms, (;
                coeff = -r * t.coeff,
                A_I = t.A_I,
                D_Is = [t.D_Is... , (a,)],
            ))
        end

        terms = normalize_terms(new_terms)
    end

    return terms
end

function formal_term_to_enclosure(
    t,
    A_coeffs::AbstractMatrix,
    D_coeffs::AbstractMatrix,
    A_cache::Dict{Tuple, Any},
    D_cache::Dict{Tuple, Any};
    pdeg::Integer = 20,
    progress = nothing,
)
    A_I = cached_exact_derivative!(A_cache, A_coeffs, t.A_I)
    out = truncated_coeff_enclosure(A_I, pdeg)

    for J in t.D_Is
        D_J = cached_exact_derivative!(D_cache, D_coeffs, J)
        D_J_enc = truncated_coeff_enclosure(D_J, pdeg)
        out = enclosure_mul(out, D_J_enc, pdeg; progress)
    end

    if t.coeff != 1
        out = enclosure_scale(out, exact(t.coeff), pdeg)
    end

    return out
end

"""
Compute a rigorous truncated numerator enclosure for a derivative of `A / D`.

Given Chebyshev coefficient arrays `A_coeffs` and `D_coeffs`, and a derivative
multiindex `I`, return an enclosure for the numerator `N_I` in

    ∂_I(A / D) = N_I / D^(length(I) + 1).

The returned value is an enclosure `(; coeffs, tail, pdeg)`: `coeffs` contains
the truncated Chebyshev coefficients for `N_I`, while `tail` rigorously bounds
all discarded coefficient mass introduced by truncation and products. The
formal quotient-rule expansion is generated symbolically from `I`, then each
term is evaluated with cached exact derivatives of `A` and `D`.
"""
function quotient_derivative_numerator_enclosure(
    A_coeffs::AbstractMatrix,
    D_coeffs::AbstractMatrix,
    I;
    pdeg::Integer = 20,
    progress = nothing,
)
    terms = quotient_numerator_formal_terms(I)

    A_cache = Dict{Tuple, Any}(() => A_coeffs)
    D_cache = Dict{Tuple, Any}(() => D_coeffs)

    isempty(terms) && error("No numerator terms produced")

    out = formal_term_to_enclosure(
        terms[1],
        A_coeffs,
        D_coeffs,
        A_cache,
        D_cache;
        pdeg,
        progress,
    )

    for t in terms[2:end]
        term_enc = formal_term_to_enclosure(
            t,
            A_coeffs,
            D_coeffs,
            A_cache,
            D_cache;
            pdeg,
            progress,
        )

        out = enclosure_add(out, term_enc, pdeg)
    end

    return out
end

"""
Compute numerator enclosures for derivatives of the inverse metric entries.

For each inverse metric component `u^{ij} = A^{ij} / D` and each 2D
derivative multiindex `a = (a1, a2)` with total order at most `k`, compute a
rigorous truncated Chebyshev enclosure for the numerator `N^{ij}_a` in

    ∂_a u^{ij} = N^{ij}_a / D^(a1 + a2 + 1).

The result stores these enclosures in `data[(i, j, a)]`, where each value is
the `(; coeffs, tail, pdeg)` enclosure returned by
`quotient_derivative_numerator_enclosure`. Symmetry is used for the off-diagonal
inverse metric entry, so `(1, 2)` and `(2, 1)` both use `inverse_coeffs.A12`.
"""
function compute_inverse_derivative_numerator_pack(
    inverse_coeffs;
    k::Integer,
    pdeg::Integer = 20,
    progress = nothing,
)
    k >= 0 || error("Maximum derivative order k must be >= 0")
    pdeg >= 0 || error("Inverse derivative numerator pack requires pdeg >= 0")

    A = Dict(
        (1, 1) => inverse_coeffs.A11,
        (1, 2) => inverse_coeffs.A12,
        (2, 1) => inverse_coeffs.A12,
        (2, 2) => inverse_coeffs.A22,
    )

    D = inverse_coeffs.D

    data = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()

    for i in 1:2, j in 1:2
        for total in 0:k, a1 in 0:total
            a = (a1, total - a1)
            I = Tuple(vcat(fill(1, a[1]), fill(2, a[2])))

            data[(i, j, a)] = quotient_derivative_numerator_enclosure(
                A[(i, j)],
                D,
                I;
                pdeg,
                progress,
            )
        end
    end

    return (;
        max_derivative_order = k,
        data,
    )
end

"""
Build coefficient-space Ricci numerator data from inverse metric coefficients.

For each inverse metric entry `u^{ij} = A^{ij} / D`, second derivatives have
the form

    partial_ab u^{ij} = N^{ij}_{ab} / D^3.

This routine first computes the second-derivative numerator pack, then forms
the four Ricci endomorphism numerators

    R11_num, R12_num, R21_num, R22_num,

so that `Rij = Rij_num / D^3`. It also forms the numerator of the pointwise
endomorphism norm squared

    ||Ric||^2 = 1/2 * (R11^2 + 2 R12 R21 + R22^2)

entirely in coefficient-enclosure space before any subdivision evaluation.
Therefore `ricci_norm_squared_num / D^6` encloses `||Ric||^2`.
"""
function compute_ricci_numerators_truncated_coefficient_space(
    inverse_coeffs;
    pdeg::Integer = 20,
    progress = nothing,
)
    pdeg > 0 || error("Truncated Ricci numerator construction requires pdeg > 0")

    pack = compute_inverse_derivative_numerator_pack(
        inverse_coeffs;
        k = 2,
        pdeg,
        progress,
    )

    U = pack.data
    D = inverse_coeffs.D

    R11_enc = enclosure_add(U[(1, 1, (2, 0))], U[(1, 2, (1, 1))], pdeg)
    R12_enc = enclosure_add(U[(1, 1, (1, 1))], U[(1, 2, (0, 2))], pdeg)
    R21_enc = enclosure_add(U[(1, 2, (2, 0))], U[(2, 2, (1, 1))], pdeg)
    R22_enc = enclosure_add(U[(1, 2, (1, 1))], U[(2, 2, (0, 2))], pdeg)

    offdiag = enclosure_mul(R12_enc, R21_enc, pdeg; progress)
    offdiag = enclosure_scale(offdiag, interval_constant(2), pdeg)

    ricci_norm_squared_num = enclosure_add(
        enclosure_mul(R11_enc, R11_enc, pdeg; progress),
        offdiag,
        pdeg,
    )
    ricci_norm_squared_num = enclosure_add(
        ricci_norm_squared_num,
        enclosure_mul(R22_enc, R22_enc, pdeg; progress),
        pdeg,
    )
    ricci_norm_squared_num = enclosure_scale(ricci_norm_squared_num, interval_half(), pdeg)

    return (;
        R11_num = enclosure_to_coeffs(R11_enc),
        R12_num = enclosure_to_coeffs(R12_enc),
        R21_num = enclosure_to_coeffs(R21_enc),
        R22_num = enclosure_to_coeffs(R22_enc),
        ricci_norm_squared_num = enclosure_to_coeffs(ricci_norm_squared_num),
        D,
        pdeg,
        pack,
    )
end

function derivative_exponent_multiindex_2d(indices::Vararg{Int})
    a1 = count(==(1), indices)
    a2 = count(==(2), indices)

    a1 + a2 == length(indices) || error("Derivative indices must be 1 or 2")

    return (a1, a2)
end


"""
Build inverse-metric derivative numerator components up to order `k`.

For each inverse metric component `u^{ij} = A^{ij} / D` and each multiindex
`a` with `|a| <= k`, this returns `deriv_num[(i, j, a)]`, a coefficient array
for the numerator in

    partial_a u^{ij} = deriv_num[(i, j, a)] / D^(|a| + 1).

When `k == 3`, it additionally builds the coefficient-space numerator for
`||nabla Riem||^2`, whose denominator is `D^9`. When `k >= 4`, it additionally
builds the coefficient-space numerator for `||nabla^2 Riem||^2`, whose
denominator is `D^12`. All contractions for these norms are done with
`enclosure_add` and `enclosure_mul` before local interval subdivision.
"""
function compute_inverse_derivative_numerator_components_truncated_coeff_space(
    inverse_coeffs;
    k::Integer = 3,
    pdeg::Union{Nothing,Integer} = nothing,
    progress = nothing,
)
    if isnothing(pdeg)
        pdeg = maximum(size(inverse_coeffs.D))  -1
    end
    
    pdeg >= 0 || error("Inverse derivative construction requires pdeg >= 0")

    pack = compute_inverse_derivative_numerator_pack(
        inverse_coeffs;
        k,
        pdeg,
        progress,
    )

    deriv_num = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()
    deriv_enc = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()

    for ((i, j, a), enc) in pack.data
        deriv_enc[(i, j, a)] = enc
        deriv_num[(i, j, a)] = enclosure_to_coeffs(deriv_enc[(i, j, a)])
    end

    cov_riem_norm_squared_num =
        k == 3 ? compute_cov_riem_norm_squared_numerator_coeff_space(
            deriv_enc;
            pdeg,
            progress,
        ) : nothing

    cov_cov_riem_norm_squared_num =
        k >= 4 ? compute_cov_cov_riem_norm_squared_numerator_coeff_space(
            deriv_enc;
            pdeg,
            progress,
        ) : nothing

    return (;
        deriv_num,
        D = inverse_coeffs.D,
        pdeg,
        pack,
        cov_riem_norm_squared_num,
        cov_cov_riem_norm_squared_num,
    )
end

"""
Construct the coefficient-space numerator for `||nabla Riem||^2`.

`deriv_enc` must contain Chebyshev enclosures for inverse-metric derivative
numerators up to order three, indexed as `(i, j, a)`. The formula is assembled
directly from those numerator enclosures using coefficient-space products and
sums, preserving cancellations before subdivision.

The returned coefficient array represents the numerator of the norm squared;
the corresponding denominator is `D^9`.
"""
function compute_cov_riem_norm_squared_numerator_coeff_space(
    deriv_enc;
    pdeg::Integer = 20,
    progress = nothing,
)
    u(i, j) = deriv_enc[(i, j, (0, 0))]
    du(i, j, a) = deriv_enc[(i, j, derivative_exponent_multiindex_2d(a))]
    d2u(i, j, a, b) = deriv_enc[(i, j, derivative_exponent_multiindex_2d(a, b))]
    d3u(i, j, a, b, c) = deriv_enc[(i, j, derivative_exponent_multiindex_2d(a, b, c))]

    norm_squared_num = zero_enclosure(pdeg)

    for i in 1:2, j in 1:2, k in 1:2, l in 1:2, m in 1:2, n in 1:2
        bracket = enclosure_mul(u(m, n), d3u(i, k, j, l, n), pdeg; progress)
        bracket = enclosure_add(
            bracket,
            enclosure_mul(d2u(i, k, n, l), du(m, n, j), pdeg; progress),
            pdeg,
        )
        bracket = enclosure_sub(
            bracket,
            enclosure_mul(d2u(n, k, j, l), du(m, i, n), pdeg; progress),
            pdeg,
        )
        bracket = enclosure_add(
            bracket,
            enclosure_mul(d2u(i, k, j, n), du(m, n, l), pdeg; progress),
            pdeg,
        )
        bracket = enclosure_sub(
            bracket,
            enclosure_mul(d2u(i, n, j, l), du(m, k, n), pdeg; progress),
            pdeg,
        )

        term = enclosure_mul(d3u(j, l, i, k, m), bracket, pdeg; progress)
        norm_squared_num = enclosure_add(norm_squared_num, term, pdeg)
    end

    return enclosure_to_coeffs(norm_squared_num)
end

"""
Construct the coefficient-space numerator for `||nabla^2 Riem||^2`.

`deriv_enc` must contain Chebyshev enclosures for inverse-metric derivative
numerators up to order four. This routine builds the left and right contracted
factors appearing in the second covariant Riemann norm formula entirely in
coefficient-enclosure space, then sums their products.

The returned coefficient array represents the numerator of the norm squared;
the corresponding denominator is `D^12`.
"""
function compute_cov_cov_riem_norm_squared_numerator_coeff_space(
    deriv_enc;
    pdeg::Integer = 20,
    progress = nothing,
)
    u(i, j) = deriv_enc[(i, j, (0, 0))]
    du(i, j, a) = deriv_enc[(i, j, derivative_exponent_multiindex_2d(a))]
    d2u(i, j, a, b) = deriv_enc[(i, j, derivative_exponent_multiindex_2d(a, b))]
    d3u(i, j, a, b, c) = deriv_enc[(i, j, derivative_exponent_multiindex_2d(a, b, c))]
    d4u(i, j, a, b, c, d) = deriv_enc[(i, j, derivative_exponent_multiindex_2d(a, b, c, d))]

    norm_squared_num = zero_enclosure(pdeg)

    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        for m in 1:2, b in 1:2, a in 1:2, s in 1:2
            left = enclosure_mul(du(b, m, a), d3u(i, k, j, l, m), pdeg; progress)
            left = enclosure_add(left, enclosure_mul(u(b, m), d4u(i, k, j, l, m, a), pdeg; progress), pdeg)
            left = enclosure_add(left, enclosure_mul(d3u(i, k, m, l, a), du(b, m, j), pdeg; progress), pdeg)
            left = enclosure_add(left, enclosure_mul(d2u(i, k, m, l), d2u(b, m, j, a), pdeg; progress), pdeg)
            left = enclosure_sub(left, enclosure_mul(d3u(m, k, j, l, a), du(b, i, m), pdeg; progress), pdeg)
            left = enclosure_sub(left, enclosure_mul(d2u(m, k, j, l), d2u(b, i, m, a), pdeg; progress), pdeg)
            left = enclosure_add(left, enclosure_mul(d3u(i, k, j, m, a), du(b, m, l), pdeg; progress), pdeg)
            left = enclosure_add(left, enclosure_mul(d2u(i, k, j, m), d2u(b, m, l, a), pdeg; progress), pdeg)
            left = enclosure_sub(left, enclosure_mul(d3u(i, m, j, l, a), du(b, k, m), pdeg; progress), pdeg)
            left = enclosure_sub(left, enclosure_mul(d2u(i, m, j, l), d2u(b, k, m, a), pdeg; progress), pdeg)

            right = enclosure_mul(u(a, s), d4u(j, l, i, k, b, s), pdeg; progress)
            right = enclosure_add(right, enclosure_mul(d3u(j, l, s, k, b), du(a, s, i), pdeg; progress), pdeg)
            right = enclosure_sub(right, enclosure_mul(d3u(a, s, i, k, b), du(j, l, s), pdeg; progress), pdeg)
            right = enclosure_add(right, enclosure_mul(d3u(a, s, i, s, b), du(j, l, k), pdeg; progress), pdeg)
            right = enclosure_sub(right, enclosure_mul(d3u(j, s, i, k, b), du(a, s, l), pdeg; progress), pdeg)

            term = enclosure_mul(left, right, pdeg; progress)
            norm_squared_num = enclosure_add(norm_squared_num, term, pdeg)
        end
    end

    return enclosure_to_coeffs(norm_squared_num)
end

"""
Certify a local-subdivision `C^0` bound for the covariant Riemann tensor.

The input should come from
`compute_inverse_derivative_numerator_components_truncated_coeff_space` with
`k = 3`. On each subdivision box this routine verifies `D > 0`, evaluates the
precomputed numerator for `||nabla Riem||^2`, divides by `D^9`, and records the
largest upper bound over all boxes.

The norm bound is formed from the coefficient-space contraction, rather than
from interval-level component contractions.
"""
function compute_cov_riem_by_local_subdivision_truncated(
    cov_riem_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    hasproperty(cov_riem_coeffs, :cov_riem_norm_squared_num) &&
        cov_riem_coeffs.cov_riem_norm_squared_num !== nothing || error(
            "Covariant Riemann norm requires a coefficient-space norm numerator"
        )
    trunc = (;
        cov_riem_norm_squared_num = truncate_coeffs_with_tail(
            cov_riem_coeffs.cov_riem_norm_squared_num,
            pdeg,
        ),
        D = truncate_coeffs_with_tail(cov_riem_coeffs.D, pdeg),
    )

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_cov_riem_norm = big"0"
    global_D_lower = big"Inf"

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)
        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on covariant Riemann box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        cov_riem_norm_squared_num_box = local_coeff_sum_centered_enclosure_with_tail(
            trunc.cov_riem_norm_squared_num,
            xbox,
            ybox,
        )
        cov_riem_norm_squared_box = cov_riem_norm_squared_num_box / (D_box^9)

        cov_riem_norm_bound = sup(sqrt(cov_riem_norm_squared_box))

        cov_riem_norm_bound < 0 && error(
            "covariant Riemann norm-squared enclosure has negative upper bound: $cov_riem_norm_squared_box"
        )

        global_cov_riem_norm = max(global_cov_riem_norm, cov_riem_norm_bound)
        global_D_lower = min(global_D_lower, D_lower)

        advance_progress!(progress)
    end

    return (;
        nx,
        ny,
        pdeg,
        trunc,
        D_lower = global_D_lower,
        cov_riem_norm_bound = global_cov_riem_norm,
    )
end

"""
Build coefficient-space Riemann numerator data from inverse metric coefficients.

Second derivatives of `u^{ij}` have denominator `D^3`. The routine constructs
`riem_norm_squared_num` by contracting their numerator enclosures directly in
coefficient space. Thus
`riem_norm_squared_num / D^6` encloses `||Riem||^2` on subdivision boxes.
"""
function compute_riem_numerators_truncated_coeff_space(
    inverse_coeffs;
    pdeg::Integer = 20,
    progress = nothing,
)
    pdeg > 0 || error("Truncated Riemann numerator construction requires pdeg > 0")

    pack = compute_inverse_derivative_numerator_pack(
        inverse_coeffs;
        k = 2,
        pdeg,
        progress,
    )

    U = pack.data
    D = inverse_coeffs.D

    Riem_num_enclosure = Dict{NTuple{4, Int}, Any}()

    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        a =
            k == 1 && l == 1 ? (2, 0) :
            k == 2 && l == 2 ? (0, 2) :
            (1, 1)

        Riem_num_enclosure[(i, j, k, l)] = U[(i, j, a)]
    end

    riem_norm_squared_num = zero_enclosure(pdeg)

    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        term = enclosure_mul(
            Riem_num_enclosure[(j, l, i, k)],
            Riem_num_enclosure[(i, k, j, l)],
            pdeg;
            progress,
        )
        riem_norm_squared_num = enclosure_add(riem_norm_squared_num, term, pdeg)
    end

    return (;
        riem_norm_squared_num = enclosure_to_coeffs(riem_norm_squared_num),
        D,
        pdeg,
    )
end

"""
Certify a local-subdivision `C^0` bound for the Riemann tensor.

For each subdivision box the routine verifies positivity of `D`, evaluates the
precomputed coefficient-space numerator for `||Riem||^2`, and divides by
`D^6`. The global return value is the maximum certified norm bound over all
boxes.
"""
function compute_riem_by_local_subdivision_truncated(
    riem_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = (;
        riem_norm_squared_num = truncate_coeffs_with_tail(
            riem_coeffs.riem_norm_squared_num,
            pdeg,
        ),
        D = truncate_coeffs_with_tail(riem_coeffs.D, pdeg),
    )

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_riem_norm = big"0"
    global_D_lower = big"Inf"

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)

        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on Riemann box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        riem_norm_squared_num_box = local_coeff_sum_centered_enclosure_with_tail(
            trunc.riem_norm_squared_num,
            xbox,
            ybox,
        )
        riem_norm_squared_box = riem_norm_squared_num_box / (D_box^6)

        riem_norm_bound = sup(sqrt(riem_norm_squared_box))

        riem_norm_bound < 0 && error(
            "Riemann norm-squared enclosure has negative upper bound: $riem_norm_squared_box"
        )

        global_riem_norm = max(global_riem_norm, riem_norm_bound)
        global_D_lower = min(global_D_lower, D_lower)

        advance_progress!(progress)
    end

    return (;
        nx,
        ny,
        pdeg,
        trunc,
        D_lower = global_D_lower,
        riem_norm_bound = global_riem_norm,
    )
end

"""
Certify a local-subdivision `C^0` bound for the Ricci endomorphism.

The input should come from
`compute_ricci_numerators_truncated_coefficient_space`. On each box this
routine evaluates the Ricci component numerators over `D^3` for diagnostics,
and evaluates the precomputed coefficient-space numerator for `||Ric||^2` over
`D^6` for the actual norm bound.
"""
function compute_ricci_bound_by_local_subdivision_truncated(
    ricci_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = (;
        R11_num = truncate_coeffs_with_tail(ricci_coeffs.R11_num, pdeg),
        R12_num = truncate_coeffs_with_tail(ricci_coeffs.R12_num, pdeg),
        R21_num = truncate_coeffs_with_tail(ricci_coeffs.R21_num, pdeg),
        R22_num = truncate_coeffs_with_tail(ricci_coeffs.R22_num, pdeg),
        ricci_norm_squared_num = truncate_coeffs_with_tail(
            ricci_coeffs.ricci_norm_squared_num,
            pdeg,
        ),
        D = truncate_coeffs_with_tail(ricci_coeffs.D, pdeg),
    )

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_R11 = big"0"
    global_R12 = big"0"
    global_R21 = big"0"
    global_R22 = big"0"
    global_ricci_norm = big"0"
    global_D_lower = big"Inf"

    box_results = []

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        R11_num_box = local_coeff_sum_centered_enclosure_with_tail(trunc.R11_num, xbox, ybox)
        R12_num_box = local_coeff_sum_centered_enclosure_with_tail(trunc.R12_num, xbox, ybox)
        R21_num_box = local_coeff_sum_centered_enclosure_with_tail(trunc.R21_num, xbox, ybox)
        R22_num_box = local_coeff_sum_centered_enclosure_with_tail(trunc.R22_num, xbox, ybox)
        D_box       = local_coeff_sum_centered_enclosure_with_tail(trunc.D,       xbox, ybox)

        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on Ricci box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        R11_box = R11_num_box / (D_box^3)
        R12_box = R12_num_box / (D_box^3)
        R21_box = R21_num_box / (D_box^3)
        R22_box = R22_num_box / (D_box^3)

        R11_bound = sup(abs(R11_box))
        R12_bound = sup(abs(R12_box))
        R21_bound = sup(abs(R21_box))
        R22_bound = sup(abs(R22_box))

        ricci_norm_squared_num_box = local_coeff_sum_centered_enclosure_with_tail(
            trunc.ricci_norm_squared_num,
            xbox,
            ybox,
        )
        ricci_norm_squared_box = ricci_norm_squared_num_box / (D_box^6)

        ricci_norm_bound = sup(sqrt(ricci_norm_squared_box))

        ricci_norm_bound < 0 && error(
            "Ricci norm-squared enclosure has negative upper bound: $ricci_norm_squared_box"
        )

        global_R11 = max(global_R11, R11_bound)
        global_R12 = max(global_R12, R12_bound)
        global_R21 = max(global_R21, R21_bound)
        global_R22 = max(global_R22, R22_bound)
        global_ricci_norm = max(global_ricci_norm, ricci_norm_bound)
        global_D_lower = min(global_D_lower, D_lower)

        push!(box_results, (;
            ix,
            iy,
            xbox,
            ybox,
            R11_num_box,
            R12_num_box,
            R21_num_box,
            R22_num_box,
            D_box,
            D_lower,
            R11_box,
            R12_box,
            R21_box,
            R22_box,
            R11_bound,
            R12_bound,
            R21_bound,
            R22_bound,
            ricci_norm_squared_num_box,
            ricci_norm_squared_box,
            ricci_norm_bound,
        ))

        advance_progress!(progress)
    end

    return (;
        nx,
        ny,
        pdeg,
        trunc,
        box_results,
        D_lower = global_D_lower,
        R11_bound = global_R11,
        R12_bound = global_R12,
        R21_bound = global_R21,
        R22_bound = global_R22,
        ricci_norm_bound = global_ricci_norm,
    )
end

function zero_enclosure(pdeg::Integer = 20)
    return truncated_coeff_enclosure(cheb_constant(interval_constant(0)), pdeg)
end

"""
Build the coefficient-space numerator for `||Ric - Id||^2`.

Using the formula

    ||Ric - Id||^2 = 1/2 * u^{ab}_{cb} u^{cd}_{ad}
                    + 2 * u^{ml}_{ml}
                    + 4,

and the convention `partial_ab u^{ij} = N^{ij}_{ab} / D^3`, this routine forms
a single numerator with common denominator `D^6`. The quadratic contraction,
trace term, and powers of `D` are all assembled in coefficient-enclosure space.

The returned `norm_minus_id_num` satisfies

    ||Ric - Id||^2 <= norm_minus_id_num / D^6

after rigorous local interval evaluation.
"""
function compute_ricci_minus_identity_numerator_truncated_coefficient_space(
    inverse_coeffs;
    pdeg::Integer = 20,
    progress = nothing,
)
    pdeg > 0 || error("Truncated Ricci lower-bound construction requires pdeg > 0")

    pack = compute_inverse_derivative_numerator_pack(
        inverse_coeffs;
        k = 2,
        pdeg,
        progress,
    )

    U = pack.data
    D_enc = truncated_coeff_enclosure(inverse_coeffs.D, pdeg)

    second_num(i, j, a, b) =
        U[(i, j, derivative_exponent_multiindex_2d(a, b))]

    ricci_trace_num = Dict{Tuple{Int, Int}, Any}()

    for a in 1:2, c in 1:2
        out = zero_enclosure(pdeg)
        for b in 1:2
            out = enclosure_add(out, second_num(a, b, c, b), pdeg)
        end
        ricci_trace_num[(a, c)] = out
    end

    scalar_trace_num = zero_enclosure(pdeg)
    for m in 1:2, l in 1:2
        scalar_trace_num = enclosure_add(scalar_trace_num, second_num(m, l, m, l), pdeg)
    end

    quadratic_num = zero_enclosure(pdeg)
    for a in 1:2, c in 1:2
        term = enclosure_mul(ricci_trace_num[(a, c)], ricci_trace_num[(c, a)], pdeg; progress)
        quadratic_num = enclosure_add(quadratic_num, term, pdeg)
    end

    D2 = enclosure_mul(D_enc, D_enc, pdeg; progress)
    D3 = enclosure_mul(D2, D_enc, pdeg; progress)
    D6 = enclosure_mul(D3, D3, pdeg; progress)

    numerator = enclosure_scale(quadratic_num, interval_half(), pdeg)
    numerator = enclosure_add(
        numerator,
        enclosure_mul(enclosure_scale(scalar_trace_num, interval_constant(2), pdeg), D3, pdeg; progress),
        pdeg,
    )
    numerator = enclosure_add(
        numerator,
        enclosure_scale(D6, interval_constant(4), pdeg),
        pdeg,
    )

    return (;
        norm_minus_id_num = enclosure_to_coeffs(numerator),
        D = inverse_coeffs.D,
        pdeg,
        pack,
    )
end

"""
Certify a local-subdivision `C^0` bound for the second covariant Riemann tensor.

The input should come from
`compute_inverse_derivative_numerator_components_truncated_coeff_space` with
`k = 4`. On each box this routine verifies `D > 0`, evaluates the precomputed
coefficient-space numerator for `||nabla^2 Riem||^2`, divides by `D^12`, and
takes the largest upper bound over all boxes.

The norm contraction is performed before subdivision in coefficient space.
"""
function compute_cov_cov_riem_by_local_subdivision_truncated(
    cov_cov_riem_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    hasproperty(cov_cov_riem_coeffs, :cov_cov_riem_norm_squared_num) &&
        cov_cov_riem_coeffs.cov_cov_riem_norm_squared_num !== nothing || error(
            "Second covariant Riemann norm requires a coefficient-space norm numerator"
        )
    trunc = (;
        cov_cov_riem_norm_squared_num = truncate_coeffs_with_tail(
            cov_cov_riem_coeffs.cov_cov_riem_norm_squared_num,
            pdeg,
        ),
        D = truncate_coeffs_with_tail(cov_cov_riem_coeffs.D, pdeg),
    )

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_cov_cov_riem_norm = big"0"
    global_D_lower = big"Inf"

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)
        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on cov-cov Riemann box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        cov_cov_riem_norm_squared_num_box = local_coeff_sum_centered_enclosure_with_tail(
            trunc.cov_cov_riem_norm_squared_num,
            xbox,
            ybox,
        )
        cov_cov_riem_norm_squared_box = cov_cov_riem_norm_squared_num_box / (D_box^12)

        cov_cov_riem_norm_bound = sup(sqrt(cov_cov_riem_norm_squared_box))

        cov_cov_riem_norm_bound < 0 && error(
            "cov-cov Riemann norm-squared enclosure has negative upper bound: $cov_cov_riem_norm_squared_box"
        )

        global_cov_cov_riem_norm =
            max(global_cov_cov_riem_norm, cov_cov_riem_norm_bound)
        global_D_lower = min(global_D_lower, D_lower)

        advance_progress!(progress)
    end

    return (;
        nx,
        ny,
        pdeg,
        trunc,
        D_lower = global_D_lower,
        cov_cov_riem_norm_bound = global_cov_cov_riem_norm,
    )
end

"""
Certify a pointwise lower Ricci bound from `||Ric - Id||`.

The input should come from
`compute_ricci_minus_identity_numerator_truncated_coefficient_space`. On each
subdivision box this routine evaluates `norm_minus_id_num / D^6`, takes an
upper bound `epsilon^2` for `||Ric - Id||^2`, and records the lower pointwise
bound

    Ric >= 1 - epsilon.

The returned `ricci_lower_bound` is the minimum certified lower bound over all
boxes.
"""
function compute_ricci_minus_identity_bound_by_local_subdivision_truncated(
    norm_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = (;
        norm_minus_id_num = truncate_coeffs_with_tail(norm_coeffs.norm_minus_id_num, pdeg),
        D = truncate_coeffs_with_tail(norm_coeffs.D, pdeg),
    )

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_norm_squared = big"0"
    global_epsilon = big"0"
    global_lower_bound = big"1"
    global_D_lower = big"Inf"

    box_results = []

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)
        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on Ricci lower-bound box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        norm_minus_id_num_box = local_coeff_sum_centered_enclosure_with_tail(
            trunc.norm_minus_id_num,
            xbox,
            ybox,
        )

        norm_squared_box = norm_minus_id_num_box / (D_box^6)

        norm_squared_bound = sup(norm_squared_box)

        norm_squared_bound < 0 && error(
            "Ricci-minus-identity norm-squared enclosure has negative upper bound: $norm_squared_box"
        )

        epsilon_bound = sup(sqrt(interval(norm_squared_bound)))
        lower_bound = inf(interval_constant(1) - sqrt(interval(norm_squared_bound)))

        global_norm_squared = max(global_norm_squared, norm_squared_bound)
        global_epsilon = max(global_epsilon, epsilon_bound)
        global_lower_bound = min(global_lower_bound, lower_bound)
        global_D_lower = min(global_D_lower, D_lower)

        push!(box_results, (;
            ix,
            iy,
            xbox,
            ybox,
            D_box,
            D_lower,
            norm_minus_id_num_box,
            norm_squared_box,
            norm_squared_bound,
            epsilon_bound,
            lower_bound,
        ))

        advance_progress!(progress)
    end

    return (;
        nx,
        ny,
        pdeg,
        trunc,
        box_results,
        D_lower = global_D_lower,
        ricci_minus_identity_norm_squared_bound = global_norm_squared,
        epsilon_bound = global_epsilon,
        ricci_lower_bound = global_lower_bound,
    )
end

"""
Print the certified Ricci lower-bound summary returned by subdivision.

The summary reports the global bound for `||Ric - Id||^2`, the derived
`epsilon`, the pointwise lower bound `Ric >= 1 - epsilon`, and the denominator
positivity certificate.
"""
function print_truncated_subdivision_ricci_lower_bound_summary(step)
    println()
    println("Rigorous subdivision-certified lower pointwise Ricci bound:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||Ric - Id||_inf^2 <= $(step.ricci_minus_identity_norm_squared_bound)")
    println("    epsilon <= $(step.epsilon_bound)")
    println("    Ric >= $(step.ricci_lower_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Denominator coefficient tail bound:")
    println("    D tail <= $(step.trunc.D.tail)")
end

"""
Print the certified `C^0` Ricci bound returned by subdivision.

The summary includes the Ricci endomorphism norm bound, component-wise Ricci
bounds, and the denominator positivity/tail certificates used by the proof.
"""
function print_truncated_subdivision_ricci_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bound for Ricci:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||Ric||_inf <= $(step.ricci_norm_bound)")
    println()
    println("Ricci entry bounds:")
    println("    ||R^1_1||_inf <= $(step.R11_bound)")
    println("    ||R^1_2||_inf <= $(step.R12_bound)")
    println("    ||R^2_1||_inf <= $(step.R21_bound)")
    println("    ||R^2_2||_inf <= $(step.R22_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Denominator coefficient tail bound:")
    println("    D tail <= $(step.trunc.D.tail)")
end

"""
Print the certified `C^0` Riemann bound returned by subdivision.

The summary reports the global `||Riem||` bound together with
the denominator positivity certificate and truncation tail for `D`.
"""
function print_truncated_subdivision_riem_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bound for Riemann:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||Riem||_inf <= $(step.riem_norm_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Denominator coefficient tail bound:")
    println("    D tail <= $(step.trunc.D.tail)")
end

"""
Print the certified `C^0` covariant Riemann bound returned by subdivision.

The summary reports the global `||nabla Riem||` plus
the denominator certificate and truncation tail for `D`.
"""
function print_truncated_subdivision_cov_riem_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bound for covariant Riemann:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||∇Riem||_inf <= $(step.cov_riem_norm_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Denominator coefficient tail bound:")
    println("    D tail <= $(step.trunc.D.tail)")
end

"""
Print the certified `C^0` second covariant Riemann bound.

The summary reports the global `||nabla^2 Riem||` bound, plus
the denominator certificate and truncation tail for `D`.
"""
function print_truncated_subdivision_cov_cov_riem_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bound for second covariant Riemann:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||∇²Riem||_inf <= $(step.cov_cov_riem_norm_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Denominator coefficient tail bound:")
    println("    D tail <= $(step.trunc.D.tail)")
end
