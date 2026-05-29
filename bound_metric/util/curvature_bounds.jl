function derivative_coeff_pack(coeffs::AbstractMatrix{<:Number})
    x = differentiate_coeffs_x(coeffs)
    y = differentiate_coeffs_y(coeffs)
    xx = differentiate_coeffs_x(x)
    xy = differentiate_coeffs_y(x)
    yy = differentiate_coeffs_y(y)
    return (; f = coeffs, x, y, xx, xy, yy)
end

first_derivative_coeffs(pack, dim::Integer) =
    dim == 1 ? pack.x :
    dim == 2 ? pack.y :
    error("Expected derivative dimension 1 or 2")

second_derivative_coeffs(pack, a::Integer, b::Integer) =
    a == 1 && b == 1 ? pack.xx :
    a == 2 && b == 2 ? pack.yy :
    a != b ? pack.xy :
    error("Expected derivative dimensions 1 or 2")

function coeff_sup_bound_number(coeffs::AbstractMatrix{<:Number})
    return sup(chebyshev_coeff_sup_bound(intervalize_coefficients(coeffs)))
end

function truncated_coeff_enclosure(coeffs::AbstractMatrix{<:Number}, pdeg::Integer)
    trunc = truncate_coeffs_with_tail(coeffs, pdeg)
    tail = trunc.tail isa Interval ? sup(abs(trunc.tail)) : abs(trunc.tail)
    return (; coeffs = trunc.coeffs, tail, pdeg = trunc.pdeg)
end

function truncate_enclosure_coeffs(coeffs::AbstractMatrix{<:Number}, tail, pdeg::Integer)
    trunc = truncate_coeffs_with_tail(coeffs, pdeg)

    old_tail = tail isa Interval ? sup(abs(tail)) : abs(tail)
    new_tail = trunc.tail isa Interval ? sup(abs(trunc.tail)) : abs(trunc.tail)

    return (; coeffs = trunc.coeffs, tail = old_tail + new_tail, pdeg = trunc.pdeg)
end

function enclosure_add(f, g, pdeg::Integer)
    return truncate_enclosure_coeffs(cheb_add(f.coeffs, g.coeffs), f.tail + g.tail, pdeg)
end

function enclosure_sub(f, g, pdeg::Integer)
    return truncate_enclosure_coeffs(cheb_sub(f.coeffs, g.coeffs), f.tail + g.tail, pdeg)
end

function enclosure_scale(f, c, pdeg::Integer)
    c_abs = c isa Interval ? sup(abs(c)) : abs(c)
    return truncate_enclosure_coeffs(cheb_scale(f.coeffs, c), c_abs * f.tail, pdeg)
end

function enclosure_mul(f, g, pdeg::Integer; progress = nothing)
    product_coeffs = cheb_mul_fast(f.coeffs, g.coeffs)
    advance_progress!(progress)

    product_tail =
        f.tail * coeff_sup_bound_number(g.coeffs) +
        g.tail * coeff_sup_bound_number(f.coeffs) +
        f.tail * g.tail

    return truncate_enclosure_coeffs(product_coeffs, product_tail, pdeg)
end

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

function exact_derivative_coeffs(coeffs::AbstractMatrix, I)
    out = coeffs
    for a in I
        a == 1 && (out = differentiate_coeffs_x(out); continue)
        a == 2 && (out = differentiate_coeffs_y(out); continue)
        error("Multiindex entries must be 1 or 2")
    end
    return out
end

function make_derivative_cache(coeffs::AbstractMatrix)
    return Dict{Tuple, Any}(() => coeffs)
end

function cached_exact_derivative!(cache::Dict{Tuple, Any}, coeffs::AbstractMatrix, I)
    key = Tuple(I)
    if !haskey(cache, key)
        cache[key] = exact_derivative_coeffs(coeffs, key)
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


# ------------------------------------------------------------
# Evaluate formal numerator using tail-summed enclosures
# ------------------------------------------------------------

function formal_term_to_enclosure(
    t,
    A_coeffs::AbstractMatrix,
    D_coeffs::AbstractMatrix,
    A_cache::Dict{Tuple, Any},
    D_cache::Dict{Tuple, Any};
    pdeg::Integer,
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

function quotient_derivative_numerator_enclosure(
    A_coeffs::AbstractMatrix,
    D_coeffs::AbstractMatrix,
    I;
    pdeg::Integer,
    progress = nothing,
)
    terms = quotient_numerator_formal_terms(I)

    A_cache = make_derivative_cache(A_coeffs)
    D_cache = make_derivative_cache(D_coeffs)

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

function multiindices_2d_upto(k::Integer)
    k >= 0 || error("Derivative order k must be >= 0")

    out = Tuple{Int, Int}[]

    for total in 0:k
        for a1 in 0:total
            a2 = total - a1
            push!(out, (a1, a2))
        end
    end

    return out
end

function expand_multiindex_2d(a::Tuple{Int, Int})
    a1, a2 = a
    a1 >= 0 && a2 >= 0 || error("Multiindex entries must be nonnegative")

    return Tuple(vcat(fill(1, a1), fill(2, a2)))
end

"""
This is the equivalent of R^i{}_j{}^k{}_l=-1/2u^{jl}_{ik}
"""
function compute_inverse_derivative_numerator_pack(
    inverse_coeffs;
    k::Integer,
    pdeg::Integer,
    progress = nothing,
)
    k >= 0 || error("Maximum derivative order k must be >= 0")
    pdeg > 0 || error("Truncated inverse derivative numerator pack requires pdeg > 0")

    A = Dict(
        (1, 1) => inverse_coeffs.A11,
        (1, 2) => inverse_coeffs.A12,
        (2, 1) => inverse_coeffs.A12,
        (2, 2) => inverse_coeffs.A22,
    )

    D = inverse_coeffs.D
    scale = interval_constant(-1) / exact(2)

    data = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()

    for i in 1:2, j in 1:2
        for a in multiindices_2d_upto(k)
            I = expand_multiindex_2d(a)

            num = quotient_derivative_numerator_enclosure(
                A[(i, j)],
                D,
                I;
                pdeg,
                progress,
            )

            data[(i, j, a)] = enclosure_scale(num, scale, pdeg)
        end
    end

    return (;
        max_derivative_order = k,
        data,
    )
end

function compute_ricci_numerators_truncated_coefficient_space(
    inverse_coeffs;
    pdeg::Integer,
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

    R11_num = enclosure_to_coeffs(enclosure_add(U[(1, 1, (2, 0))], U[(1, 2, (1, 1))], pdeg))
    R12_num = enclosure_to_coeffs(enclosure_add(U[(1, 1, (1, 1))], U[(1, 2, (0, 2))], pdeg))
    R21_num = enclosure_to_coeffs(enclosure_add(U[(1, 2, (2, 0))], U[(2, 2, (1, 1))], pdeg))
    R22_num = enclosure_to_coeffs(enclosure_add(U[(1, 2, (1, 1))], U[(2, 2, (0, 2))], pdeg))

    return (; R11_num, R12_num, R21_num, R22_num, D, pdeg, pack)
end

function prepare_ricci_coeffs_for_subdivision(ricci_coeffs, pdeg::Integer)
    return (;
        R11_num = truncate_coeffs_with_tail(ricci_coeffs.R11_num, pdeg),
        R12_num = truncate_coeffs_with_tail(ricci_coeffs.R12_num, pdeg),
        R21_num = truncate_coeffs_with_tail(ricci_coeffs.R21_num, pdeg),
        R22_num = truncate_coeffs_with_tail(ricci_coeffs.R22_num, pdeg),
        D       = truncate_coeffs_with_tail(ricci_coeffs.D, pdeg),
    )
end

function derivative_exponent_multiindex_2d(indices::Vararg{Int})
    a1 = count(==(1), indices)
    a2 = count(==(2), indices)

    a1 + a2 == length(indices) || error("Derivative indices must be 1 or 2")

    return (a1, a2)
end


function compute_inverse_third_derivative_numerator_components_truncated_coeff_space(
    inverse_coeffs;
    pdeg::Integer,
    progress = nothing,
)
    pdeg > 0 || error("Truncated inverse third derivative construction requires pdeg > 0")

    pack = compute_inverse_derivative_numerator_pack(
        inverse_coeffs;
        k = 3,
        pdeg,
        progress,
    )

    U = pack.data
    D = inverse_coeffs.D

    third_deriv_pack = Dict{NTuple{5, Int}, Any}()

    for i in 1:2, j in 1:2, k in 1:2, l in 1:2, m in 1:2
        a = derivative_exponent_multiindex_2d(k, l, m)

        third_deriv_pack[(i, j, k, l, m)] = enclosure_to_coeffs(enclosure_scale(U[(i, j, a)],exact(-2), pdeg))
    end

    return (;
        third_deriv_pack,
        D,
        pdeg,
        pack,
    )
end
function compute_inverse_derivative_numerator_components_truncated_coeff_space(
    inverse_coeffs;
    k::Integer = 3,
    pdeg::Integer,
    progress = nothing,
)
    pdeg > 0 || error("Truncated inverse derivative construction requires pdeg > 0")

    pack = compute_inverse_derivative_numerator_pack(
        inverse_coeffs;
        k,
        pdeg,
        progress,
    )

    deriv_num = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()

    for ((i, j, a), enc) in pack.data
        # pack stores -1/2 * derivative numerator,
        # so multiply by -2 to get the actual derivative numerator.
        deriv_num[(i, j, a)] =
            enclosure_to_coeffs(enclosure_scale(enc, exact(-2), pdeg))
    end

    return (;
        deriv_num,
        D = inverse_coeffs.D,
        pdeg,
        pack,
    )
end

function prepare_inverse_deriv_coeffs_for_subdivision(cov_riem_coeffs, pdeg::Integer)
    deriv_num = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()

    for ((i, j, a), coeffs) in cov_riem_coeffs.deriv_num
        deriv_num[(i, j, a)] = truncate_coeffs_with_tail(coeffs, pdeg)
    end

    return (;
        deriv_num,
        D = truncate_coeffs_with_tail(cov_riem_coeffs.D, pdeg),
    )
end

function compute_cov_riem_by_local_subdivision_truncated(
    cov_riem_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = prepare_inverse_deriv_coeffs_for_subdivision(cov_riem_coeffs, pdeg)

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_cov_riem_norm_squared = big"0"
    global_cov_riem_norm = big"0"
    global_D_lower = big"Inf"

    box_results = []

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)
        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on covariant Riemann box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        deriv_box = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()

        for ((i, j, a), trunc_coeffs) in trunc.deriv_num
            num_box = local_coeff_sum_centered_enclosure_with_tail(
                trunc_coeffs,
                xbox,
                ybox,
            )

            denom_power = a[1] + a[2] + 1
            deriv_box[(i, j, a)] = num_box / (D_box^denom_power)
        end

        u(i, j) = deriv_box[(i, j, (0, 0))]
        du(i, j, a) = deriv_box[(i, j, derivative_exponent_multiindex_2d(a))]
        d2u(i, j, a, b) = deriv_box[(i, j, derivative_exponent_multiindex_2d(a, b))]
        d3u(i, j, a, b, c) = deriv_box[(i, j, derivative_exponent_multiindex_2d(a, b, c))]

        cov_riem_norm_squared_box = interval_constant(0)

        for i in 1:2, j in 1:2, k in 1:2, l in 1:2, m in 1:2, n in 1:2
            bracket =
                u(m, n) * d3u(i, k, j, l, n) +
                d2u(l, k, n, l) * du(m, n, j) -
                d2u(n, k, j, l) * du(m, l, n) +
                d2u(l, k, j, n) * du(m, n, l) -
                d2u(l, n, j, l) * du(m, k, n)

            cov_riem_norm_squared_box +=
                d3u(j, l, i, k, m) * bracket
        end

        cov_riem_norm_squared_box *= interval_constant(1) / exact(8)

        cov_riem_norm_squared_bound = sup(cov_riem_norm_squared_box)

        cov_riem_norm_squared_bound < 0 && error(
            "covariant Riemann norm-squared enclosure has negative upper bound: $cov_riem_norm_squared_box"
        )

        cov_riem_norm_bound = sqrt(cov_riem_norm_squared_bound)

        global_cov_riem_norm_squared = max(global_cov_riem_norm_squared, cov_riem_norm_squared_bound)
        global_cov_riem_norm = max(global_cov_riem_norm, cov_riem_norm_bound)
        global_D_lower = min(global_D_lower, D_lower)

        push!(box_results, (;
            ix,
            iy,
            xbox,
            ybox,
            D_box,
            D_lower,
            deriv_box,
            cov_riem_norm_squared_box,
            cov_riem_norm_squared_bound,
            cov_riem_norm_bound,
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
        cov_riem_norm_squared_bound = global_cov_riem_norm_squared,
        cov_riem_norm_bound = global_cov_riem_norm,
    )
end

function compute_riem_numerators_truncated_coeff_space(
    inverse_coeffs;
    pdeg::Integer,
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

    Riem_num = Dict{NTuple{4, Int}, Any}()

    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        a =
            k == 1 && l == 1 ? (2, 0) :
            k == 2 && l == 2 ? (0, 2) :
            (1, 1)

        Riem_num[(i, j, k, l)] = enclosure_to_coeffs(U[(i, j, a)])
    end

    return (;
        Riem_num,
        D,
        pdeg,
        pack,
    )
end

function prepare_riem_coeffs_for_subdivision(riem_coeffs, pdeg::Integer)
    Riem_num = Dict{NTuple{4, Int}, Any}()

    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        Riem_num[(i, j, k, l)] =
            truncate_coeffs_with_tail(riem_coeffs.Riem_num[(i, j, k, l)], pdeg)
    end

    return (;
        Riem_num,
        D = truncate_coeffs_with_tail(riem_coeffs.D, pdeg),
    )
end


function compute_riem_by_local_subdivision_truncated(
    riem_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = prepare_riem_coeffs_for_subdivision(riem_coeffs, pdeg)

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_riem_norm_squared = big"0"
    global_riem_norm = big"0"
    global_D_lower = big"Inf"

    box_results = []

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)

        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on Riemann box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        Riem_num_box = Dict{NTuple{4, Int}, Any}()
        Riem_box = Dict{NTuple{4, Int}, Any}()
        Riem_bound = Dict{NTuple{4, Int}, Any}()

        for i in 1:2, j in 1:2, k in 1:2, l in 1:2
            key = (i, j, k, l)

            Riem_num_box[key] =
                local_coeff_sum_centered_enclosure_with_tail(
                    trunc.Riem_num[key],
                    xbox,
                    ybox,
                )

            Riem_box[key] = Riem_num_box[key] / (D_box^3)
            Riem_bound[key] = sup(abs(Riem_box[key]))
        end

        riem_norm_squared_box = interval_constant(0)

        for i in 1:2, j in 1:2, k in 1:2, l in 1:2
            riem_norm_squared_box +=
                Riem_box[(j, l, i, k)] * Riem_box[(i, k, j, l)] # suspect want chebmul_fast here. 
        end

        riem_norm_squared_bound = sup(riem_norm_squared_box)

        riem_norm_squared_bound < 0 && error(
            "Riemann norm-squared enclosure has negative upper bound: $riem_norm_squared_box"
        )

        riem_norm_bound = sqrt(riem_norm_squared_bound)

        global_riem_norm_squared = max(global_riem_norm_squared, riem_norm_squared_bound)
        global_riem_norm = max(global_riem_norm, riem_norm_bound)
        global_D_lower = min(global_D_lower, D_lower)

        push!(box_results, (;
            ix,
            iy,
            xbox,
            ybox,
            Riem_num_box,
            D_box,
            D_lower,
            Riem_box,
            Riem_bound,
            riem_norm_squared_box,
            riem_norm_squared_bound,
            riem_norm_bound,
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
        riem_norm_squared_bound = global_riem_norm_squared,
        riem_norm_bound = global_riem_norm,
    )
end

function compute_ricci_bound_by_local_subdivision_truncated(
    ricci_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = prepare_ricci_coeffs_for_subdivision(ricci_coeffs, pdeg)

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_R11 = big"0"
    global_R12 = big"0"
    global_R21 = big"0"
    global_R22 = big"0"
    global_ricci_norm_squared = big"0"
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

        ricci_norm_squared_box =
            interval_constant(2) * (
                R11_box^2 +
                interval_constant(2) * R12_box * R21_box +
                R22_box^2
            )

        ricci_norm_squared_bound = sup(ricci_norm_squared_box)

        ricci_norm_squared_bound < 0 && error(
            "Ricci norm-squared enclosure has negative upper bound: $ricci_norm_squared_box"
        )

        ricci_norm_bound = sqrt(ricci_norm_squared_bound)

        global_R11 = max(global_R11, R11_bound)
        global_R12 = max(global_R12, R12_bound)
        global_R21 = max(global_R21, R21_bound)
        global_R22 = max(global_R22, R22_bound)
        global_ricci_norm_squared = max(global_ricci_norm_squared, ricci_norm_squared_bound)
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
            ricci_norm_squared_box,
            ricci_norm_squared_bound,
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
        ricci_norm_squared_bound = global_ricci_norm_squared,
        ricci_norm_bound = global_ricci_norm,
    )
end

function compute_cov_cov_riem_by_local_subdivision_truncated(
    cov_cov_riem_coeffs;
    pdeg::Integer = 20,
    nx::Integer = 8,
    ny::Integer = nx,
    progress = nothing,
)
    trunc = prepare_inverse_deriv_coeffs_for_subdivision(cov_cov_riem_coeffs, pdeg)

    xboxes = subdivide_minus_one_one(nx)
    yboxes = subdivide_minus_one_one(ny)

    global_cov_cov_riem_norm_squared = big"0"
    global_cov_cov_riem_norm = big"0"
    global_D_lower = big"Inf"

    box_results = []

    for (ix, xbox) in pairs(xboxes), (iy, ybox) in pairs(yboxes)
        D_box = local_coeff_sum_centered_enclosure_with_tail(trunc.D, xbox, ybox)
        D_lower = inf(D_box)

        if !isfinite(D_lower) || D_lower <= 0
            error(
                "Could not certify positivity of D on cov-cov Riemann box " *
                "(ix=$ix, iy=$iy): D_box = $D_box, inf(D_box) = $D_lower"
            )
        end

        deriv_box = Dict{Tuple{Int, Int, Tuple{Int, Int}}, Any}()

        for ((i, j, a), trunc_coeffs) in trunc.deriv_num
            num_box = local_coeff_sum_centered_enclosure_with_tail(
                trunc_coeffs,
                xbox,
                ybox,
            )

            denom_power = a[1] + a[2] + 1
            deriv_box[(i, j, a)] = num_box / (D_box^denom_power)
        end

        u(i, j) = deriv_box[(i, j, (0, 0))]
        du(i, j, a) = deriv_box[(i, j, derivative_exponent_multiindex_2d(a))]
        d2u(i, j, a, b) = deriv_box[(i, j, derivative_exponent_multiindex_2d(a, b))]
        d3u(i, j, a, b, c) = deriv_box[(i, j, derivative_exponent_multiindex_2d(a, b, c))]
        d4u(i, j, a, b, c, d) = deriv_box[(i, j, derivative_exponent_multiindex_2d(a, b, c, d))]

        cov_cov_riem_norm_squared_box = interval_constant(0)

        for i in 1:2, j in 1:2, k in 1:2, l in 1:2
            for m in 1:2, b in 1:2, a in 1:2, s in 1:2

                left =
                    du(b, m, a) * d3u(i, k, j, l, m) +
                    u(b, m)     * d4u(i, k, j, l, m, a) +

                    d3u(i, k, m, l, a) * du(b, m, j) +
                    d2u(i, k, m, l)    * d2u(b, m, j, a) -

                    d3u(m, k, j, l, a) * du(b, i, m) -
                    d2u(m, k, j, l)    * d2u(b, i, m, a) +

                    d3u(i, k, j, m, a) * du(b, m, l) +
                    d2u(i, k, j, m)    * d2u(b, m, l, a) -

                    d3u(i, m, j, l, a) * du(b, k, m) -
                    d2u(i, m, j, l)    * d2u(b, k, m, a)

                right =
                    u(a, s)     * d4u(j, l, i, k, b, s) +
                    d3u(j, l, s, k, b) * du(a, s, i) -
                    d3u(a, s, i, k, b) * du(j, l, s) +
                    d3u(a, s, i, s, b) * du(j, l, k) -
                    d3u(j, s, i, k, b) * du(a, s, l)

                cov_cov_riem_norm_squared_box += left * right
            end
        end

        cov_cov_riem_norm_squared_box *= interval_constant(1) / exact(16)

        cov_cov_riem_norm_squared_bound = sup(cov_cov_riem_norm_squared_box)

        cov_cov_riem_norm_squared_bound < 0 && error(
            "cov-cov Riemann norm-squared enclosure has negative upper bound: $cov_cov_riem_norm_squared_box"
        )

        cov_cov_riem_norm_bound = sqrt(cov_cov_riem_norm_squared_bound)

        global_cov_cov_riem_norm_squared =
            max(global_cov_cov_riem_norm_squared, cov_cov_riem_norm_squared_bound)
        global_cov_cov_riem_norm =
            max(global_cov_cov_riem_norm, cov_cov_riem_norm_bound)
        global_D_lower = min(global_D_lower, D_lower)

        push!(box_results, (;
            ix,
            iy,
            xbox,
            ybox,
            D_box,
            D_lower,
            deriv_box,
            cov_cov_riem_norm_squared_box,
            cov_cov_riem_norm_squared_bound,
            cov_cov_riem_norm_bound,
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
        cov_cov_riem_norm_squared_bound = global_cov_cov_riem_norm_squared,
        cov_cov_riem_norm_bound = global_cov_cov_riem_norm,
    )
end

function print_truncated_subdivision_ricci_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bound for Ricci:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||Ric||_inf <= $(step.ricci_norm_bound)")
    println("    ||Ric||_inf^2 <= $(step.ricci_norm_squared_bound)")
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

function print_truncated_subdivision_riem_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bound for Riemann:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||Riem||_inf <= $(step.riem_norm_bound)")
    println("    ||Riem||_inf^2 <= $(step.riem_norm_squared_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Denominator coefficient tail bound:")
    println("    D tail <= $(step.trunc.D.tail)")
end

function print_truncated_subdivision_cov_riem_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bound for covariant Riemann:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||∇Riem||_inf <= $(step.cov_riem_norm_bound)")
    println("    ||∇Riem||_inf^2 <= $(step.cov_riem_norm_squared_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Denominator coefficient tail bound:")
    println("    D tail <= $(step.trunc.D.tail)")
end

function print_truncated_subdivision_cov_cov_riem_bound_summary(step)
    println()
    println("Rigorous subdivision-certified C^0 bound for second covariant Riemann:")
    println("    boxes: $(step.nx) x $(step.ny)")
    println("    pdeg: $(step.pdeg <= 0 ? "full" : step.pdeg)")
    println("    ||∇²Riem||_inf <= $(step.cov_cov_riem_norm_bound)")
    println("    ||∇²Riem||_inf^2 <= $(step.cov_cov_riem_norm_squared_bound)")
    println()
    println("Denominator positivity certificate:")
    println("    inf D >= $(step.D_lower)")
    println()
    println("Denominator coefficient tail bound:")
    println("    D tail <= $(step.trunc.D.tail)")
end