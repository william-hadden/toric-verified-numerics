const DEL_PEZZO_NORMALS = ((-1, 1), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 0))
const DEL_PEZZO_LAMBDAS = (1, 1, 1, 1, 1, 1)

function compute_second_derivatives(coefficients::AbstractMatrix{<:Number})
    interval_coefficients = intervalize_coefficients(coefficients)
    pack = build_derivative_pack(interval_coefficients)

    return (;
        uxx = pack.uxx,
        uyy = pack.uyy,
        uxy = pack.uxy,
        pack,
    )
end

function facet_series()
    half = interval_half()
    x_series = zeros(Interval{BigFloat}, 2, 1)
    y_series = zeros(Interval{BigFloat}, 1, 2)
    x_series[1, 1] = half
    x_series[2, 1] = half
    y_series[1, 1] = half
    y_series[1, 2] = -half

    return ntuple(length(DEL_PEZZO_NORMALS)) do a
        normal = DEL_PEZZO_NORMALS[a]
        cheb_add_constant(
            cheb_add(
                cheb_scale(x_series, interval_constant(normal[1])),
                cheb_scale(y_series, interval_constant(normal[2])),
            ),
            interval_constant(DEL_PEZZO_LAMBDAS[a]),
        )
    end
end

product_except(series, excluded::Integer; progress = nothing) =
    cheb_product_fast([series[a] for a in eachindex(series) if a != excluded]; progress)

product_except(series, excluded_a::Integer, excluded_b::Integer; progress = nothing) =
    cheb_product_fast([series[c] for c in eachindex(series) if c != excluded_a && c != excluded_b]; progress)

facet_normal_det(a::Integer, b::Integer) =
    DEL_PEZZO_NORMALS[a][1] * DEL_PEZZO_NORMALS[b][2] -
    DEL_PEZZO_NORMALS[a][2] * DEL_PEZZO_NORMALS[b][1]

function factored_canonical_metric_coeffs(; progress = nothing)
    facets = facet_series()
    lprod = cheb_product_fast(collect(facets); progress)
    lprod_v11 = cheb_constant(interval_constant(0))
    lprod_v12 = cheb_constant(interval_constant(0))
    lprod_v22 = cheb_constant(interval_constant(0))
    B = cheb_constant(interval_constant(0))

    for a in eachindex(facets)
        normal = DEL_PEZZO_NORMALS[a]
        omitted = product_except(facets, a; progress)
        normal[1] != 0 && (lprod_v11 = cheb_add(lprod_v11, cheb_scale(omitted, interval_half())))
        normal[1] != 0 && normal[2] != 0 && (lprod_v12 = cheb_add(lprod_v12, cheb_scale(omitted, interval_constant(normal[1] * normal[2]) / exact(2))))
        normal[2] != 0 && (lprod_v22 = cheb_add(lprod_v22, cheb_scale(omitted, interval_half())))
    end

    for a in 1:(length(facets) - 1), b in (a + 1):length(facets)
        det_int = facet_normal_det(a, b)
        det_int == 0 && continue
        B = cheb_add(B, cheb_scale(product_except(facets, a, b; progress), interval_constant(det_int^2) / exact(4)))
    end

    return (; lprod, lprod_v11, lprod_v12, lprod_v22, B)
end

function cheb_coeffs_to_lobatto_values_2d(coeffs::AbstractMatrix{<:Number}, degx::Integer, degy::Integer)
    return cheb_coeffs_to_lobatto_values_2d(cheb_pad(coeffs, degx, degy))
end

function factored_canonical_metric_values(degx::Integer, degy::Integer; progress = nothing)
    facets = facet_series()
    facet_values = map(f -> cheb_coeffs_to_lobatto_values_2d(f, degx, degy), facets)

    lprod = fill(interval_constant(1), degx + 1, degy + 1)
    for values in facet_values
        lprod .*= values
        advance_progress!(progress)
    end

    lprod_v11 = fill(interval_constant(0), degx + 1, degy + 1)
    lprod_v12 = fill(interval_constant(0), degx + 1, degy + 1)
    lprod_v22 = fill(interval_constant(0), degx + 1, degy + 1)
    B = fill(interval_constant(0), degx + 1, degy + 1)

    for a in eachindex(facet_values)
        normal = DEL_PEZZO_NORMALS[a]
        omitted = fill(interval_constant(1), degx + 1, degy + 1)
        for c in eachindex(facet_values)
            c == a && continue
            omitted .*= facet_values[c]
        end
        advance_progress!(progress)

        normal[1] != 0 && (lprod_v11 .+= omitted .* interval_half())
        normal[1] != 0 && normal[2] != 0 && (lprod_v12 .+= omitted .* (interval_constant(normal[1] * normal[2]) / exact(2)))
        normal[2] != 0 && (lprod_v22 .+= omitted .* interval_half())
    end

    for a in 1:(length(facet_values) - 1), b in (a + 1):length(facet_values)
        det_int = facet_normal_det(a, b)
        det_int == 0 && continue

        omitted = fill(interval_constant(1), degx + 1, degy + 1)
        for c in eachindex(facet_values)
            (c == a || c == b) && continue
            omitted .*= facet_values[c]
        end
        advance_progress!(progress)

        B .+= omitted .* (interval_constant(det_int^2) / exact(4))
    end

    coeffs = (;
        lprod = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(lprod)),
        lprod_v11 = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(lprod_v11)),
        lprod_v12 = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(lprod_v12)),
        lprod_v22 = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(lprod_v22)),
        B = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(B)),
    )

    return (;
        lprod,
        lprod_v11,
        lprod_v12,
        lprod_v22,
        B,
        coeffs,
    )
end

function build_inverse_metric_coeffs(derivatives; progress = nothing)
    h11 = intervalize_coefficients(derivatives.uxx)
    h12 = intervalize_coefficients(derivatives.uxy)
    h22 = intervalize_coefficients(derivatives.uyy)
    v = factored_canonical_metric_coeffs(; progress)

    h11_h22 = cheb_mul_fast(h11, h22)
    advance_progress!(progress)
    h12_h12 = cheb_mul_fast(h12, h12)
    advance_progress!(progress)
    det_h = cheb_sub(h11_h22, h12_h12)

    D_v22_h11 = cheb_mul_fast(v.lprod_v22, h11)
    advance_progress!(progress)
    D_v11_h22 = cheb_mul_fast(v.lprod_v11, h22)
    advance_progress!(progress)
    D_v12_h12 = cheb_mul_fast(v.lprod_v12, h12)
    advance_progress!(progress)
    D_lprod_det_h = cheb_mul_fast(v.lprod, det_h)
    advance_progress!(progress)

    D =
        cheb_add(
            cheb_add(
                v.B,
                cheb_sub(
                    cheb_add(
                        D_v22_h11,
                        D_v11_h22,
                    ),
                    cheb_scale(D_v12_h12, interval_constant(2)),
                ),
            ),
            D_lprod_det_h,
        )

    A11 = cheb_add(v.lprod_v22, cheb_mul_fast(v.lprod, h22))
    advance_progress!(progress)
    A12 = cheb_scale(cheb_add(v.lprod_v12, cheb_mul_fast(v.lprod, h12)), interval_constant(-1))
    advance_progress!(progress)
    A22 = cheb_add(v.lprod_v11, cheb_mul_fast(v.lprod, h11))
    advance_progress!(progress)

    return (; A11, A12, A21 = A12, A22, D, canonical = v)
end

function build_inverse_metric_coeffs_real_space(derivatives; progress = nothing)
    h11 = intervalize_coefficients(derivatives.uxx)
    h12 = intervalize_coefficients(derivatives.uxy)
    h22 = intervalize_coefficients(derivatives.uyy)

    h_degx = maximum(size(A, 1) for A in (h11, h12, h22)) - 1
    h_degy = maximum(size(A, 2) for A in (h11, h12, h22)) - 1
    canonical_deg = length(DEL_PEZZO_NORMALS)
    degx = canonical_deg + 2 * h_degx
    degy = canonical_deg + 2 * h_degy

    h11_values = cheb_coeffs_to_lobatto_values_2d(h11, degx, degy)
    h12_values = cheb_coeffs_to_lobatto_values_2d(h12, degx, degy)
    h22_values = cheb_coeffs_to_lobatto_values_2d(h22, degx, degy)
    v = factored_canonical_metric_values(degx, degy; progress)

    det_h_values = h11_values .* h22_values .- h12_values .* h12_values
    D_values =
        v.B .+
        v.lprod_v22 .* h11_values .+
        v.lprod_v11 .* h22_values .-
        interval_constant(2) .* v.lprod_v12 .* h12_values .+
        v.lprod .* det_h_values
    advance_progress!(progress)

    A11_values = v.lprod_v22 .+ v.lprod .* h22_values
    advance_progress!(progress)
    A12_values = interval_constant(-1) .* (v.lprod_v12 .+ v.lprod .* h12_values)
    advance_progress!(progress)
    A22_values = v.lprod_v11 .+ v.lprod .* h11_values
    advance_progress!(progress)

    A11 = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(A11_values))
    A12 = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(A12_values))
    A22 = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(A22_values))
    D = cheb_trim_exact(cheb_lobatto_values_to_coeffs_2d(D_values))

    return (; A11, A12, A21 = A12, A22, D, canonical = v.coeffs)
end

function build_inverse_metric(derivatives; progress = nothing, method::Symbol = :coefficient_space)
    if method === :coefficient_space
        return build_inverse_metric_coeffs(derivatives; progress)
    elseif method === :real_space
        return build_inverse_metric_coeffs_real_space(derivatives; progress)
    else
        error("Unknown inverse metric build method: $method")
    end
end

function build_inverse_metric(derivatives, method::Symbol; progress = nothing)
    return build_inverse_metric(derivatives; progress, method)
end

function build_inverse_metric_real_space(derivatives; progress = nothing)
    return build_inverse_metric(derivatives; progress, method = :real_space)
end
