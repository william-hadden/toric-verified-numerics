const DEL_PEZZO_NORMALS = ((-1, 1), (0, 1), (1, 0), (1, -1), (0, -1), (-1, 0))

function compute_second_derivatives(coefficients::AbstractMatrix{<:Number})
    pack = build_derivative_pack(intervalize_coefficients(coefficients))
    return (; uxx = pack.uxx, uxy = pack.uxy, uyy = pack.uyy)
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
            interval_constant(1),
        )
    end
end

product_except(series, excluded::Integer; progress = nothing) =
    cheb_product_fast(
        [series[a] for a in eachindex(series) if a != excluded];
        progress,
    )

product_except(series, excluded_a::Integer, excluded_b::Integer; progress = nothing) =
    cheb_product_fast(
        [series[c] for c in eachindex(series) if c != excluded_a && c != excluded_b];
        progress,
    )

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

        if normal[1] != 0
            lprod_v11 = cheb_add(lprod_v11, cheb_scale(omitted, interval_half()))
        end

        if normal[1] != 0 && normal[2] != 0
            lprod_v12 = cheb_add(
                lprod_v12,
                cheb_scale(omitted, interval_constant(normal[1] * normal[2]) / exact(2)),
            )
        end

        if normal[2] != 0
            lprod_v22 = cheb_add(lprod_v22, cheb_scale(omitted, interval_half()))
        end
    end

    for a in 1:(length(facets) - 1), b in (a + 1):length(facets)
        normal_a = DEL_PEZZO_NORMALS[a]
        normal_b = DEL_PEZZO_NORMALS[b]

        det_int = normal_a[1] * normal_b[2] - normal_a[2] * normal_b[1]
        det_int == 0 && continue

        B = cheb_add(
            B,
            cheb_scale(
                product_except(facets, a, b; progress),
                interval_constant(det_int^2) / exact(4),
            ),
        )
    end

    return (; lprod, lprod_v11, lprod_v12, lprod_v22, B)
end

# worker
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

    D = cheb_add(
        cheb_add(
            v.B,
            cheb_sub(
                cheb_add(D_v22_h11, D_v11_h22),
                cheb_scale(D_v12_h12, interval_constant(2)),
            ),
        ),
        D_lprod_det_h,
    )

    A11_lprod_h22 = cheb_mul_fast(v.lprod, h22)
    advance_progress!(progress)

    A12_lprod_h12 = cheb_mul_fast(v.lprod, h12)
    advance_progress!(progress)

    A22_lprod_h11 = cheb_mul_fast(v.lprod, h11)
    advance_progress!(progress)

    A11 = cheb_add(v.lprod_v22, A11_lprod_h22)
    A12 = cheb_scale(cheb_add(v.lprod_v12, A12_lprod_h12), interval_constant(-1))
    A22 = cheb_add(v.lprod_v11, A22_lprod_h11)

    return (; A11, A12, A21 = A12, A22, D, canonical = v)
end