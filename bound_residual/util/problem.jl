"""
Differentiate a one-variable Chebyshev series in coefficient space.

If
`f(t) = sum_{k=0}^N a_k T_k(t)` and `f'(t) = sum_{k=0}^{N-1} b_k T_k(t)`,
then the coefficients satisfy
`b_{N-1} = 2N a_N`,
`b_k = b_{k+2} + 2(k+1) a_{k+1}` for `k = N-2, ..., 1`,
and `b_0 = a_1 + b_2 / 2`.

This is the standard first-kind Chebyshev differentiation recurrence. In the
notation of Mason and Handscomb, `Chebyshev Polynomials`, §2.4.5, equations
`(2.50)`-`(2.51)`, our input coefficients `{a_k}` are their `{A_r}` and our
output coefficients `{b_k}` are their `{a_r}`. The implementation keeps one
trailing zero mode so the output has the same length as the input coefficient
vector.
"""
function cheb_diff_1d(coeffs::AbstractVector{<:Number})
    N = length(coeffs) - 1
    deriv = zeros(eltype(coeffs), N + 1)
    N == 0 && return deriv

    deriv[N] = exact(2 * N) * coeffs[N + 1]
    for k in (N - 2):-1:1
        deriv[k + 1] = deriv[k + 3] + exact(2 * (k + 1)) * coeffs[k + 2]
    end
    deriv[1] = N == 1 ? coeffs[2] : coeffs[2] + deriv[3] / exact(2)

    return deriv
end

"""
Differentiate a coefficient array in the `x` variable on `[0,1]`.
"""
function differentiate_coeffs_x(coeffs::AbstractMatrix{<:Number})
    out = zeros(eltype(coeffs), size(coeffs))
    for j in axes(coeffs, 2)
        out[:, j] .= exact(2) .* cheb_diff_1d(@view coeffs[:, j])
    end
    return out
end

"""
Differentiate a coefficient array in the `y` variable on `[0,1]`.
"""
function differentiate_coeffs_y(coeffs::AbstractMatrix{<:Number})
    out = zeros(eltype(coeffs), size(coeffs))
    for i in axes(coeffs, 1)
        out[i, :] .= -exact(2) .* cheb_diff_1d(vec(@view coeffs[i, :]))
    end
    return out
end

"""
Assemble `u0` together with its first and second derivative coefficient arrays.
"""
function build_derivative_pack(coeffs::AbstractMatrix{<:Number})
    u0 = copy(coeffs)
    ux = differentiate_coeffs_x(u0)
    uy = differentiate_coeffs_y(u0)
    uxx = differentiate_coeffs_x(ux)
    uyy = differentiate_coeffs_y(uy)
    uxy = (differentiate_coeffs_y(ux) .+ differentiate_coeffs_x(uy)) ./ exact(2)
    return (u0 = u0, ux = ux, uy = uy, uxx = uxx, uyy = uyy, uxy = uxy)
end

"""
Evaluate the scalar equation coefficients from equation `(34)` of Doran et al.,
`Numerical Kähler-Einstein metric on the third del Pezzo`, at one point.
"""
function eqn_coefficients_at_point(x, y)
    c1a = -((x - exact(1)) * (exact(1) + x) * (x - y - exact(1)) * (exact(1) + x - y) * (y^2 - exact(1)))
    c1b = (x - exact(1)) * (exact(1) + x) * (-exact(1) - y) * (exact(1) - y) * (x - y - exact(1)) * (exact(1) + x - y)
    c2xx = (x^2 - exact(1)) * (x^2 - exact(2) * x * y + exact(2) * y^2 - exact(2))
    c2yy = (y^2 - exact(1)) * (exact(2) * x^2 - exact(2) * x * y + y^2 - exact(2))
    c2xy = -exact(2) * (x^2 - exact(1)) * (y^2 - exact(1))
    c3 = exact(3) - exact(2) * (x^2 - x * y + y^2)
    return (; c1a, c1b, c2xx, c2yy, c2xy, c3)
end

"""
Return the coefficient arrays for `1`, `x`, and `y` on the stored basis.
"""
function coordinate_series()
    one_series = cheb_constant(1.0)
    x_series = zeros(Float64, 2, 1)
    y_series = zeros(Float64, 1, 2)
    x_series[1, 1] = 0.5
    x_series[2, 1] = 0.5
    y_series[1, 1] = 0.5
    y_series[1, 2] = -0.5
    return (; one_series, x_series, y_series)
end

"""
Build the coefficient arrays of the fixed equation coefficients in equation
`(34)` of Doran et al., `Numerical Kähler-Einstein metric on the third del Pezzo`.
"""
function build_eqn_coefficients()
    coords = coordinate_series()
    one_series = coords.one_series
    x_series = coords.x_series
    y_series = coords.y_series

    x_minus_one = cheb_add_constant(x_series, -1.0)
    one_plus_x = cheb_add_constant(x_series, 1.0)
    x_minus_y_minus_one = cheb_add(cheb_sub(x_series, y_series), cheb_constant(-1.0))
    one_plus_x_minus_y = cheb_add(cheb_sub(x_series, y_series), one_series)
    y_squared_minus_one = cheb_add_constant(cheb_mul2(y_series, y_series), -1.0)

    minus_one_minus_y = cheb_add_constant(cheb_scale(y_series, -1.0), -1.0)
    one_minus_y = cheb_add_constant(cheb_scale(y_series, -1.0), 1.0)
    x_squared_minus_one = cheb_add_constant(cheb_mul2(x_series, x_series), -1.0)
    x_times_y = cheb_mul2(x_series, y_series)
    y_squared = cheb_mul2(y_series, y_series)
    x_squared = cheb_mul2(x_series, x_series)

    c1a = cheb_scale(
        cheb_mul2(
            cheb_mul2(
                cheb_mul2(
                    cheb_mul2(x_minus_one, one_plus_x),
                    x_minus_y_minus_one,
                ),
                one_plus_x_minus_y,
            ),
            y_squared_minus_one,
        ),
        -1.0,
    )

    c1b = cheb_mul2(
        cheb_mul2(
            cheb_mul2(
                cheb_mul2(
                    cheb_mul2(x_minus_one, one_plus_x),
                    minus_one_minus_y,
                ),
                one_minus_y,
            ),
            x_minus_y_minus_one,
        ),
        one_plus_x_minus_y,
    )

    c2xx = cheb_mul2(
        x_squared_minus_one,
        cheb_add_constant(
            cheb_add(
                cheb_sub(x_squared, cheb_scale(x_times_y, 2)),
                cheb_scale(y_squared, 2),
            ),
            -2,
        ),
    )

    c2yy = cheb_mul2(
        y_squared_minus_one,
        cheb_add_constant(
            cheb_add(
                cheb_sub(cheb_scale(x_squared, 2), cheb_scale(x_times_y, 2)),
                y_squared,
            ),
            -2,
        ),
    )

    c2xy = cheb_scale(cheb_mul2(x_squared_minus_one, y_squared_minus_one), -2)
    c3 = cheb_add_constant(
        cheb_sub(
            cheb_scale(one_series, 3),
            cheb_scale(cheb_sub(cheb_add(x_squared, y_squared), x_times_y), 2),
        ),
        0.0,
    )

    return (; c1a, c1b, c2xx, c2yy, c2xy, c3)
end

"""
Compute the coefficient array of `H = 2u0 - 2(x ux + y uy)`.
RHS of 3.27 - \rho_can - c
"""
function compute_H_coeffs(pack)
    return cheb_scale(
        cheb_sub(
            pack.u0,
            cheb_add(cheb_mul_x(pack.ux), cheb_mul_y(pack.uy)),
        ),
        2,
    )
end

"""
Compute the coefficient array of the polynomial factor `G`.
"""
function compute_G_coeffs(pack, eqn_coeffs)
    term1 = cheb_mul2(eqn_coeffs.c1a, cheb_mul2(pack.uxx, pack.uyy))
    term2 = cheb_mul2(eqn_coeffs.c1b, cheb_mul2(pack.uxy, pack.uxy))
    term3 = cheb_mul2(eqn_coeffs.c2xx, pack.uxx)
    term4 = cheb_mul2(eqn_coeffs.c2yy, pack.uyy)
    term5 = cheb_mul2(eqn_coeffs.c2xy, pack.uxy)
    return cheb_add(
        cheb_sub(
            cheb_add(cheb_add(term1, term2), cheb_add(term3, term4)),
            term5,
        ),
        eqn_coeffs.c3,
    )
end

"""
Estimate the additive constant in `F` from one interior collocation point.
"""
function evaluate_raw_F_at_reference(coeffs::AbstractMatrix{<:Number}; ref_index::Tuple{Int, Int} = DEFAULT_REF_INDEX)
    pack = build_derivative_pack(coeffs)
    xarr, yarr = make_grids(size(coeffs, 1) - 1)
    x = xarr[ref_index[1]]
    y = yarr[ref_index[2]]

    u = evaluate_coeffs_at_point(pack.u0, x, y)
    ux = evaluate_coeffs_at_point(pack.ux, x, y)
    uy = evaluate_coeffs_at_point(pack.uy, x, y)
    uxx = evaluate_coeffs_at_point(pack.uxx, x, y)
    uyy = evaluate_coeffs_at_point(pack.uyy, x, y)
    uxy = evaluate_coeffs_at_point(pack.uxy, x, y)

    eqn_coeffs = eqn_coefficients_at_point(x, y)
    G = eqn_coeffs.c1a * uxx * uyy + eqn_coeffs.c1b * (uxy^2) + eqn_coeffs.c2xx * uxx + eqn_coeffs.c2yy * uyy - eqn_coeffs.c2xy * uxy + eqn_coeffs.c3
    H = exact(2) * u - exact(2) * (x * ux + y * uy)

    return log(G) + H
end

"""
Load `u0` and shift only its constant Chebyshev mode so the sampled `F` is near zero.
"""
function step1_load_and_normalize_u0(coeffs_path::AbstractString; ref_index::Tuple{Int, Int} = DEFAULT_REF_INDEX)
    # raw_coeffs = load_coeffs_csv(coeffs_path)
    raw_coeffs = interval.(load_coeffs_csv(coeffs_path))

    # This sampled estimate of c is intentionally provisional and can later be
    # replaced by a more canonical computation of the residual constant.
    c_est = evaluate_raw_F_at_reference(raw_coeffs; ref_index)

    normalized_coeffs = cheb_add_constant(raw_coeffs, -c_est / exact(2))
    normalized_ref_value = evaluate_raw_F_at_reference(normalized_coeffs; ref_index)
    pack = build_derivative_pack(normalized_coeffs)

    return (;
        raw_coeffs,
        normalized_coeffs,
        pack,
        c_est,
        ref_index,
        normalized_ref_value,
    )
end

"""
Compute coarse `C^0` bounds for `H` and `G` from coefficient sums.
"""
function step2_coarse_sup_bounds(pack)
    eqn_coeffs = build_eqn_coefficients()
    H_coeffs = cheb_pad(compute_H_coeffs(pack), size(pack.u0, 1) - 1, size(pack.u0, 2) - 1)
    G_coeffs = cheb_pad(compute_G_coeffs(pack, eqn_coeffs), 162, 162)
    H_bound = chebyshev_coeff_sup_bound(H_coeffs)
    G_bound = chebyshev_coeff_sup_bound(G_coeffs)

    return (;
        H_coeffs,
        G_coeffs,
        H_bound,
        G_bound,
    )
end

"""
Bound `sup_{|t|<=M} |exp(t) - exp_p(t)|` by the positive exponential tail at `M`.

This uses
`sup_{|t|<=M} |exp(t) - exp_p(t)| <= sum_{k=p+1}^∞ M^k / k!`.

Write `a_k = M^k / k!`. Then
`a_{k+1} / a_k = M / (k + 1)`.
Once this ratio is less than `1`, the remaining terms decrease at least
geometrically. More precisely, if `q = M / (k + 1) < 1`, then for every
`j >= k`,
`a_{j+1} / a_j = M / (j + 1) <= q`,
so
`sum_{j=k}^∞ a_j <= a_k * sum_{m=0}^∞ q^m = a_k / (1 - q)`.

The code first computes the recurrence
`a_{k+1} = a_k * M / (k + 1)`
until `q = M / (k + 1) < 1` becomes valid, adding any necessary initial
terms explicitly, and then computes the tail bound using the geometric series.
"""
function exp_tail_sup_bound(M::Real, p::Integer)
    M >= 0.0 || error("Expected a nonnegative sup bound")
    p >= 0 || error("Taylor degree must be nonnegative")

    term = 1.0
    for k in 1:p
        term *= M / k
    end
    term *= M / (p + 1)

    tail = 0.0
    k = p + 1
    while M / (k + 1) >= 1.0
        tail += term
        k += 1
        term *= M / k
    end

    ratio = M / (k + 1)
    tail += term / (1.0 - ratio)

    return tail
end

function exp_tail_sup_bound(M::Interval{Float64}, p::Integer)::Interval{Float64}
    sup(M) >= 0.0 || error("Expected a nonnegative sup bound")
    p >= 0 || error("Taylor degree must be nonnegative")

    return Interval(exp_tail_sup_bound(sup(M), p))
end

"""
Bound `||G * (exp(H) - exp_p(H))||_∞` using coarse sup bounds for `G` and `H`.
"""
function step3_taylor_tail_bound(H_bound, G_bound, p::Integer)
    total_bound = G_bound * exp_tail_sup_bound(H_bound, p)
    return (; total_bound)
end

"""
Bound `||G * exp_p(H) - 1||_∞` entirely in coefficient space.
"""
function step4_poly_bound(H_coeffs::AbstractMatrix{<:Number}, G_coeffs::AbstractMatrix{<:Number}, p::Integer)
    exp_p_H = cheb_horner_scalar_poly(H_coeffs, exp_taylor_coeffs(p))
    P_coeffs = cheb_add_constant(cheb_mul2(G_coeffs, exp_p_H), -1.0)
    P_coeffs = cheb_pad(P_coeffs, 162 + 79 * p, 162 + 79 * p)
    poly_bound = chebyshev_coeff_sup_bound(P_coeffs)

    return (;
        grid_degree = size(P_coeffs, 1) - 1,
        P_coeffs,
        poly_bound,
    )
end

"""
Bound the derivatives of the MA equation in coefficient space, using the coarse sup bounds for `H`.
"""
function step3_compute_MA_derivative_bound(H_coeffs::AbstractMatrix{<:Number}, G_coeffs::AbstractMatrix{<:Number}, H_bound)
    H = copy(H_coeffs)
    Hx_coeffs = differentiate_coeffs_x(H)
    Hy_coeffs = differentiate_coeffs_y(H)

    G = copy(G_coeffs)
    Gx_coeffs = differentiate_coeffs_x(G)
    Gy_coeffs = differentiate_coeffs_y(G)

    MAx_bound = exp(H_bound) * chebyshev_coeff_sup_bound(cheb_add(Gx_coeffs, cheb_mul2(G_coeffs, Hx_coeffs)))
    MAy_bound = exp(H_bound) * chebyshev_coeff_sup_bound(cheb_add(Gy_coeffs, cheb_mul2(G_coeffs, Hy_coeffs)))
    dMA_bound = sqrt(MAx_bound^2 + MAy_bound^2)

    return (;
        MAx_bound,
        MAy_bound,
        dMA_bound
    )
end

"""
Bound `||G * exp(H) - 1||_∞` using the mean value theorem.
"""
function step4_compute_residual_bound_by_MVT(H_coeffs::AbstractMatrix{<:Number}, G_coeffs::AbstractMatrix{<:Number}, dMA_bound)
    MAp0 = evaluate_coeffs_at_point(G_coeffs, 0.5, 0.5) * exp(evaluate_coeffs_at_point(H_coeffs, 0.5, 0.5)) 
    d = sqrt(1/4 + 1/4) 
    residual = abs(MAp0-1) + dMA_bound * d
    return (;
        residual_bound = residual
    )
end

"""
Compute the Chebyshev values of G and H on the grid.
"""
function step2_compute_GH_values_grid(coeffs::AbstractMatrix{<:Number}; pdeg::Integer=240)
    pack = build_lobatto_derivative_pack(coeffs; pdeg)

    G = similar(pack.u)
    H = similar(pack.u)

    for j in 1:pdeg, i in 1:pdeg
        x = pack.xarr[i]
        y = pack.yarr[j]

        c = eqn_coefficients_at_point(x, y)

        G[i,j] =
            c.c1a  * pack.uxx[i,j] * pack.uyy[i,j] +
            c.c1b  * pack.uxy[i,j]^2 +
            c.c2xx * pack.uxx[i,j] +
            c.c2yy * pack.uyy[i,j] -
            c.c2xy * pack.uxy[i,j] +
            c.c3

        H[i,j] = 2 * pack.u[i,j] - 2 * (x * pack.ux[i,j] + y * pack.uy[i,j])
    end

    return (; G, H, pack)
end


"""
Bound the derivatives of the MA equation in coefficient space, using the Chebyshev values of G and H on the grid.
"""
function step3_compute_MA_derivative_bound_v2(H_values::AbstractMatrix{<:Number}, G_values::AbstractMatrix{<:Number})
    H_coeffs = cheb_lobatto_values_to_coeffs_2d(H_values)
    G_coeffs = cheb_lobatto_values_to_coeffs_2d(G_values)

    H_bound = chebyshev_coeff_sup_bound(H_coeffs)

    Hx_coeffs = differentiate_coeffs_x(H_coeffs)
    Hy_coeffs = differentiate_coeffs_y(H_coeffs)

    Gx_coeffs = differentiate_coeffs_x(G_coeffs)
    Gy_coeffs = differentiate_coeffs_y(G_coeffs)

    MAx_coeffs = cheb_add(Gx_coeffs, cheb_mul2(G_coeffs, Hx_coeffs))
    MAy_coeffs = cheb_add(Gy_coeffs, cheb_mul2(G_coeffs, Hy_coeffs))

    exp_H_bound = exp(H_bound)

    MAx_bound = exp_H_bound * chebyshev_coeff_sup_bound(MAx_coeffs)
    MAy_bound = exp_H_bound * chebyshev_coeff_sup_bound(MAy_coeffs)
    dMA_bound = sqrt(MAx_bound^2 + MAy_bound^2)

    return (;
        H_coeffs,
        G_coeffs,
        MAx_bound,
        MAy_bound,
        dMA_bound,
    )
end
