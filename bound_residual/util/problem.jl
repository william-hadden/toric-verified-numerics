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
    raw_coeffs = load_rational_coeffs_csv(coeffs_path)

    # This sampled estimate of c is intentionally provisional and can later be
    # replaced by a more canonical computation of the residual constant.
    c_est = evaluate_raw_F_at_reference(raw_coeffs; ref_index)

    normalized_coeffs = cheb_add_constant(raw_coeffs, -c_est / exact(2))
    normalized_ref_value = evaluate_raw_F_at_reference(normalized_coeffs; ref_index)

    return (;
        raw_coeffs,
        normalized_coeffs,
        c_est,
        ref_index,
        normalized_ref_value,
    )
end

"""
Compute the Chebyshev values of G and H on the grid.
"""
function step2_compute_GH_values_grid(coeffs::AbstractMatrix{<:Number}; pdeg::Integer=240)
    coeffs_pad = cheb_pad(coeffs, pdeg - 1, pdeg - 1)
    u_vals = cheb_coeffs_to_lobatto_values_2d(coeffs_pad)

    pack = build_lobatto_derivative_pack(u_vals; pdeg)

    G = similar(u_vals)
    H = similar(u_vals)

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

        H[i,j] = exact(2) * u_vals[i,j] - exact(2) * (x * pack.ux[i,j] + y * pack.uy[i,j])
    end

    return (; G, H, pack)
end


"""
Bound the derivatives of the MA equation on the Lobatto grid, using the Chebyshev values of G and H on the grid.
"""
function step3_compute_MA_derivatives(H_values::AbstractMatrix{<:Number}, G_values::AbstractMatrix{<:Number}; pdeg::Integer=240)
    println("Computing H_coeffs...")
    H_coeffs = cheb_lobatto_values_to_coeffs_2d(H_values)
    println("Computing G_coeffs...")
    G_coeffs = cheb_lobatto_values_to_coeffs_2d(G_values)

    println("Computing H_bound...")
    H_bound = chebyshev_coeff_sup_bound(H_coeffs)
    println("Computing exp_H_bound...")
    exp_H_bound = exp(H_bound)

    println("Computing H derivative vals...")
    H_pack = build_lobatto_first_derivative_pack(H_values; pdeg)

    function differentiate_MA_factor_vals(MAa_factor_vals, direction::Symbol)
        MAa_pack = build_lobatto_first_derivative_pack(MAa_factor_vals; pdeg)
        if direction == :x
            return MAa_pack.ux .+ H_pack.ux .* MAa_factor_vals
        elseif direction == :y
            return MAa_pack.uy .+ H_pack.uy .* MAa_factor_vals
        end
    end

    function bound_MA_factor(label::AbstractString, MAa_factor_vals)
        println("Computing $(label) factor coeffs...")
        MAa_factor_coeffs = cheb_lobatto_values_to_coeffs_2d(MAa_factor_vals)
        println("Computing $(label) bound...")
        MAa_bound = exp_H_bound * chebyshev_coeff_sup_bound(MAa_factor_coeffs)
        println("$(label) bound: $(MAa_bound)")
        return MAa_factor_coeffs, MAa_bound
    end

    println("Computing MAx factor vals...")
    MAx_factor_vals = differentiate_MA_factor_vals(G_values, :x)
    MAx_factor_coeffs, MAx_bound = bound_MA_factor("MAx", MAx_factor_vals)

    println("Computing MAy factor vals...")
    MAy_factor_vals = differentiate_MA_factor_vals(G_values, :y)
    MAy_factor_coeffs, MAy_bound = bound_MA_factor("MAy", MAy_factor_vals)

    println("Computing MAxx factor vals...")
    MAxx_factor_vals = differentiate_MA_factor_vals(MAx_factor_vals, :x)
    MAxx_factor_coeffs, MAxx_bound = bound_MA_factor("MAxx", MAxx_factor_vals)

    println("Computing MAxy factor vals...")
    MAxy_factor_vals = differentiate_MA_factor_vals(MAx_factor_vals, :y)
    MAxy_factor_coeffs, MAxy_bound = bound_MA_factor("MAxy", MAxy_factor_vals)

    println("Computing MAyy factor vals...")
    MAyy_factor_vals = differentiate_MA_factor_vals(MAy_factor_vals, :y)
    MAyy_factor_coeffs, MAyy_bound = bound_MA_factor("MAyy", MAyy_factor_vals)

    println("Computing MAxxx factor vals...")
    MAxxx_factor_vals = differentiate_MA_factor_vals(MAxx_factor_vals, :x)
    MAxxx_factor_coeffs, MAxxx_bound = bound_MA_factor("MAxxx", MAxxx_factor_vals)

    println("Computing MAxxy factor vals...")
    MAxxy_factor_vals = differentiate_MA_factor_vals(MAxx_factor_vals, :y)
    MAxxy_factor_coeffs, MAxxy_bound = bound_MA_factor("MAxxy", MAxxy_factor_vals)

    println("Computing MAxyy factor vals...")
    MAxyy_factor_vals = differentiate_MA_factor_vals(MAxy_factor_vals, :y)
    MAxyy_factor_coeffs, MAxyy_bound = bound_MA_factor("MAxyy", MAxyy_factor_vals)

    println("Computing MAyyy factor vals...")
    MAyyy_factor_vals = differentiate_MA_factor_vals(MAyy_factor_vals, :y)
    MAyyy_factor_coeffs, MAyyy_bound = bound_MA_factor("MAyyy", MAyyy_factor_vals)

    write_bound_entry("ma_first_derivatives", "x", MAx_bound)
    write_bound_entry("ma_first_derivatives", "y", MAy_bound)
    write_bound_entry("ma_derivatives", "d1", Dict(
        "x" => MAx_bound,
        "y" => MAy_bound,
    ))
    write_bound_entry("ma_derivatives", "d2", Dict(
        "xx" => MAxx_bound,
        "xy" => MAxy_bound,
        "yx" => MAxy_bound,
        "yy" => MAyy_bound,
    ))
    write_bound_entry("ma_derivatives", "d3", Dict(
        "xxx" => MAxxx_bound,
        "xxy" => MAxxy_bound,
        "xyx" => MAxxy_bound,
        "yxx" => MAxxy_bound,
        "xyy" => MAxyy_bound,
        "yxy" => MAxyy_bound,
        "yyx" => MAxyy_bound,
        "yyy" => MAyyy_bound,
    ))

    return (;
        H_coeffs,
        G_coeffs,
        MAx_bound,
        MAy_bound,
        MAxx_bound,
        MAxy_bound,
        MAyy_bound,
        MAxxx_bound,
        MAxxy_bound,
        MAxyy_bound,
        MAyyy_bound,
    )
end


"""
Bound the first derivative of the MA equation using the first partial derivative bounds.
"""
function step4_compute_MA_derivative_bound(MAx_bound, MAy_bound)
    dMA_bound = sqrt(MAx_bound^2 + MAy_bound^2)
    return (;
        dMA_bound,
    )
end

"""
Bound `||G * exp(H) - 1||_∞` using the mean value theorem.
"""
function step5_compute_residual_bound_by_MVT(H_coeffs::AbstractMatrix{<:Number}, G_coeffs::AbstractMatrix{<:Number}, dMA_bound)
    midpoint = interval(BigFloat(0.5))
    MAp0 = evaluate_coeffs_at_point(G_coeffs, midpoint, midpoint) * exp(evaluate_coeffs_at_point(H_coeffs, midpoint, midpoint))
    d = sqrt(interval(BigFloat(1)) / exact(4) + interval(BigFloat(1)) / exact(4))
    residual = abs(MAp0 - exact(1)) + dMA_bound * d

    # volume of one torus fibre is 4pi^2, volume of whole manifold 12pi^2
    manifold_volume_sqrt = sqrt(interval(BigFloat(12)) * interval(BigFloat, pi)^2)
    write_bound_entry("ma_residual_bounds", "C0", residual)
    write_bound_entry("ma_sobolev_bounds", "L2", manifold_volume_sqrt * residual)

    return (;
        residual_bound = residual
    )
end


"""
Compute the first covariant derivative bound for `u = MA`.

We bound the pointwise squared norm by
`|nabla u|^2 = G^{ij} d_i u d_j u`, then multiply by `12*pi^2` and
take the square root to get an `L^2` norm bound.
"""
function step_nabla()
    axes = ("x", "y")
    metric_inverse = read_bound("metric_inverse")["value"]
    ma_first_derivatives = read_bound("ma_first_derivatives")

    nabla_MA_C0_squared_bound = interval(BigFloat(0))
    for i in axes, j in axes
        nabla_MA_C0_squared_bound +=
            metric_inverse[i * j] *
            ma_first_derivatives[i] *
            ma_first_derivatives[j]
    end

    nabla_MA_L2_bound = sqrt(
        interval(BigFloat(12)) *
        interval(BigFloat, pi)^2 *
        nabla_MA_C0_squared_bound
    )

    write_bound_entry("ma_sobolev_bounds", "nabla_L2", nabla_MA_L2_bound)
    return nothing
end

"""
Compute a `C^0` upper bound for `|Delta u|`, where `u = MA`.

For torus-invariant functions in symplectic coordinates,
`Delta u = d_i(G^{ij} d_j u)`, so
`Delta u = (d_i G^{ij}) d_j u + G^{ij} d_i d_j u`.
"""
function bound_laplace_MA()
    axes = ("x", "y")
    metric_inverse = read_bound("metric_inverse")
    ma_derivatives = read_bound("ma_derivatives")

    bound = interval(BigFloat(0))
    for i in axes, j in axes
        metric_component = i * j
        bound +=
            metric_inverse["d1"][i][metric_component] *
            ma_derivatives["d1"][j]
        bound +=
            metric_inverse["value"][metric_component] *
            ma_derivatives["d2"][metric_component]
    end

    return bound
end

function nabla_2_from_laplace_bound(laplace_bound::Number, nabla_bound::Number)
    ricci_C0_bound = read_bound("curvature_bounds")["ricci_C0"]
    return sqrt(laplace_bound^2 + ricci_C0_bound * nabla_bound^2)
end

function step_nabla2()
    laplace_MA_C0_bound = bound_laplace_MA()
    laplace_MA_L2_bound = sqrt(
        interval(BigFloat(12)) *
        interval(BigFloat, pi)^2 *
        laplace_MA_C0_bound^2
    )
    nabla_MA_L2_bound = read_bound("ma_sobolev_bounds")["nabla_L2"]
    nabla2_L2_bound = nabla_2_from_laplace_bound(laplace_MA_L2_bound, nabla_MA_L2_bound)

    write_bound_entry("ma_sobolev_bounds", "nabla2_L2", nabla2_L2_bound)
    return nothing
end

"""
Compute a `C^0` upper bound for `|nabla Delta u|^2`, where `u = MA`.

For torus-invariant functions in symplectic coordinates, the Laplacian is
`Delta u = d_i(G^{ij} d_j u)`. Here `G^{ij}` denotes the inverse metric
coefficients, and repeated indices range over `x` and `y`. Differentiating
once more in direction `k` gives
`d_k Delta u = d_k d_i(G^{ij} d_j u)`.
Applying the product rule twice gives the fully expanded expression
`d_k d_i(G^{ij} d_j u)`
`= (d_k d_i G^{ij}) d_j u`
`+ (d_i G^{ij}) d_k d_j u`
`+ (d_k G^{ij}) d_i d_j u`
`+ G^{ij} d_k d_i d_j u`.
Then use the bounds from the individual terms from JSON.
"""
function bound_nabla_laplace_MA()
    axes = ("x", "y")
    metric_inverse = read_bound("metric_inverse")
    ma_derivatives = read_bound("ma_derivatives")

    # d_k d_i(G^{ij} d_j u)
    # = (d_k d_i G^{ij}) d_j u
    # + (d_i G^{ij}) d_k d_j u
    # + (d_k G^{ij}) d_i d_j u
    # + G^{ij} d_k d_i d_j u
    derivative_bounds = Dict{String, Interval{BigFloat}}()
    for k in axes
        total = interval(BigFloat(0))
        for i in axes, j in axes
            metric_component = i * j
            total +=
                metric_inverse["d2"][k * i][metric_component] *
                ma_derivatives["d1"][j]
            total +=
                metric_inverse["d1"][i][metric_component] *
                ma_derivatives["d2"][k * j]
            total +=
                metric_inverse["d1"][k][metric_component] *
                ma_derivatives["d2"][i * j]
            total +=
                metric_inverse["value"][metric_component] *
                ma_derivatives["d3"][k * i * j]
        end
        derivative_bounds[k] = total
    end

    bound = interval(BigFloat(0))
    for k in axes, l in axes
        bound +=
            metric_inverse["value"][k * l] *
            derivative_bounds[k] *
            derivative_bounds[l]
    end

    return bound
end

"""
Apply the `D^3u` estimate from the integrated Bochner/commutator proposition.

For `u = MA`, the proposition gives
`||nabla^3 u||_L2^2 <= 2 ||nabla Delta u||_L2^2
    + Cl{D^3u-estimate-second-summand} * ||u||_L2_2^2`,
where
`Cl{D^3u-estimate-second-summand} = 2 K1^2 + 3n K2 + n K3`.

Here `n = 4`, `K1 = ||Ric||_C0`, `K2 = ||Riem||_C0`, and
`K3 = ||nabla Riem||_C0`.
The input `nabla_laplace_MA_L2_squared_bound` is
`||nabla Delta u||_L2^2`. The lower Sobolev norm uses the project
convention `||u||_L2_2 = ||u||_L2 + ||nabla u||_L2 + ||nabla^2 u||_L2`.
"""
function nabla_3_from_laplace_bound(nabla_laplace_MA_L2_squared_bound::Number)
    curvature_bounds = read_bound("curvature_bounds")
    K1 = curvature_bounds["ricci_C0"]
    K2 = curvature_bounds["riemann_C0"]
    K3 = curvature_bounds["nabla_riemann_C0"]
    n = interval(BigFloat(4))
    Cl_D3u_estimate_second_summand =
        interval(BigFloat(2)) * K1^2 +
        interval(BigFloat(3)) * n * K2 +
        n * K3

    ma_sobolev_bounds = read_bound("ma_sobolev_bounds")
    MA_L2_bound = ma_sobolev_bounds["L2"]
    MA_L2_2_bound =
        MA_L2_bound +
        ma_sobolev_bounds["nabla_L2"] +
        ma_sobolev_bounds["nabla2_L2"]

    return sqrt(
        interval(BigFloat(2)) * nabla_laplace_MA_L2_squared_bound +
        Cl_D3u_estimate_second_summand * MA_L2_2_bound^2
    )
end

function step_nabla3()
    nabla_laplace_MA_C0_squared_bound = bound_nabla_laplace_MA()
    manifold_volume = interval(BigFloat(12)) * interval(BigFloat, pi)^2
    nabla_laplace_MA_L2_squared_bound = manifold_volume * nabla_laplace_MA_C0_squared_bound
    nabla3_L2_bound = nabla_3_from_laplace_bound(nabla_laplace_MA_L2_squared_bound)

    write_bound_entry("ma_sobolev_bounds", "nabla3_L2", nabla3_L2_bound)
    return nothing
end
