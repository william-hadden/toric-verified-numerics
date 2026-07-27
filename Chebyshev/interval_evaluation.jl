# Interval and local-box evaluation of tensor-product Chebyshev series.
"""
Evaluate T_0(x), ..., T_N(x) on an interval x using recurrence.
"""
function chebyshev_values(x, N::Integer)
    N >= 0 || error("Chebyshev degree must be nonnegative")
    T = Vector{typeof(x)}(undef, N + 1)

    T[1] = one(x)          # T_0
    if N >= 1
        T[2] = x           # T_1
    end

    for k in 2:N
        T[k + 1] = exact(2) * x * T[k] - T[k - 1]
    end

    return T
end

"""
Evaluate a 2D Chebyshev coefficient array on an interval box
xbox by ybox.

coeffs[i,j] corresponds to T_{i-1}(x) T_{j-1}(y).
"""
function eval_cheb_2d_interval(coeffs::AbstractMatrix, xbox, ybox)
    Nx, Ny = size(coeffs)

    Tx = chebyshev_values(xbox, Nx - 1)
    Ty = chebyshev_values(ybox, Ny - 1)

    s = zero(coeffs[1, 1] * Tx[1] * Ty[1])

    for j in 1:Ny, i in 1:Nx
        s += coeffs[i, j] * Tx[i] * Ty[j]
    end

    return s
end

"""Return the Chebyshev-Lobatto points `cos(pi*k/N)` on `[-1,1]`."""
cheb_grid(N::Integer) = cospi.(interval.(BigFloat.(collect(0:N))) ./ exact(BigFloat(N)))

"""Return the tensor-product grids on `[0,1]^2` used by exported coefficients."""
function make_grids(N::Integer)
    half = exact(BigFloat(0.5))
    xarr = reverse(half .+ half .* cheb_grid(N))
    return xarr, copy(xarr)
end

"""Evaluate an exported tensor-product Chebyshev series at `(x, y)` in `[0,1]^2`."""
function evaluate_coeffs_at_point(coeffs::AbstractMatrix{<:Number}, x::Real, y::Real)
    Tx = chebyshev_values(exact(2) * x - exact(1), size(coeffs, 1) - 1)
    Ty = chebyshev_values(exact(1) - exact(2) * y, size(coeffs, 2) - 1)
    T = promote_type(eltype(coeffs), eltype(Tx))
    value = zero(T)

    for j in axes(coeffs, 2)
        inner = zero(T)
        for i in axes(coeffs, 1)
            inner += coeffs[i, j] * Tx[i]
        end
        value += inner * Ty[j]
    end

    return value
end

"""
Split [-1,1] into n equal interval pieces.
"""
function subdivide_minus_one_one(n::Integer)
    xs = range(big"-1", big"1"; length = n + 1)

    return [
        interval(BigFloat(xs[k]), BigFloat(xs[k + 1]))
        for k in 1:n
    ]
end

function affine_box_to_unit(box)
    l = inf(box)
    r = sup(box)
    li = interval(l)
    ri = interval(r)
    a = (li + ri) / exact(2)
    b = (ri - li) / exact(2)
    return interval(a), interval(b)
end

function cheb_shifted_power_coeffs(n::Integer, a, b)
    T = Vector{Vector{typeof(a)}}(undef, n + 1)

    T[1] = [one(a)]

    if n >= 1
        T[2] = [a, b]
    end

    for k in 2:n
        Tk = T[k]
        Tkm1 = T[k - 1]

        xTk = zeros(typeof(a), length(Tk) + 1)

        for i in eachindex(Tk)
            xTk[i] += a * Tk[i]
            xTk[i + 1] += b * Tk[i]
        end

        Tkp1 = exact(2) .* xTk

        for i in eachindex(Tkm1)
            Tkp1[i] -= Tkm1[i]
        end

        T[k + 1] = Tkp1
    end

    return T
end

function local_power_coeffs_cheb_2d(coeffs::AbstractMatrix, xbox, ybox)
    coeffs = intervalize_coefficients(coeffs)

    Nx, Ny = size(coeffs)

    ax, bx = affine_box_to_unit(xbox)
    ay, by = affine_box_to_unit(ybox)

    Tx = cheb_shifted_power_coeffs(Nx - 1, ax, bx)
    Ty = cheb_shifted_power_coeffs(Ny - 1, ay, by)

    local2 = zeros(typeof(coeffs[1, 1]), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        cij = coeffs[i, j]
        px = Tx[i]
        py = Ty[j]

        for q in eachindex(py), p in eachindex(px)
            local2[p, q] += cij * px[p] * py[q]
        end
    end

    return local2
end

"""
Enclose a 2D Chebyshev series on a local interval box by coefficient summation.

The Chebyshev series is first rewritten as a power series in local coordinates
centered on `xbox x ybox`. The constant coefficient `c00` is the centered value
part, and the sum of absolute values of all nonconstant local coefficients gives
a rigorous remainder bound because the local coordinates range over `[-1, 1]`.

Returns the interval `c00 + [-tail, tail]`.
"""
function local_coeff_sum_centered_enclosure_cheb_2d(coeffs::AbstractMatrix, xbox, ybox)
    local2 = local_power_coeffs_cheb_2d(coeffs, xbox, ybox)

    Nx, Ny = size(local2)

    c00 = local2[1, 1]
    tail = zero(abs(c00))

    for q in 1:Ny, p in 1:Nx
        p == 1 && q == 1 && continue
        tail += abs(local2[p, q])
    end

    return c00 + symmetric_interval(tail)
end
