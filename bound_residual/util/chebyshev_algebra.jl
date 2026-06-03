const _CHEB_HAS_FFTW = let
    try
        @eval import FFTW
        true
    catch
        false
    end
end

const _CHEB_INTERVAL_EVAL_CACHE = Dict{Tuple{Int, DataType, Int}, Any}()
const _CHEB_INTERVAL_RECOVERY_CACHE = Dict{Tuple{Int, DataType, Int}, Any}()

"""
Return a dense zero coefficient array with the requested bidegree.
"""
function cheb_zero(degx::Integer, degy::Integer, ::Type{T} = Float64) where {T<:Number}
    degx >= 0 || error("Expected a nonnegative x-degree")
    degy >= 0 || error("Expected a nonnegative y-degree")
    return zeros(T, degx + 1, degy + 1)
end

coeff_is_exact_zero(x::Number) = iszero(x)
coeff_is_exact_zero(x::Interval) = isequal_interval(x, zero(x))

cheb_interval_endpoint_type(::Type{Interval{R}}) where {R<:Real} = R

function cheb_interval_precision(::Type{Interval{R}}) where {R<:Real}
    if R === BigFloat
        return precision(BigFloat)
    elseif R <: AbstractFloat
        return precision(R)
    else
        return 0
    end
end

cheb_interval_cache_key(N::Integer, ::Type{T}) where {T<:Interval} = (N, T, cheb_interval_precision(T))

cheb_interval_constant(::Type{T}, x) where {T<:Interval} = interval(cheb_interval_endpoint_type(T)(x))

"""
Trim exact zero rows and columns from the outer boundary of a coefficient array.
"""
function cheb_trim_exact(coeffs::AbstractMatrix{<:Number})
    last_row = size(coeffs, 1)
    while last_row > 1 && all(coeff_is_exact_zero, @view coeffs[last_row, :])
        last_row -= 1
    end

    last_col = size(coeffs, 2)
    while last_col > 1 && all(coeff_is_exact_zero, @view coeffs[:, last_col])
        last_col -= 1
    end

    return copy(@view coeffs[1:last_row, 1:last_col])
end

"""
Pad a coefficient array with zeros up to the requested bidegree.
"""
function cheb_pad(coeffs::AbstractMatrix{<:Number}, degx::Integer, degy::Integer)
    degx >= size(coeffs, 1) - 1 || error("Requested x-degree is too small")
    degy >= size(coeffs, 2) - 1 || error("Requested y-degree is too small")

    out = cheb_zero(degx, degy, eltype(coeffs))
    out[1:size(coeffs, 1), 1:size(coeffs, 2)] .= coeffs
    return out
end

"""
Return the cached interval cosine evaluation matrix for Lobatto nodes of order `N`.

Entry `(r+1, k+1)` equals `cos(pi * r * k / N)` enclosed as an interval.
"""
function cheb_interval_eval_matrix(N::Integer, ::Type{T}) where {T<:Interval}
    N >= 0 || error("Expected a nonnegative Lobatto order")
    key = cheb_interval_cache_key(N, T)

    return get!(_CHEB_INTERVAL_EVAL_CACHE, key) do
        if N == 0
            return reshape(T[one(T)], 1, 1)
        end

        matrix = Matrix{T}(undef, N + 1, N + 1)
        for r in 0:N, k in 0:N
            matrix[r + 1, k + 1] = cospi(cheb_interval_constant(T, (r * k) // N))
        end
        matrix
    end
end

"""
Return the cached interval DCT-I recovery matrix for Lobatto order `N`.

This encodes the endpoint half weights in the coefficient recovery formula.
"""
function cheb_interval_recovery_matrix(N::Integer, ::Type{T}) where {T<:Interval}
    N >= 0 || error("Expected a nonnegative Lobatto order")
    key = cheb_interval_cache_key(N, T)

    return get!(_CHEB_INTERVAL_RECOVERY_CACHE, key) do
        if N == 0
            return reshape(T[one(T)], 1, 1)
        end

        matrix = copy(cheb_interval_eval_matrix(N, T))
        matrix .*= cheb_interval_constant(T, 2 // N)
        matrix[1, :] ./= exact(2)
        matrix[end, :] ./= exact(2)
        matrix[:, 1] ./= exact(2)
        matrix[:, end] ./= exact(2)
        matrix
    end
end

"""
Add two dense coefficient arrays after zero-padding to the common bidegree.
"""
function cheb_add(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number})
    degx = max(size(A, 1), size(B, 1)) - 1
    degy = max(size(A, 2), size(B, 2)) - 1
    T = promote_type(eltype(A), eltype(B))
    out = cheb_zero(degx, degy, T)

    for j in axes(A, 2), i in axes(A, 1)
        out[i, j] += A[i, j]
    end
    for j in axes(B, 2), i in axes(B, 1)
        out[i, j] += B[i, j]
    end

    return cheb_trim_exact(out)
end

"""
Add a scalar to the constant mode of a coefficient array.
"""
function cheb_add_constant(A::AbstractMatrix{<:Number}, c)
    T = promote_type(eltype(A), typeof(c))
    out = Matrix{T}(undef, size(A)...)
    out .= A
    out[1, 1] += c
    return cheb_trim_exact(out)
end

"""
Add `scale * T_modex * T_modey * coeffs` to a two-dimensional output array.

The caller must provide `out` with enough room for the full tensor-product
update. If `coeffs` has size `(Nx + 1, Ny + 1)`, then `out` must have size at
least `(modex + Nx + 1, modey + Ny + 1)`.

Applying
`T_m T_n = (T_{m+n} + T_{|m-n|}) / 2` (DLMF 18.18.21)
in the `x` variable and again in the `y` variable produces four target modes.
"""
function cheb_add_scaled_basis_product_2d!(out::AbstractMatrix{<:Number}, coeffs::AbstractMatrix{<:Number}, modex::Integer, modey::Integer, scale)
    quarter_scale = scale / exact(4)

    for j in axes(coeffs, 2), i in axes(coeffs, 1)
        contribution = quarter_scale * coeffs[i, j]
        x_hi = i + modex
        x_lo = abs((i - 1) - modex) + 1
        y_hi = j + modey
        y_lo = abs((j - 1) - modey) + 1

        out[x_hi, y_hi] += contribution
        out[x_hi, y_lo] += contribution
        out[x_lo, y_hi] += contribution
        out[x_lo, y_lo] += contribution
    end

    return out
end

"""
Apply the unnormalized DCT-I to a real vector.

When FFTW is available this uses `FFTW.r2r(_, FFTW.REDFT00)`, which is the
unnormalized DCT-I matching the Lobatto interpolation formulas below.
Otherwise it falls back to the defining cosine sum. The fallback preserves
correctness but is not fast.
"""
function cheb_dct1(v::AbstractVector{<:Real})
    n = length(v)
    n >= 1 || error("Expected a nonempty vector")

    T = float(promote_type(eltype(v), Float64))
    out = zeros(T, n)

    if n == 1
        out[1] = T(v[1])
        return out
    end

    data = T.(v)

    if _CHEB_HAS_FFTW
        return FFTW.r2r(data, FFTW.REDFT00)
    end

    N = n - 1
    for k in 0:N
        total = data[1] + ((-one(T))^k) * data[end]
        for j in 1:(N - 1)
            total += 2 * data[j + 1] * cospi(j * k / N)
        end
        out[k + 1] = total
    end

    return out
end

"""
Convert first-kind Chebyshev coefficients to Lobatto-grid values.
"""
function cheb_coeffs_to_lobatto_values_1d(coeffs::AbstractVector{<:Real})
    n = length(coeffs)
    n >= 1 || error("Expected a nonempty coefficient vector")

    T = float(promote_type(eltype(coeffs), Float64))
    if n == 1
        return T[T(coeffs[1])]
    end

    doubled = T.(coeffs)
    doubled[1] *= 2
    doubled[end] *= 2
    return cheb_dct1(doubled) / 2
end

"""
Convert first-kind interval Chebyshev coefficients to Lobatto-grid values.
"""
function cheb_coeffs_to_lobatto_values_1d(coeffs::AbstractVector{<:Interval})
    n = length(coeffs)
    n >= 1 || error("Expected a nonempty coefficient vector")
    T = eltype(coeffs)

    if n == 1
        return [coeffs[1]]
    end

    return cheb_interval_eval_matrix(n - 1, T) * collect(coeffs)
end

"""
Convert Lobatto-grid values to first-kind Chebyshev coefficients.

If the values are sampled at the `N + 1` Lobatto points, this returns the
unique degree-`N` interpolating Chebyshev series.
"""
function cheb_lobatto_values_to_coeffs_1d(values::AbstractVector{<:Real})
    n = length(values)
    n >= 1 || error("Expected a nonempty values vector")

    T = float(promote_type(eltype(values), Float64))
    if n == 1
        return T[T(values[1])]
    end

    N = n - 1
    coeffs = cheb_dct1(T.(values)) / N
    coeffs[1] /= 2
    coeffs[end] /= 2
    return coeffs
end

"""
Convert interval Lobatto-grid values to first-kind Chebyshev coefficients.

This uses the DCT-I interpolation formula with endpoint half weights, carried
out entirely in interval arithmetic.
"""
function cheb_lobatto_values_to_coeffs_1d(values::AbstractVector{<:Interval})
    n = length(values)
    n >= 1 || error("Expected a nonempty values vector")
    T = eltype(values)

    if n == 1
        return [values[1]]
    end

    return cheb_interval_recovery_matrix(n - 1, T) * collect(values)
end

"""
Apply a one-dimensional transform to every column of a matrix.
"""
function cheb_map_columns(transform::F, A::AbstractMatrix{<:Number}) where {F}
    first_column = transform(@view A[:, 1])
    out = Matrix{eltype(first_column)}(undef, length(first_column), size(A, 2))
    out[:, 1] .= first_column

    for j in 2:size(A, 2)
        out[:, j] .= transform(@view A[:, j])
    end

    return out
end

"""
Apply a one-dimensional transform to every row of a matrix.
"""
function cheb_map_rows(transform::F, A::AbstractMatrix{<:Number}) where {F}
    first_row = transform(@view A[1, :])
    out = Matrix{eltype(first_row)}(undef, size(A, 1), length(first_row))
    out[1, :] .= first_row

    for i in 2:size(A, 1)
        out[i, :] .= transform(@view A[i, :])
    end

    return out
end

"""
Convert a coefficient array to its values on the tensor-product Lobatto grid.
"""
function cheb_coeffs_to_lobatto_values_2d(coeffs::AbstractMatrix{<:Real})
    tmp = cheb_map_columns(cheb_coeffs_to_lobatto_values_1d, coeffs)
    return cheb_map_rows(cheb_coeffs_to_lobatto_values_1d, tmp)
end

"""
Convert an interval coefficient array to Lobatto-grid values using cached
interval cosine matrices.
"""
function cheb_coeffs_to_lobatto_values_2d(coeffs::AbstractMatrix{<:Interval})
    T = eltype(coeffs)
    Cx = cheb_interval_eval_matrix(size(coeffs, 1) - 1, T)
    Cy = cheb_interval_eval_matrix(size(coeffs, 2) - 1, T)
    return Cx * Matrix(coeffs) * Cy
end

"""
Convert tensor-product Lobatto-grid values to Chebyshev coefficients.
"""
function cheb_lobatto_values_to_coeffs_2d(values::AbstractMatrix{<:Real})
    tmp = cheb_map_columns(cheb_lobatto_values_to_coeffs_1d, values)
    return cheb_map_rows(cheb_lobatto_values_to_coeffs_1d, tmp)
end

"""
Convert interval Lobatto-grid values to coefficients using cached interval
recovery matrices.
"""
function cheb_lobatto_values_to_coeffs_2d(values::AbstractMatrix{<:Interval})
    T = eltype(values)
    Rx = cheb_interval_recovery_matrix(size(values, 1) - 1, T)
    Ry = cheb_interval_recovery_matrix(size(values, 2) - 1, T)
    return Rx * Matrix(values) * Ry
end

"""
Zero coefficients below `atol` and trim the outer zero rows and columns.
"""
function cheb_trim_small(coeffs::AbstractMatrix{<:Real}; atol::Real = 0.0)
    out = Matrix{float(promote_type(eltype(coeffs), typeof(atol)))}(undef, size(coeffs)...)
    for I in eachindex(coeffs)
        value = coeffs[I]
        out[I] = abs(value) <= atol ? zero(eltype(out)) : value
    end
    return cheb_trim_exact(out)
end

"""
Multiply two tensor-product Chebyshev series by the Lobatto-grid/DCT trick.

This pads both factors to the full product bidegree, evaluates them on the
tensor-product Chebyshev-Lobatto grid, multiplies pointwise, and transforms
back to coefficient space. When FFTW is available the transforms are DCT-I
based; otherwise the same transform is evaluated by its defining cosine sums.
"""
function cheb_mul2_dct(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    degx = size(A, 1) + size(B, 1) - 2
    degy = size(A, 2) + size(B, 2) - 2
    T = float(promote_type(eltype(A), eltype(B), Float64))

    A_pad = cheb_pad(T.(A), degx, degy)
    B_pad = cheb_pad(T.(B), degx, degy)

    valuesA = cheb_coeffs_to_lobatto_values_2d(A_pad)
    valuesB = cheb_coeffs_to_lobatto_values_2d(B_pad)
    coeffs = cheb_lobatto_values_to_coeffs_2d(valuesA .* valuesB)

    scale = max(maximum(abs, valuesA), maximum(abs, valuesB), maximum(abs, coeffs), one(T))
    atol = 100 * eps(T) * length(coeffs) * scale
    return cheb_trim_small(coeffs; atol)
end

"""
Multiply two tensor-product interval Chebyshev series by the Lobatto-grid/DCT
trick, using only interval arithmetic in the transforms.
"""
function cheb_mul2_interval_dct(A::AbstractMatrix{<:Interval}, B::AbstractMatrix{<:Interval})
    degx = size(A, 1) + size(B, 1) - 2
    degy = size(A, 2) + size(B, 2) - 2
    T = promote_type(eltype(A), eltype(B))

    A_pad = cheb_pad(Matrix{T}(A), degx, degy)
    B_pad = cheb_pad(Matrix{T}(B), degx, degy)

    valuesA = cheb_coeffs_to_lobatto_values_2d(A_pad)
    valuesB = cheb_coeffs_to_lobatto_values_2d(B_pad)
    coeffs = cheb_lobatto_values_to_coeffs_2d(valuesA .* valuesB)

    return cheb_trim_exact(coeffs)
end

"""
Multiply two tensor-product Chebyshev series in coefficient space by the
direct product identity.
"""
function cheb_mul2_direct(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number})
    if length(A) <= length(B)
        small = A
        large = B
    else
        small = B
        large = A
    end

    T = promote_type(eltype(A), eltype(B))
    out = zeros(T, size(A, 1) + size(B, 1) - 1, size(A, 2) + size(B, 2) - 1)

    for j in axes(small, 2), i in axes(small, 1)
        coeff = small[i, j]
        coeff_is_exact_zero(coeff) && continue
        cheb_add_scaled_basis_product_2d!(out, large, i - 1, j - 1, coeff)
    end

    return cheb_trim_exact(out)
end

"""
Multiply two tensor-product Chebyshev series in coefficient space.

Keyword `method` may be `:direct` for the exact coefficient-space product
identity or `:dct` for the Lobatto-grid/DCT trick.
"""
function cheb_mul2(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number}; method::Symbol = :direct)
    if method === :direct
        return cheb_mul2_direct(A, B)
    elseif method === :dct
        eltype(A) <: Interval && error("method=:dct is floating-point only; use method=:interval_dct for interval coefficients")
        eltype(B) <: Interval && error("method=:dct is floating-point only; use method=:interval_dct for interval coefficients")
        eltype(A) <: AbstractFloat || error("method=:dct requires floating-point coefficients")
        eltype(B) <: AbstractFloat || error("method=:dct requires floating-point coefficients")
        return cheb_mul2_dct(A, B)
    elseif method === :interval_dct
        eltype(A) <: Interval || error("method=:interval_dct requires interval coefficients")
        eltype(B) <: Interval || error("method=:interval_dct requires interval coefficients")
        return cheb_mul2_interval_dct(A, B)
    else
        error("Unknown Chebyshev multiplication method: $method")
    end
end

"""
Bound the sup norm of a Chebyshev series by summing absolute coefficients.
"""
chebyshev_coeff_sup_bound(coeffs::AbstractMatrix{<:Number}) = sum(abs, coeffs)



"""
Define the Chebyshev differentitation matrices.
"""
const _CHEB_DIFF_CACHE = Dict{Tuple{Int, DataType}, Any}()

function cheb_diff_matrix(N::Integer, ::Type{T}=Float64) where {T<:Number}
    N >= 1 || error("Need N >= 1")

    key = (N, T)
    return get!(_CHEB_DIFF_CACHE, key) do
        z = [cospi(cheb_interval_constant(T, k // N)) for k in 0:N]
        c = ones(T, N+1)
        c[1] = exact(2)
        c[end] = exact(2)
        c .= c .* [isodd(k) ? -one(T) : one(T) for k in 0:N]

        D = zeros(T, N+1, N+1)

        for i in 1:N+1, j in 1:N+1
            if i != j
                D[i,j] = (c[i] / c[j]) / (z[i] - z[j])
            end
        end

        for i in 1:N+1
            D[i,i] = -sum(D[i,j] for j in 1:N+1 if j != i)
        end

        D
    end
end

"""
Assemble the Lobatto-grid values of a Chebyshev series and its first and second 
derivatives by applying the Chebyshev differentiation matrices.
"""
function build_lobatto_derivative_pack(u_vals::AbstractMatrix{<:Number}; pdeg::Integer=size(coeffs,1))
    Nx = pdeg - 1
    Ny = pdeg - 1
    T = eltype(u_vals)

    Dxξ = cheb_diff_matrix(Nx, T)
    Dyη = cheb_diff_matrix(Ny, T)

    Dx = exact(2) .* Dxξ
    Dy = -exact(2) .* Dyη

    ux  = Dx * u_vals
    uy  = u_vals * transpose(Dy)

    uxx = Dx * ux
    uyy = uy * transpose(Dy)
    uxy = ux * transpose(Dy)

    uxxx = Dx * uxx
    uxxy = uxx * transpose(Dy)
    uxyy = uxy * transpose(Dy)
    uyyy = uyy * transpose(Dy)
    
    zξ = [cospi(interval(BigFloat(k)) / exact(BigFloat(Nx))) for k in 0:Nx]
    zη = [cospi(interval(BigFloat(k)) / exact(BigFloat(Ny))) for k in 0:Ny]

    xarr = (zξ .+ exact(1)) ./ exact(2)
    yarr = (exact(1) .- zη) ./ exact(2)

    return (; ux, uy, uxx, uyy, uxy, uxxx, uxxy, uxyy, uyyy, xarr, yarr, pdeg)
end

function build_lobatto_first_derivative_pack(u_vals::AbstractMatrix{<:Number}; pdeg::Integer=size(coeffs,1))
    Nx = pdeg - 1
    Ny = pdeg - 1
    T = eltype(u_vals)

    Dxξ = cheb_diff_matrix(Nx, T)
    Dyη = cheb_diff_matrix(Ny, T)

    Dx = exact(2) .* Dxξ
    Dy = -exact(2) .* Dyη

    ux  = Dx * u_vals
    uy  = u_vals * transpose(Dy)
    
    zξ = [cospi(interval(BigFloat(k)) / exact(BigFloat(Nx))) for k in 0:Nx]
    zη = [cospi(interval(BigFloat(k)) / exact(BigFloat(Ny))) for k in 0:Ny]

    xarr = (zξ .+ exact(1)) ./ exact(2)
    yarr = (exact(1) .- zη) ./ exact(2)

    return (; ux, uy, xarr, yarr, pdeg)
end