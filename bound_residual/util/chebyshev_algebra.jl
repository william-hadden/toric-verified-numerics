"""
Return a dense zero coefficient array with the requested bidegree.
"""
function cheb_zero(degx::Integer, degy::Integer)::Matrix{Float64}
    degx >= 0 || error("Expected a nonnegative x-degree")
    degy >= 0 || error("Expected a nonnegative y-degree")
    return zeros(Float64, degx + 1, degy + 1)
end

"""
Trim exact zero rows and columns from the outer boundary of a coefficient array.
"""
function cheb_trim_exact(coeffs::AbstractMatrix{<:Number})
    last_row = size(coeffs, 1)
    while last_row > 1 && all(iszero, @view coeffs[last_row, :])
        last_row -= 1
    end

    last_col = size(coeffs, 2)
    while last_col > 1 && all(iszero, @view coeffs[:, last_col])
        last_col -= 1
    end

    return copy(@view coeffs[1:last_row, 1:last_col])
end

"""
Return the `1 x 1` coefficient array of a constant series.
"""
function cheb_constant(c)::Matrix{Float64}
    coeffs = zeros(Float64, 1, 1)
    coeffs[1, 1] = c
    return coeffs
end

"""
Pad a coefficient array with zeros up to the requested bidegree.
"""
function cheb_pad(coeffs::AbstractMatrix{<:Real}, degx::Integer, degy::Integer)::Matrix{Float64}
    degx >= size(coeffs, 1) - 1 || error("Requested x-degree is too small")
    degy >= size(coeffs, 2) - 1 || error("Requested y-degree is too small")

    out = cheb_zero(degx, degy)
    out[1:size(coeffs, 1), 1:size(coeffs, 2)] .= coeffs
    return out
end

"""
Add two dense coefficient arrays after zero-padding to the common bidegree.
"""
function cheb_add(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})::Matrix{Float64}
    degx = max(size(A, 1), size(B, 1)) - 1
    degy = max(size(A, 2), size(B, 2)) - 1
    out = cheb_zero(degx, degy)

    for j in axes(A, 2), i in axes(A, 1)
        out[i, j] += A[i, j]
    end
    for j in axes(B, 2), i in axes(B, 1)
        out[i, j] += B[i, j]
    end

    return cheb_trim_exact(out)
end

"""
Subtract two dense coefficient arrays after zero-padding to the common bidegree.
"""
function cheb_sub(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})::Matrix{Float64}
    degx = max(size(A, 1), size(B, 1)) - 1
    degy = max(size(A, 2), size(B, 2)) - 1
    out = cheb_zero(degx, degy)

    for j in axes(A, 2), i in axes(A, 1)
        out[i, j] += A[i, j]
    end
    for j in axes(B, 2), i in axes(B, 1)
        out[i, j] -= B[i, j]
    end

    return cheb_trim_exact(out)
end

"""
Scale a dense coefficient array by a scalar.
"""
function cheb_scale(A::AbstractMatrix{<:Real}, c)::Matrix{Float64}
    out = Matrix{Float64}(undef, size(A)...)
    for I in eachindex(A)
        out[I] = A[I] * c
    end
    return cheb_trim_exact(out)
end

"""
Add a scalar to the constant mode of a coefficient array.
"""
function cheb_add_constant(A::AbstractMatrix{<:Real}, c)::Matrix{Float64}
    out = Matrix{Float64}(undef, size(A)...)
    out .= A
    out[1, 1] += c
    return cheb_trim_exact(out)
end

"""
Add `scale * T_mode * coeffs` to a one-dimensional output array.
"""
function cheb_add_scaled_basis_product_1d!(out::AbstractVector{Float64}, coeffs::AbstractVector{<:Real}, mode::Integer, scale)
    half_scale = scale / 2

    @inbounds for j in eachindex(coeffs)
        contribution = half_scale * coeffs[j]
        out[j + mode] += contribution
        out[abs((j - 1) - mode) + 1] += contribution
    end

    return out
end

"""
Multiply two one-dimensional Chebyshev series in coefficient space.
"""
function cheb_mul1(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})::Vector{Float64}
    if length(a) <= length(b)
        small = a
        large = b
    else
        small = b
        large = a
    end

    out = zeros(Float64, length(a) + length(b) - 1)

    @inbounds for i in eachindex(small)
        coeff = small[i]
        iszero(coeff) && continue
        cheb_add_scaled_basis_product_1d!(out, large, i - 1, coeff)
    end

    return cheb_trim_exact(reshape(out, :, 1))[:, 1]
end

"""
Add `scale * T_modex * T_modey * coeffs` to a two-dimensional output array.
"""
function cheb_add_scaled_basis_product_2d!(out::AbstractMatrix{Float64}, coeffs::AbstractMatrix{<:Real}, modex::Integer, modey::Integer, scale)
    quarter_scale = scale / 4

    @inbounds for j in axes(coeffs, 2), i in axes(coeffs, 1)
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
Multiply two tensor-product Chebyshev series in coefficient space.
"""
function cheb_mul2(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})::Matrix{Float64}
    if length(A) <= length(B)
        small = A
        large = B
    else
        small = B
        large = A
    end

    out = zeros(Float64, size(A, 1) + size(B, 1) - 1, size(A, 2) + size(B, 2) - 1)

    @inbounds for j in axes(small, 2), i in axes(small, 1)
        coeff = small[i, j]
        iszero(coeff) && continue
        cheb_add_scaled_basis_product_2d!(out, large, i - 1, j - 1, coeff)
    end

    return cheb_trim_exact(out)
end

"""
Multiply a tensor-product Chebyshev series by the coordinate `x` on `[0,1]^2`.
"""
function cheb_mul_x(A::AbstractMatrix{<:Real})::Matrix{Float64}
    out = zeros(Float64, size(A, 1) + 1, size(A, 2))

    @inbounds for j in axes(A, 2), i in axes(A, 1)
        coeff = A[i, j]
        half_contribution = coeff / 2
        quarter_contribution = coeff / 4

        out[i, j] += half_contribution
        out[i + 1, j] += quarter_contribution
        out[abs((i - 1) - 1) + 1, j] += quarter_contribution
    end

    return cheb_trim_exact(out)
end

"""
Multiply a tensor-product Chebyshev series by the coordinate `y` on `[0,1]^2`.
"""
function cheb_mul_y(A::AbstractMatrix{<:Real})::Matrix{Float64}
    out = zeros(Float64, size(A, 1), size(A, 2) + 1)

    @inbounds for j in axes(A, 2), i in axes(A, 1)
        coeff = A[i, j]
        half_contribution = coeff / 2
        quarter_contribution = coeff / 4

        out[i, j] += half_contribution
        out[i, j + 1] -= quarter_contribution
        out[i, abs((j - 1) - 1) + 1] -= quarter_contribution
    end

    return cheb_trim_exact(out)
end

"""
Return the scalar coefficients of the degree-`p` Taylor polynomial of `exp`.
"""
function exp_taylor_coeffs(p::Integer)::Vector{Float64}
    p >= 0 || error("Taylor degree must be nonnegative")
    coeffs = Vector{Float64}(undef, p + 1)
    coeffs[1] = 1.0

    for k in 1:p
        coeffs[k + 1] = coeffs[k] / k
    end

    return coeffs
end

"""
Evaluate an ordinary one-variable polynomial `q(z)` at a tensor-product
Chebyshev series `H(x,y)` by Horner's rule.

If `coeffs = [a_0, ..., a_p]`, define the scalar polynomial
`q(z) = sum_{k=0}^p a_k z^k`.
This function returns the coefficient array of the two-variable series
`q(H(x,y)) = sum_{k=0}^p a_k H(x,y)^k`.

The computation uses the nested identity
`q(H) = a_0 + H(a_1 + H(a_2 + ... + H(a_{p-1} + H a_p)...))`,
so the polynomial is one-dimensional in the formal variable `z`, while `H`
itself is a tensor-product Chebyshev series in `(x,y)`.

The code starts from the top coefficient `a_p` and iterates
`acc <- a_k + H * acc`.
Here `H * acc` is multiplication in the coefficient algebra of
two-variable Chebyshev series, implemented by `cheb_mul2`, and adding `a_k`
means adding `a_k T_0(x) T_0(y)` to the constant mode.

This is the usual Horner identity for polynomial evaluation, applied in the
ring of tensor-product Chebyshev series.
"""
function cheb_horner_scalar_poly(H::AbstractMatrix{<:Real}, coeffs::AbstractVector{<:Real})::Matrix{Float64}
    isempty(coeffs) && error("Expected at least one scalar coefficient")
    acc = cheb_constant(coeffs[end])

    for k in (length(coeffs) - 1):-1:1
        # Horner step: replace the current partial polynomial r(H) by
        # a_{k-1} + H * r(H).
        acc = cheb_add_constant(cheb_mul2(H, acc), coeffs[k])
    end

    return acc
end

"""
Bound the sup norm of a Chebyshev series by summing absolute coefficients.
"""
chebyshev_coeff_sup_bound(coeffs::AbstractMatrix{<:Real}) = sum(abs, coeffs)
