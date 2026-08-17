"""
Differentiate a one-variable first-kind Chebyshev series in coefficient space.

The result retains one trailing zero mode, so it has the same length as the
input coefficient vector.
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
Differentiate exported coefficients in the `x` variable on `[0,1]`.
"""
function differentiate_coeffs_x(coeffs::AbstractMatrix{<:Number})
    out = zeros(eltype(coeffs), size(coeffs))
    for j in axes(coeffs, 2)
        out[:, j] .= exact(2) .* cheb_diff_1d(@view coeffs[:, j])
    end
    return out
end

"""
Differentiate exported coefficients in the `y` variable on `[0,1]`.
"""
function differentiate_coeffs_y(coeffs::AbstractMatrix{<:Number})
    out = zeros(eltype(coeffs), size(coeffs))
    for i in axes(coeffs, 1)
        out[i, :] .= -exact(2) .* cheb_diff_1d(vec(@view coeffs[i, :]))
    end
    return out
end

"""
Differentiate exported coefficients along dimension 1 (`x`) or 2 (`y`).
"""
partial_coeffs(coeffs::AbstractMatrix{<:Number}, dim::Integer) =
    dim == 1 ? differentiate_coeffs_x(coeffs) :
    dim == 2 ? differentiate_coeffs_y(coeffs) :
    throw(ArgumentError("Expected derivative dimension 1 or 2"))

"""
Assemble a series and its first and second derivative coefficient arrays.
"""
function build_derivative_pack(coeffs::AbstractMatrix{<:Number})
    u0 = copy(coeffs)
    ux = differentiate_coeffs_x(u0)
    uy = differentiate_coeffs_y(u0)
    uxx = differentiate_coeffs_x(ux)
    uyy = differentiate_coeffs_y(uy)
    uxy = (differentiate_coeffs_y(ux) .+ differentiate_coeffs_x(uy)) ./ exact(2)
    return (; u0, ux, uy, uxx, uyy, uxy)
end

"""
Return only the second-derivative coefficient arrays of a Chebyshev series.
"""
function compute_second_derivatives(coefficients::AbstractMatrix{<:Number})
    pack = build_derivative_pack(intervalize_coefficients(coefficients))
    return (; uxx = pack.uxx, uxy = pack.uxy, uyy = pack.uyy)
end
