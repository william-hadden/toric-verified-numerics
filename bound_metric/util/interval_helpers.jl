intervalize_coefficients(coefficients::AbstractMatrix{<:Interval}) = Matrix(coefficients)
intervalize_coefficients(coefficients::AbstractMatrix{<:Number}) = interval.(coefficients)

interval_constant(x) = interval(BigFloat(x))
interval_half() = interval_constant(1) / exact(2)
interval_quarter() = interval_constant(1) / exact(4)

function cheb_constant(c)
    T = promote_type(typeof(c), Float64)
    coeffs = zeros(T, 1, 1)
    coeffs[1, 1] = c
    return coeffs
end

function cheb_sub(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number})
    degx = max(size(A, 1), size(B, 1)) - 1
    degy = max(size(A, 2), size(B, 2)) - 1
    T = promote_type(eltype(A), eltype(B))
    out = cheb_zero(degx, degy, T)

    for j in axes(A, 2), i in axes(A, 1)
        out[i, j] += A[i, j]
    end
    for j in axes(B, 2), i in axes(B, 1)
        out[i, j] -= B[i, j]
    end

    return cheb_trim_exact(out)
end

function cheb_scale(A::AbstractMatrix{<:Number}, c)
    T = promote_type(eltype(A), typeof(c))
    out = Matrix{T}(undef, size(A)...)
    for I in eachindex(A)
        out[I] = A[I] * c
    end
    return cheb_trim_exact(out)
end

function cheb_mul_fast(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number})
    return cheb_mul2(intervalize_coefficients(A), intervalize_coefficients(B); method = :interval_dct)
end

function cheb_product_fast(series; progress = nothing)
    isempty(series) && return cheb_constant(interval_constant(1))
    out = first(series)
    for k in 2:length(series)
        out = cheb_mul_fast(out, series[k])
        advance_progress!(progress)
    end
    return out
end
