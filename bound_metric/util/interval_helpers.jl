intervalize_coefficients(coefficients::AbstractMatrix{<:Interval}) = Matrix(coefficients)
intervalize_coefficients(coefficients::AbstractMatrix{<:Number}) = interval.(coefficients)

interval_constant(x) = interval(BigFloat(x))
interval_half() = interval_constant(1) / exact(2)
interval_quarter() = interval_constant(1) / exact(4)

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
