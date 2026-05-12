const DEFAULT_REF_INDEX = (2, 2)
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const U0_PATH = joinpath(REPO_ROOT, "data", "happrox", "coeffs.csv")

"""
Parse one numeric CSV entry as `BigFloat`.
"""
function parse_csv_number(entry::AbstractString)::BigFloat
    cleaned = strip(replace(entry, '"' => ""))
    return parse(BigFloat, cleaned)
end

"""
Load the coefficient matrix from the canonical CSV export.
"""
function load_coeffs_csv(path::AbstractString)::Matrix{BigFloat}
    lines = readlines(path)
    isempty(lines) && error("Coefficient CSV is empty: $path")

    rows = Vector{Vector{BigFloat}}(undef, length(lines))
    expected_cols = nothing

    for (i, line) in pairs(lines)
        entries = split(chomp(line), ',')
        expected_cols === nothing && (expected_cols = length(entries))
        length(entries) == expected_cols || error("Row $i has $(length(entries)) columns; expected $expected_cols")
        rows[i] = parse_csv_number.(entries)
    end

    coeffs = Matrix{BigFloat}(undef, length(rows), expected_cols)
    for i in eachindex(rows)
        coeffs[i, :] .= rows[i]
    end
    return coeffs
end

"""
Return the one-dimensional Chebyshev-Lobatto points `cos(pi*k/N)` on `[-1,1]`.
"""
cheb_grid(N::Integer) = cospi.(interval.(BigFloat.(collect(0:N))) ./ exact(BigFloat(N)))

"""
Return the tensor-product grids on `[0,1]^2` used by the exported coefficients.
"""
function make_grids(N::Integer)
    half = exact(BigFloat(0.5))
    xarr = reverse(half .+ half .* cheb_grid(N))
    yarr = copy(xarr)
    return xarr, yarr
end

"""
Return the values `T_0(z), ..., T_degree(z)` by the three-term recurrence.
"""
function chebyshev_values(z, degree::Integer)
    degree >= 0 || error("Chebyshev degree must be nonnegative")
    T = typeof(z)
    values = Vector{T}(undef, degree + 1)
    values[1] = one(z)

    if degree >= 1
        values[2] = z
        for n in 2:degree
            values[n + 1] = exact(2) * z * values[n] - values[n - 1]
        end
    end

    return values
end

"""
Evaluate a tensor-product Chebyshev series at a single point `(x, y)` in `[0,1]^2`.
"""
function evaluate_coeffs_at_point(coeffs::AbstractMatrix{<:Number}, x::Real, y::Real)
    Tx = chebyshev_values(exact(2) * x - exact(1), size(coeffs, 1) - 1)
    Ty = chebyshev_values(exact(1) - exact(2) * y, size(coeffs, 2) - 1)

    # Initialise values as zero with type being the more general type of the types among
    # coeffs, Tx. Usually expect coeffs to be Interval{BigFloat} and Tx to be BigFloat.
    value = zero(promote_type(eltype(coeffs), eltype(Tx))) 

    for j in axes(coeffs, 2)
        inner = zero(promote_type(eltype(coeffs), eltype(Tx)))
        for i in axes(coeffs, 1)
            inner += coeffs[i, j] * Tx[i]
        end
        value += inner * Ty[j]
    end

    return value
end
