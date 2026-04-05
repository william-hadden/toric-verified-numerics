const DEFAULT_TAYLOR_DEGREE = 12
const DEFAULT_REF_INDEX = (2, 2)
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const U0_PATH = joinpath(REPO_ROOT, "data", "happrox", "coeffs.csv")

"""
Parse one numeric CSV entry as `Float64`.
"""
function parse_csv_number(entry::AbstractString)::Float64
    cleaned = strip(replace(entry, '"' => ""))
    value = tryparse(Float64, cleaned)
    value === nothing && error("Could not parse Float64 from CSV entry: $entry")
    return value
end

"""
Load the coefficient matrix from the canonical CSV export.
"""
function load_coeffs_csv(path::AbstractString)::Matrix{Float64}
    lines = readlines(path)
    isempty(lines) && error("Coefficient CSV is empty: $path")

    rows = Vector{Vector{Float64}}(undef, length(lines))
    expected_cols = nothing

    for (i, line) in pairs(lines)
        entries = split(chomp(line), ',')
        expected_cols === nothing && (expected_cols = length(entries))
        length(entries) == expected_cols || error("Row $i has $(length(entries)) columns; expected $expected_cols")
        rows[i] = parse_csv_number.(entries)
    end

    coeffs = Matrix{Float64}(undef, length(rows), expected_cols)
    for i in eachindex(rows)
        coeffs[i, :] .= rows[i]
    end
    return coeffs
end

"""
Return the one-dimensional Chebyshev-Lobatto points `cos(pi*k/N)` on `[-1,1]`.
"""
cheb_grid(N::Integer) = cos.(pi .* collect(0:N) ./ N)

"""
Return the tensor-product grids on `[0,1]^2` used by the exported coefficients.
"""
function make_grids(N::Integer)
    xarr = reverse(0.5 .+ 0.5 .* cheb_grid(N))
    yarr = copy(xarr)
    return xarr, yarr
end

"""
Return the values `T_0(z), ..., T_degree(z)` by the three-term recurrence.
"""
function chebyshev_values(z, degree::Integer)
    degree >= 0 || error("Chebyshev degree must be nonnegative")
    values = Vector{Float64}(undef, degree + 1)
    values[1] = 1.0

    if degree >= 1
        values[2] = Float64(z)
        for n in 2:degree
            values[n + 1] = 2.0 * z * values[n] - values[n - 1]
        end
    end

    return values
end

"""
Evaluate a tensor-product Chebyshev series at a single point `(x, y)` in `[0,1]^2`.
"""
function evaluate_coeffs_at_point(coeffs::AbstractMatrix{<:Real}, x::Real, y::Real)::Float64
    Tx = chebyshev_values(2 * x - 1, size(coeffs, 1) - 1)
    Ty = chebyshev_values(1 - 2 * y, size(coeffs, 2) - 1)

    value = 0.0
    for j in axes(coeffs, 2)
        inner = 0.0
        for i in axes(coeffs, 1)
            inner += coeffs[i, j] * Tx[i]
        end
        value += inner * Ty[j]
    end

    return value
end
