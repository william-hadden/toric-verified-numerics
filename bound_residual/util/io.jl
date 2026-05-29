import JSON
import Printf

const DEFAULT_REF_INDEX = (2, 2)
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const U0_PATH = joinpath(REPO_ROOT, "data", "happrox", "coeffs-rational.csv")
const VERIFIED_BOUNDS_PATH = joinpath(REPO_ROOT, "data", "verified_bounds.json")

"""
Parse one rational CSV entry as an `Interval{BigFloat}` at the active precision.
"""
function parse_rational_interval(entry::AbstractString)::Interval{BigFloat}
    numerator_entry, denominator_entry = split(entry, '/')
    numerator = parse(BigInt, numerator_entry)
    denominator = parse(BigInt, denominator_entry)

    return interval(numerator) / interval(denominator)
end

"""
Load the rational coefficient matrix from the canonical CSV export.
"""
function load_rational_coeffs_csv(path::AbstractString)::Matrix{Interval{BigFloat}}
    rows = Vector{Vector{Interval{BigFloat}}}()

    open(path, "r") do io
        for line in eachline(io)
            push!(rows, parse_rational_interval.(split(line, ',')))
        end
    end

    coeffs = Matrix{Interval{BigFloat}}(undef, length(rows), length(rows[1]))
    for i in eachindex(rows)
        coeffs[i, :] .= rows[i]
    end
    return coeffs
end

"""
Read one top-level bound dictionary from the verified bounds JSON file.
"""
function read_bound(key::AbstractString; path::AbstractString = VERIFIED_BOUNDS_PATH)
    return parse_bound_value(JSON.parsefile(path)[key])
end

read_bound(key::Symbol; path::AbstractString = VERIFIED_BOUNDS_PATH) = read_bound(String(key); path)

parse_bound_value(bounds::AbstractDict) = Dict(key => parse_bound_value(value) for (key, value) in bounds)

parse_bound_value(bound::Real) = Interval{BigFloat}(interval(bound))

function parse_bound_value(bound::AbstractString)::Interval{BigFloat}
    lower = setrounding(BigFloat, RoundDown) do
        parse(BigFloat, bound)
    end
    upper = setrounding(BigFloat, RoundUp) do
        parse(BigFloat, bound)
    end
    return interval(lower, upper)
end

const SERIALIZED_BOUND_DECIMAL_DIGITS = 77

function serialize_bound_value(bound::BigFloat)
    return Printf.@sprintf("%.*f", SERIALIZED_BOUND_DECIMAL_DIGITS, bound)
end

serialize_bound_value(bound::Interval) = serialize_bound_value(BigFloat(sup(bound)))
serialize_bound_value(bound::Real) = serialize_bound_value(BigFloat(bound))
serialize_bound_value(bounds::NamedTuple) = Dict(String(key) => serialize_bound_value(value) for (key, value) in pairs(bounds))
serialize_bound_value(bounds::AbstractDict) = Dict(String(key) => serialize_bound_value(value) for (key, value) in bounds)

"""
Write one entry into a top-level bound dictionary.
"""
function write_bound_entry(group_key::AbstractString, value_key::AbstractString, value; path::AbstractString = VERIFIED_BOUNDS_PATH)
    bounds = isfile(path) ? JSON.parsefile(path) : Dict{String,Any}()
    group = get!(bounds, group_key) do
        Dict{String,Any}()
    end
    group isa AbstractDict || error("Bound '$group_key' is not a dictionary")
    group[value_key] = serialize_bound_value(value)

    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, bounds, 4)
        println(io)
    end

    return nothing
end

write_bound_entry(group_key::Symbol, value_key::Symbol, value; path::AbstractString = VERIFIED_BOUNDS_PATH) =
    write_bound_entry(String(group_key), String(value_key), value; path)

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
