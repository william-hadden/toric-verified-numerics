import JSON
import Printf

const DEFAULT_REF_INDEX = (2, 2)
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const U0_PATH = joinpath(REPO_ROOT, "data", "happrox", "coeffs-rational.csv")
const METRIC_U0_PATH = joinpath(REPO_ROOT, "data", "happrox", "coeffs.csv")
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
Load a rectangular coefficient matrix using `parse_entry` for each CSV field.
"""
function load_coeff_matrix_csv(path::AbstractString, parse_entry)::Matrix{Interval{BigFloat}}
    rows = Vector{Vector{Interval{BigFloat}}}()
    open(path, "r") do io
        for line in eachline(io)
            push!(rows, parse_entry.(split(line, ',')))
        end
    end
    isempty(rows) && error("Coefficient CSV is empty: $path")
    width = length(first(rows))
    all(length(row) == width for row in rows) || error("Coefficient CSV is not rectangular: $path")

    coeffs = Matrix{Interval{BigFloat}}(undef, length(rows), width)
    for i in eachindex(rows)
        coeffs[i, :] .= rows[i]
    end
    return coeffs
end

"""Load the rational coefficient matrix from the canonical CSV export."""
load_rational_coeffs_csv(path::AbstractString) = load_coeff_matrix_csv(path, parse_rational_interval)

"""Parse a decimal metric coefficient as an outward-rounded interval."""
function parse_metric_decimal_interval(entry::AbstractString)::Interval{BigFloat}
    stripped = strip(entry)
    if startswith(stripped, '"') && endswith(stripped, '"') && length(stripped) >= 2
        stripped = stripped[2:end - 1]
    end
    occursin('/', stripped) && error(
        "Metric coefficient CSV entries must be decimal strings, got rational entry: $stripped",
    )
    return parse_bound_value(stripped)
end

"""Load a rectangular CSV matrix of decimal metric coefficients."""
load_metric_coeffs_csv(path::AbstractString) = load_coeff_matrix_csv(path, parse_metric_decimal_interval)

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
