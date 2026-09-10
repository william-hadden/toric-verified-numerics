import JSON

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

"""
Load the rational coefficient matrix from the canonical CSV export.
"""
load_rational_coeffs_csv(path::AbstractString) = load_coeff_matrix_csv(path, parse_rational_interval)

"""
Parse a decimal metric coefficient as an outward-rounded interval.
"""
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

"""
Load a rectangular CSV matrix of decimal metric coefficients.
"""
load_metric_coeffs_csv(path::AbstractString) = load_coeff_matrix_csv(path, parse_metric_decimal_interval)

"""
Read JSON into memory and close the file before returning.

Unlike `JSON.parsefile` in JSON 0.21, this does not leave a memory mapping
that can prevent Windows from reopening the file for writing.
"""
read_bounds_json(path::AbstractString = VERIFIED_BOUNDS_PATH) = JSON.parse(read(path, String))

"""
Read one top-level bound dictionary from the verified bounds JSON file.
"""
function read_bound(key::AbstractString; path::AbstractString = VERIFIED_BOUNDS_PATH)
    return parse_bound_value(read_bounds_json(path)[key])
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

"""Serialize a bound with directed rounding to 77 decimal places."""
function serialize_bound_value(bound::BigFloat; rounding::RoundingMode = RoundUp)
    isfinite(bound) || throw(ArgumentError("Cannot serialize a nonfinite bound: $bound"))
    # Quantize the exact binary value using integers: floating-point formatting
    # rounds to nearest and can weaken a certified upper or lower bound.
    scale = big(10)^SERIALIZED_BOUND_DECIMAL_DIGITS
    units = round(BigInt, Rational{BigInt}(bound) * scale, rounding)
    whole, fraction = divrem(abs(units), scale)
    return string(units < 0 ? "-" : "", whole, ".",
        lpad(string(fraction), SERIALIZED_BOUND_DECIMAL_DIGITS, '0'))
end

serialize_bound_value(bound::Interval; rounding::RoundingMode = RoundUp) =
    serialize_bound_value(rounding == RoundDown ? inf(bound) : sup(bound); rounding)
serialize_bound_value(bound::Real; rounding::RoundingMode = RoundUp) =
    serialize_bound_value(BigFloat(bound, rounding); rounding)
serialize_bound_value(bounds::NamedTuple; rounding::RoundingMode = RoundUp) =
    Dict(String(key) => serialize_bound_value(value; rounding) for (key, value) in pairs(bounds))
serialize_bound_value(bounds::AbstractDict; rounding::RoundingMode = RoundUp) =
    Dict(String(key) => serialize_bound_value(value; rounding) for (key, value) in bounds)

"""
Write one entry into a top-level bound dictionary. Upper bounds round up;
pass `rounding = RoundDown` for a lower bound.
"""
function write_bound_entry(group_key::AbstractString, value_key::AbstractString, value;
    path::AbstractString = VERIFIED_BOUNDS_PATH, rounding::RoundingMode = RoundUp)
    bounds = isfile(path) ? read_bounds_json(path) : Dict{String,Any}()
    group = get!(bounds, group_key) do
        Dict{String,Any}()
    end
    group isa AbstractDict || error("Bound '$group_key' is not a dictionary")
    group[value_key] = serialize_bound_value(value; rounding)

    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, bounds, 4)
        println(io)
    end

    return nothing
end

write_bound_entry(group_key::Symbol, value_key::Symbol, value;
    path::AbstractString = VERIFIED_BOUNDS_PATH, rounding::RoundingMode = RoundUp) =
    write_bound_entry(String(group_key), String(value_key), value; path, rounding)
