const METRIC_U0_PATH = joinpath(REPO_ROOT, "data", "happrox", "coeffs.csv")

function parse_metric_decimal_interval(entry::AbstractString)::Interval{BigFloat}
    stripped = strip(entry)
    if startswith(stripped, '"') && endswith(stripped, '"') && length(stripped) >= 2
        stripped = stripped[2:end - 1]
    end
    occursin('/', stripped) && error("Metric coefficient CSV entries must be decimal strings, got rational entry: $stripped")
    return parse_bound_value(stripped)
end

function load_metric_coeffs_csv(path::AbstractString)::Matrix{Interval{BigFloat}}
    rows = Vector{Vector{Interval{BigFloat}}}()

    open(path, "r") do io
        for line in eachline(io)
            push!(rows, parse_metric_decimal_interval.(split(line, ',')))
        end
    end

    coeffs = Matrix{Interval{BigFloat}}(undef, length(rows), length(rows[1]))
    for i in eachindex(rows)
        coeffs[i, :] .= rows[i]
    end
    return coeffs
end
