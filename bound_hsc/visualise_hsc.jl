"""
Plot the holomorphic sectional-curvature term around the certified region saved
in a successful HSC Step 6 checkpoint.

Run from the repository root with

    julia --project=. bound_hsc/visualise_hsc.jl \
        bound_hsc/checkpoints/run_pdeg-.../step6.jls

The checkpoint directory may be supplied instead of `step6.jls`.  An optional
second argument selects the output HTML file.  The plot is exploratory: each
colour is the midpoint of a rigorous point enclosure, while the overlaid box is
the region used by the rigorous Proposition 5.6 certificate.  Although the
coefficient arrays use the internal coordinate `y`, the plot uses the moment
polytope coordinate `Y = -y`.
"""

using PlotlyJS

include(joinpath(@__DIR__, "verify_proposition_5_6.jl"))

const DEFAULT_PLOT_GRID = 224
const DEFAULT_PADDING_CELLS = 10
const COLOUR_SOFTENING = 1

format_direction_component(value::Rational) = denominator(value) == 1 ?
    string(numerator(value)) : string(value)
format_direction_component(value) = string(value)
format_direction(xi) = "(" * join(format_direction_component.(xi), ",") * ")"
plotly_rows(matrix::AbstractMatrix) = [collect(row) for row in eachrow(matrix)]

function q_tick_label(value::Real)
    rounded = round(Float64(value); sigdigits = 2)
    text = isinteger(rounded) ? string(Int(rounded)) : string(rounded)
    return replace(text, "-" => "−")
end

function original_q_colourbar(sampled_min::Real, sampled_max::Real)
    # Keep the bar readable. Fine values remain available in the hover text.
    candidates = [-3.0, -1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0, 3.0]
    lower_position = asinh(sampled_min / COLOUR_SOFTENING)
    upper_position = asinh(sampled_max / COLOUR_SOFTENING)
    endpoint_gap = 0.06 * (upper_position - lower_position)
    interior_ticks = filter(candidates) do value
        sampled_min < value < sampled_max || return false
        position = asinh(value / COLOUR_SOFTENING)
        return position - lower_position >= endpoint_gap &&
               upper_position - position >= endpoint_gap
    end
    # Always label the true displayed endpoints. Nearby interior ticks are
    # removed above so these labels do not overlap.
    q_ticks = [Float64(sampled_min); interior_ticks; Float64(sampled_max)]
    return PlotlyJS.attr(
        title = "Q₁(V)",
        tickmode = "array",
        tickvals = asinh.(q_ticks ./ COLOUR_SOFTENING),
        ticktext = q_tick_label.(q_ticks),
        tickfont = PlotlyJS.attr(size = 12),
        thickness = 26,
    )
end

function checkpoint_derivatives(step3, step4, pdeg::Integer)
    inverse_value_enclosures =
        prepare_inverse_coeffs_for_subdivision(step3.result, pdeg)
    return (
        deriv_num = step4.result.deriv_num,
        D = enclosure_to_coeffs(inverse_value_enclosures.D),
        pack = step4.result.pack,
        pdeg = step4.result.pdeg,
    )
end

function enclosing_rectangle(boxes::AbstractVector{PolytopeBox})
    isempty(boxes) && error("The saved candidate region is empty")
    return (
        xlo = minimum(box.xlo for box in boxes),
        xhi = maximum(box.xhi for box in boxes),
        ylo = minimum(box.ylo for box in boxes),
        yhi = maximum(box.yhi for box in boxes),
    )
end

function plot_window(candidate_boxes, domain::PolytopeBox, grid::Integer;
                     padding_cells::Integer = DEFAULT_PADDING_CELLS)
    padding_cells >= 0 || error("padding_cells must be nonnegative")
    candidate = enclosing_rectangle(candidate_boxes)
    dx = (domain.xhi - domain.xlo) / grid
    dy = (domain.yhi - domain.ylo) / grid

    xlo = max(domain.xlo, candidate.xlo - padding_cells * dx)
    xhi = min(domain.xhi, candidate.xhi + padding_cells * dx)
    ylo = max(domain.ylo, candidate.ylo - padding_cells * dy)
    yhi = min(domain.yhi, candidate.yhi + padding_cells * dy)

    # Avoid evaluating exactly on the singular polytope boundary.
    inset_x = dx / 10_000
    inset_y = dy / 10_000
    xlo == domain.xlo && (xlo += inset_x)
    xhi == domain.xhi && (xhi -= inset_x)
    ylo == domain.ylo && (ylo += inset_y)
    yhi == domain.yhi && (yhi -= inset_y)
    return (; xlo, xhi, ylo, yhi)
end

function point_hsc_from_intervals(prepared, X, Y)
    D0 = evaluate_hsc_coeffs_on_box(prepared.D0, X, Y)
    S = evaluate_hsc_coeffs_on_box(prepared.S, X, Y)
    inf(D0) > 0 && inf(S) > 0 || return nothing

    P = evaluate_hsc_coeffs_on_box(prepared.P, X, Y)
    Q = require_guaranteed(P / (D0^3 * S^2), "plotted HSC point")
    midpoint = (inf(Q) + sup(Q)) / exact(2)
    return (; midpoint = Float64(midpoint), lower = Float64(inf(Q)),
            upper = Float64(sup(Q)))
end

point_hsc(prepared, x::Rational, y::Rational) =
    point_hsc_from_intervals(prepared, exact_big_interval(x), exact_big_interval(y))

point_hsc(prepared, x::Real, y::Real) = point_hsc_from_intervals(
    prepared,
    interval(BigFloat, BigFloat(x)),
    interval(BigFloat, BigFloat(y)),
)

function hsc_surface_data(prepared, window; ngrid::Integer = DEFAULT_PLOT_GRID)
    ngrid >= 2 || error("ngrid must be at least 2")
    xs = collect(range(Float64(window.xlo), Float64(window.xhi); length = ngrid))
    ys = collect(range(Float64(window.ylo), Float64(window.yhi); length = ngrid))
    values = fill(NaN, ngrid, ngrid)

    for (iy, y) in enumerate(ys), (ix, x) in enumerate(xs)
        value = point_hsc(prepared, x, y)
        value === nothing || (values[iy, ix] = value.midpoint)
    end
    any(isfinite, values) || error("No valid HSC samples were obtained in the plot window")
    return xs, ys, values
end

function moment_rectangle_trace(box; name, colour, width = 2, showlegend = true)
    xlo, xhi = Float64(box.xlo), Float64(box.xhi)
    Ylo, Yhi = -Float64(box.yhi), -Float64(box.ylo)
    return PlotlyJS.scatter(
        x = [xlo, xhi, xhi, xlo, xlo],
        y = [Ylo, Ylo, Yhi, Yhi, Ylo],
        mode = "lines",
        name = name,
        showlegend = showlegend,
        hoverinfo = "skip",
        line = PlotlyJS.attr(color = colour, width = width),
    )
end

function make_hsc_plot(step6_path::AbstractString, output_path::AbstractString;
                       ngrid::Integer = DEFAULT_PLOT_GRID,
                       padding_cells::Integer = DEFAULT_PADDING_CELLS)
    step6_path = abspath(step6_path)
    step6 = read_checkpoint_envelope(step6_path, 6)
    step6.result.found || error("The Step 6 checkpoint has no certified candidate")

    setprecision(BigFloat, step6.context.bigfloat_precision) do
        step3 = read_checkpoint_envelope(supporting_checkpoint_path(step6_path, 3), 3)
        step4 = read_checkpoint_envelope(supporting_checkpoint_path(step6_path, 4), 4)
        isequal(step3.context, step6.context) || error("Step 3 and Step 6 contexts differ")
        isequal(step4.context, step6.context) || error("Step 4 and Step 6 contexts differ")

        parameters = step6.parameters
        pdeg = parameters.contraction_truncation
        derivatives = checkpoint_derivatives(step3, step4, pdeg)
        base = prepare_hsc_base_coeffs(derivatives; pdeg)
        candidate = step6.result.candidate
        xi = candidate.ξ
        xi_label = format_direction(xi)
        prepared = prepare_hsc_direction_coeffs(derivatives, base, xi; pdeg)

        step5_parameters = parameters.step5_parameters
        domain_values = step5_parameters.domain
        domain = PolytopeBox(domain_values...)
        grid = step5_parameters.grid
        window = plot_window(candidate.region, domain, grid; padding_cells)
        xs, ys, values = hsc_surface_data(prepared, window; ngrid)
        # The numerical data uses y = -Y. Reverse the rows so the plotted
        # moment coordinate Y is increasing on the vertical axis.
        moment_ys = .-reverse(ys)
        moment_values = reverse(values; dims = 1)

        finite_values = filter(isfinite, vec(values))
        sampled_min, sampled_max = extrema(finite_values)
        negative_values = filter(value -> value < 0, finite_values)
        isempty(negative_values) && error("The sampled neighbourhood contains no negative HSC")
        colour_values = asinh.(moment_values ./ COLOUR_SOFTENING)
        finite_colours = filter(isfinite, vec(colour_values))
        colour_min, colour_max = extrema(finite_colours)
        zero_fraction = -colour_min / (colour_max - colour_min)
        colourscale = [
            [0.0, "#b2182b"],
            [zero_fraction / 2, "#ef8a62"],
            [zero_fraction, "#f7f7f7"],
            [(1 + zero_fraction) / 2, "#67a9cf"],
            [1.0, "#2166ac"],
        ]

        heatmap = PlotlyJS.heatmap(
            x = xs,
            y = moment_ys,
            transpose = false,

            # Transformed values control colour:
            z = plotly_rows(colour_values),

            # PlotlyJS otherwise transposes a Julia z matrix without applying
            # the same transpose to customdata. Explicit rows keep the colour
            # and hover value attached to the same (x, Y) coordinate.
            customdata = plotly_rows(moment_values),

            zmin = colour_min,
            zmax = colour_max,
            colorscale = colourscale,
            colorbar = original_q_colourbar(sampled_min, sampled_max),

            hovertemplate =
                "x=%{x:.6f}<br>Y=%{y:.6f}<br>Q₁(V)=%{customdata:.9g}<extra></extra>",
        )

        zero_contour = PlotlyJS.contour(
            x = xs,
            y = moment_ys,
            z = plotly_rows(moment_values),
            transpose = false,
            name = "Q₁(V) = 0",
            showscale = false,
            autocontour = false,
            contours = PlotlyJS.attr(start = 0, var"end" = 0, size = 1,
                                     coloring = "none", showlabels = true),
            line = PlotlyJS.attr(color = "black", width = 2),
            hoverinfo = "skip",
            showlegend = false,
        )

        initial_box = enclosing_rectangle(candidate.region)
        region_trace = moment_rectangle_trace(
            initial_box;
            name = "certified integration region",
            colour = "#ffbf00",
            width = 3,
            showlegend=false
        )

        ix, iy = candidate.seed_cell
        seed_x = cell_centre(domain.xlo, domain.xhi, ix, grid)
        seed_y = cell_centre(domain.ylo, domain.yhi, iy, grid)
        seed_Y = -seed_y
        seed_value = point_hsc(prepared, seed_x, seed_y)
        seed_text = seed_value === nothing ? "unresolved" :
            "[$(seed_value.lower), $(seed_value.upper)]"
        seed_trace = PlotlyJS.scatter(
            x = [Float64(seed_x)],
            y = [Float64(seed_Y)],
            mode = "markers",
            name = "selected seed point",
            marker = PlotlyJS.attr(symbol = "x", size = 13, color = "black",
                                   line = PlotlyJS.attr(width = 2, color = "white")),
            text = [seed_text],
            hovertemplate =
                "selected seed<br>x=%{x:.6f}<br>Y=%{y:.6f}<br>rigorous Q₁(V)=%{text}<extra></extra>",
            showlegend = false,
        )

        # title = "HSC near the certified region, ξ = $xi_label"
        layout = PlotlyJS.Layout(
            # title = title,
            width = 900,
            height = 760,
            xaxis = PlotlyJS.attr(title = "X", constrain = "domain"),
            yaxis = PlotlyJS.attr(
                title = "Y",
                scaleanchor = "X",
                scaleratio = 1,
            ),
            legend = PlotlyJS.attr(orientation = "h", y = -0.15),
            margin = PlotlyJS.attr(l = 80, r = 40, t = 80, b = 110),
        )
        figure = PlotlyJS.Plot(
            [heatmap, zero_contour, region_trace, seed_trace],
            layout,
        )

        output_path = abspath(output_path)
        mkpath(dirname(output_path))
        PlotlyJS.savefig(figure, output_path)
        println("Wrote interactive HSC plot: ", output_path)
        println("  direction:         ξ = ", xi_label)
        println("  selected seed (x, Y): (", Float64(seed_x), ", ",
                Float64(seed_Y), ")")
        println("  sampled HSC range: [", sampled_min, ", ", sampled_max, "]")
        println("  certified region (x, Y): [", Float64(initial_box.xlo), ", ",
                Float64(initial_box.xhi), "] × [", -Float64(initial_box.yhi), ", ",
                -Float64(initial_box.ylo), "]")
        return output_path
    end
end

function main(args = ARGS)
    1 <= length(args) <= 2 || error(
        "Usage: julia --project=. bound_hsc/visualise_hsc.jl " *
        "<checkpoint-directory-or-step6.jls> [output.html]",
    )
    supplied_path = args[1]
    step6_path = isdir(supplied_path) ? joinpath(supplied_path, "step6.jls") : supplied_path
    output_path = length(args) == 2 ? args[2] :
        joinpath(dirname(step6_path), "hsc_neighbourhood.html")
    make_hsc_plot(step6_path, output_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    Base.invokelatest(main)
end
