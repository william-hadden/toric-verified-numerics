"""
Print the Q term in the holomorphic sectional curvature for several constant
vector fields and sample points in the moment polytope. It also writes an
interactive 3D HTML surface for each vector field to `bound_hsc/Q_plots`.

Run from the repository root with

    julia --project=. bound_hsc/print_Q.jl

The entries in `CONSTANT_VECTOR_FIELDS` and `SAMPLE_COORDINATES` are intended
to be easy to edit.  Q is homogeneous of degree zero in the vector field, so
only its direction matters (for example, `(1, 0)` and `(2, 0)` give the same
answer).
"""

include(joinpath(@__DIR__, "bound_hsc.jl"))
using PlotlyJS

const CONSTANT_VECTOR_FIELDS = [
    (label = "e1",       xi = (1, 0)),
    (label = "e2",       xi = (0, 1)),
    (label = "e1 + e2",  xi = (1, 1)),
    (label = "e1 - e2",  xi = (1, -1)),
    (label = "2e1 + e2", xi = (2, 1)),
]

# Taking the Cartesian product of these coordinates gives nine sample points.
const SAMPLE_COORDINATES = (big(1) // 4, big(1) // 2, big(3) // 4)

function build_hsc_derivatives(coeffs_path::AbstractString)
    println("Building curvature data from: ", coeffs_path)
    metric = load_rational_coeffs_csv(coeffs_path)
    # second_derivatives = compute_second_derivatives(metric)
    # inverse_metric = build_inverse_metric_coeffs(second_derivatives)
    derivatives = build_derivative_pack(metric)
    inverse_metric = build_inverse_metric_coeffs(derivatives)
    inverse_derivatives =
        compute_inverse_derivative_numerator_components_truncated_coeff_space(
            inverse_metric;
            k = 2,
        )

    return (;
        deriv_num = inverse_derivatives.deriv_num,
        D = inverse_derivatives.D,
    )
end

function interval_string(I; digits::Integer = 10)
    lo = Float64(inf(I))
    hi = Float64(sup(I))
    return "[" * string(round(lo; sigdigits = digits)) * ", " *
           string(round(hi; sigdigits = digits)) * "]"
end

function print_Q_table(derivatives)
    println()
    println("Q for constant vector fields (rigorous point intervals)")
    println("x\ty\tvector field\txi\tQ")

    for y in SAMPLE_COORDINATES, x in SAMPLE_COORDINATES
        X = exact_big_interval(x)
        Y = exact_big_interval(y)
        geometry = evaluate_geometry_on_box(derivatives, X, Y)
        geometry === nothing && error("Could not certify inverse metric at (x, y) = ($x, $y)")

        for field in CONSTANT_VECTOR_FIELDS
            Q = compute_Q_rigorously(geometry, field.xi, X, Y)
            Q === nothing && error("Could not certify Q at (x, y) = ($x, $y) for xi = $(field.xi)")
            isguaranteed(Q) || error("Non-guaranteed Q at (x, y) = ($x, $y) for xi = $(field.xi)")
            println(
                Float64(x), '\t', Float64(y), '\t',
                field.label, '\t', field.xi, '\t', interval_string(Q),
            )
        end
    end
end

"""
Evaluate the midpoint of the rigorous Q enclosure on an `ngrid` by `ngrid`
grid.  The endpoints are kept slightly inside the moment polytope because its
boundary is singular in these coordinates.
"""
function Q_surface_data(derivatives, xi; ngrid::Integer = 21)
    ngrid >= 2 || error("ngrid must be at least 2")
    coordinates = collect(range(0.02, 0.98; length = ngrid))
    values = Matrix{Float64}(undef, ngrid, ngrid)

    for (iy, y) in enumerate(coordinates), (ix, x) in enumerate(coordinates)
        X = interval(BigFloat, BigFloat(x))
        Y = interval(BigFloat, BigFloat(y))
        geometry = evaluate_geometry_on_box(derivatives, X, Y)
        geometry === nothing && error("Could not certify inverse metric at ($x, $y)")
        Q = compute_Q_rigorously(geometry, xi, X, Y)
        Q === nothing && error("Could not certify Q at ($x, $y) for xi = $xi")
        values[iy, ix] = Float64((inf(Q) + sup(Q)) / exact(2))
    end

    return coordinates, values
end


"""
Write an interactive Plotly 3D surface of Q for the constant vector field
`xi`.  The resulting HTML supports rotation, zooming, panning, and hover
inspection.  Returns the minimum and maximum sampled values.
"""
function print_Q_3dplot(
    derivatives,
    xi,
    output_path::AbstractString;
    label::AbstractString = string(xi),
    ngrid::Integer = 21,
)
    coordinates, values = Q_surface_data(derivatives, xi; ngrid)
    zmin, zmax = extrema(values)
    mkpath(dirname(output_path))

    trace = PlotlyJS.surface(
        x = coordinates,
        y = coordinates,
        z = values,
        colorscale = "Viridis",
        colorbar = PlotlyJS.attr(title = "Q"),
        hovertemplate = "x=%{x:.4f}<br>y=%{y:.4f}<br>Q=%{z:.8f}<extra></extra>",
    )
    layout = PlotlyJS.Layout(
        title = "Q term for xi = $label",
        width = 900,
        height = 700,
        scene = PlotlyJS.attr(
            xaxis = PlotlyJS.attr(title = "x"),
            yaxis = PlotlyJS.attr(title = "y"),
            zaxis = PlotlyJS.attr(title = "Q"),
            aspectmode = "cube",
        ),
    )
    figure = PlotlyJS.Plot(trace, layout)
    PlotlyJS.savefig(figure, output_path)

    println("Wrote ", output_path, "  Q range = [", zmin, ", ", zmax, "]")
    return (; minimum = zmin, maximum = zmax)
end

function print_constant_vector_field_plots(derivatives)
    output_dir = joinpath(@__DIR__, "Q_plots")
    for (index, field) in enumerate(CONSTANT_VECTOR_FIELDS)
        filename = "Q_" * lpad(string(index), 2, '0') * ".html"
        print_Q_3dplot(
            derivatives,
            field.xi,
            joinpath(output_dir, filename);
            label = field.label,
        )
    end
end

function main()
    setprecision(BigFloat, 100) do
        derivatives = build_hsc_derivatives(U0_PATH)
        print_Q_table(derivatives)
        print_constant_vector_field_plots(derivatives)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
