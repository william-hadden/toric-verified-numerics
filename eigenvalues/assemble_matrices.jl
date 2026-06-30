using IntervalArithmetic
using Logging

Logging.disable_logging(Logging.Info)
setprecision(BigFloat, 100)

include(joinpath(@__DIR__, "inverse_metric_box_bounds.jl"))

"""
Return a hardcoded four-triangle interval triangulation of P_delta.

Here P is the lower-square fundamental triangle with vertices (0, 0), (1, 0),
and (1, -1). For now delta is fixed to 0.1, so P_delta has vertices (0, 0),
(0.9, 0), and (0.9, -0.9).

The returned `nodes` is an n x 2 matrix of interval coordinates, and
`triangles` is a vector of triples of node indices. All geometric coordinates
are intervals so later mesh-generation and inset operations can preserve
outward containment without changing the assembly API.
"""
function get_pdelta_triangulation()
    I(x) = interval(BigFloat(x, RoundDown), BigFloat(x, RoundUp))

    nodes = [
        I(0//1)   I(0//1)
        I(9//20)  I(0//1)
        I(9//20) -I(9//20)
        I(9//10)  I(0//1)
        I(9//10) -I(9//20)
        I(9//10) -I(9//10)
    ]

    triangles = [
        (1, 3, 2),
        (2, 5, 4),
        (2, 3, 5),
        (3, 6, 5),
    ]

    return nodes, triangles
end

"""
Compute interval bounds for the inverse metric on one triangle of P_delta.

The triangle is enclosed in an axis-aligned interval box in the lower-square
coordinates, and `inverse_metric_bounds_on_lower_square_box` is called on that
box. The return value contains only the four interval tensor entries needed by
stiffness assembly: `xx`, `xy`, `yx`, and `yy`.
"""
function metric_bound_for_single_triangle(oracle, nodes, tri)
    xs = nodes[collect(tri), 1]
    ys = nodes[collect(tri), 2]
    xbox = hull(hull(xs[1], xs[2]), xs[3])
    ybox = hull(hull(ys[1], ys[2]), ys[3])

    U = inverse_metric_bounds_on_lower_square_box(oracle, xbox, ybox)
    return (; xx = U.xx, xy = U.xy, yx = U.yx, yy = U.yy)
end

"""
Compute inverse-metric interval bounds for every triangle in a P_delta mesh.

`nodes` and `triangles` are the output of `get_pdelta_triangulation`. The
inverse-metric oracle is built once and reused for all triangles. The returned
vector has the same order as `triangles`; entry `k` contains interval bounds
`xx`, `xy`, `yx`, and `yy` for `triangles[k]`.
"""
function metric_bounds_on_pdelta_triangulation(nodes, triangles)
    oracle = InverseMetricBoxOracle()
    return [metric_bound_for_single_triangle(oracle, nodes, tri) for tri in triangles]
end

"""
Run the current Step 1 prototype.

This constructs the hardcoded four-triangle P_delta triangulation and computes
triangle-wise inverse-metric bounds in memory. Nothing is written to disk.
"""
function main()
    nodes, triangles = get_pdelta_triangulation()
    println("P_delta triangulation: $(size(nodes, 1)) nodes, $(length(triangles)) triangles")

    elapsed = @elapsed bounds = metric_bounds_on_pdelta_triangulation(nodes, triangles)
    println("metric bounds computed in $(round(elapsed; digits = 3)) seconds")

    for (k, U) in pairs(bounds)
        println("triangle $k:")
        println("    xx = $(U.xx)")
        println("    xy = $(U.xy)")
        println("    yx = $(U.yx)")
        println("    yy = $(U.yy)")
    end

    return (; nodes, triangles, bounds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
