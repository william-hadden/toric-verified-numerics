using IntervalArithmetic
using Logging

global_logger(NullLogger())

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_subdivision_bounds.jl"))

const PRECISION = 256
const PDEG = 20
const N = 256

"""
Enclose the Rayleigh numerator on the square `[0,1]^2`.

The square coordinates are `(x, s)`, with `s = -y` relative to the physical
y-coordinate. It is the union of two adjacent symmetry-related triangular
sectors. Both the metric and the test function are invariant under the
reflection relating them, so this square has the same Rayleigh quotient as the
full moment hexagon.

The centered D6-invariant test function and its gradient are

    f = x^2 - x*s + s^2 - 5/12,
    f_x = 2x - s,  f_s = -x + 2s.

Its exact squared L2 norm on the square is `7/30 - (5/12)^2 = 43/720`.

Each grid cell is enclosed by interval boxes. The inverse metric and
|grad f|^2 are enclosed on that box and multiplied by its exact area. No
sign condition is imposed on the energy enclosure.
"""
function enclose_rayleigh_numerator(inverse_coefficients)
    cell_area = interval(BigFloat, 1 // N^2)
    boxes = subdivide_minus_one_one(N)
    numerator = interval(BigFloat, 0)

    for xcheb in boxes, scheb in boxes
        # The Chebyshev coordinates are xcheb = 2x - 1 and scheb = 1 - 2s.
        xbox = (xcheb + exact(1)) / exact(2)
        sbox = (exact(1) - scheb) / exact(2)
        metric = inverse_metric_box_bounds_with_tail(
            inverse_coefficients,
            xcheb,
            scheb,
        )

        qx = exact(2) * xbox - sbox
        qs = -xbox + exact(2) * sbox
        energy =
            metric.u11_box * qx^2 +
            exact(2) * metric.u12_box * qx * qs +
            metric.u22_box * qs^2
        numerator += cell_area * energy
    end

    return numerator
end

"""Compute and store the approximate-metric upper eigenvalue bound."""
function run_upper_bound()
    setprecision(BigFloat, PRECISION) do
        coefficients = load_rational_coeffs_csv(U0_PATH)
        # derivatives = compute_second_derivatives(coefficients)
        derivatives = build_derivative_pack(coefficients)
        inverse_coefficients = prepare_inverse_coeffs_for_subdivision(
            build_inverse_metric_coeffs(derivatives),
            PDEG,
        )

        numerator = enclose_rayleigh_numerator(inverse_coefficients)
        L2_norm_squared = interval(BigFloat, 43 // 720)
        rayleigh = numerator / L2_norm_squared
        # Keep nearest-decimal serialization above the interval supremum.
        decimal_unit = interval(BigFloat(10))^-SERIALIZED_BOUND_DECIMAL_DIGITS
        serialized = serialize_bound_value(rayleigh + decimal_unit)

        bounds = read_bounds_json(VERIFIED_BOUNDS_PATH)
        bounds["lambda_1_upper_bound"] = serialized
        open(VERIFIED_BOUNDS_PATH, "w") do io
            JSON.print(io, bounds, 4)
            println(io)
        end

        println("Rayleigh numerator enclosure: $numerator")
        println("Rayleigh quotient enclosure: $rayleigh")
        println("wrote lambda_1_upper_bound = $serialized")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_upper_bound()
end
