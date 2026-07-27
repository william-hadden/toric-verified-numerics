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
Enclose the Rayleigh numerator on the coefficient square `[0,1]^2`.

The square coordinates are `(x, s)`, with `s = -y` relative to the physical
negative-y sector. It is the union of two reflected fundamental sectors.
Both the metric and the test function are invariant under that reflection, so
this square has the same Rayleigh quotient as the full moment hexagon.

The centered D6-invariant test function and its gradient are

    f = x^2 - x*s + s^2 - 5/12,
    f_x = 2x - s,  f_s = -x + 2s.

Its exact squared L2 norm on the square is `7/30 - (5/12)^2 = 43/720`.

Each dyadic grid cell is enclosed by interval boxes. The inverse metric and
energy density are enclosed on that box and multiplied by its exact area. No
sign condition is imposed on the energy enclosure.
"""
function enclose_rayleigh_numerator(inverse_coefficients)
    one = interval(BigFloat, 1)
    two = interval(BigFloat, 2)
    cell_area = interval(BigFloat, 1 // N^2)
    boxes = subdivide_minus_one_one(N)
    numerator = interval(BigFloat, 0)

    for xcheb in boxes, scheb in boxes
        xbox = (xcheb + one) / two
        sbox = (one - scheb) / two
        metric = inverse_metric_box_bounds_with_tail(
            inverse_coefficients,
            xcheb,
            scheb,
        )

        qx = two * xbox - sbox
        qs = -xbox + two * sbox
        energy =
            metric.u11_box * qx^2 +
            two * metric.u12_box * qx * qs +
            metric.u22_box * qs^2
        numerator += cell_area * energy
    end

    return numerator
end

"""Compute and store the approximate-metric upper eigenvalue bound."""
function run_upper_bound()
    setprecision(BigFloat, PRECISION) do
        coefficients = load_rational_coeffs_csv(U0_PATH)
        derivatives = compute_second_derivatives(coefficients)
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

        bounds = JSON.parsefile(VERIFIED_BOUNDS_PATH)
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
