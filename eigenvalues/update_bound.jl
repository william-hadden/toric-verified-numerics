using IntervalArithmetic
using JSON
using Printf

isdefined(@__MODULE__, :parse_bound_value) ||
    include(joinpath(@__DIR__, "..", "utils", "io.jl"))
isdefined(@__MODULE__, :read_mat4_variables) ||
    include(joinpath(@__DIR__, "util", "mat4.jl"))

"""
Read the MATLAB certificate at `path`.

Return its lower and upper endpoints after checking that INTLAB certified
global generalized eigenvalue 2.
"""
function verified_fem_endpoints(path)
    data = read_mat4_variables(path)
    required = ("lambda_fem_lb", "lambda_fem_ub", "lambda_fem_ind")
    all(name -> size(data[name]) == (1, 1), required) ||
        error("The verified eigenvalue variables must be scalars")
    lower = data["lambda_fem_lb"][1]
    upper = data["lambda_fem_ub"][1]
    index = data["lambda_fem_ind"][1]
    all(isfinite, (lower, upper, index)) ||
        error("The verified eigenvalue data are not finite")
    index == 2 || error("INTLAB did not certify generalized eigenvalue 2")
    lower <= upper || error("The verified eigenvalue endpoints are reversed")
    return lower, upper
end

"""
Apply Liu's comparison to `lambda_fem_lower`.

`comparison_constant` is an interval upper bound for the FEM comparison
constant. Return a rigorous smooth inset eigenvalue lower bound.
"""
function liu_lower_bound(lambda_fem_lower, comparison_constant)
    lambda = interval(BigFloat(lambda_fem_lower))
    denominator = interval(BigFloat(1)) + comparison_constant^2 * lambda
    inf(denominator) > 0 ||
        error("Liu's comparison denominator is not positive")
    return inf(lambda / denominator)
end

"""
Apply the inset-to-compact comparison to `lambda_inset_lower`.

`delta` is the inset interval and `ricci_lower_bound` is its serialized Ricci
lower bound. Return a rigorous compact-manifold eigenvalue lower bound.
"""
function compact_lower_bound(lambda_inset_lower, delta, ricci_lower_bound)
    lambda = interval(BigFloat(lambda_inset_lower))
    mu = parse_bound_value(ricci_lower_bound)
    one, two = interval(BigFloat(1)), interval(BigFloat(2))
    three, seven = interval(BigFloat(3)), interval(BigFloat(7))
    inf(mu) > 0 || error("The Ricci lower bound must be positive")
    inf(delta) > 0 || error("The inset parameter must be positive")
    inf(three - seven * delta) > 0 ||
        error("The inset parameter must be smaller than 3/7")
    square_root = sqrt(seven * delta)
    boundary_fraction = sqrt(seven * delta / three)
    c = one - boundary_fraction -
        square_root / (two * interval(BigFloat, pi) * (three - seven * delta))
    b = boundary_fraction * three / (two * mu)
    inf(c) > 0 || error("The compact-comparison numerator is not positive")
    denominator = one + lambda * b
    inf(denominator) > 0 ||
        error("The compact-comparison denominator is not positive")
    return inf(lambda * c / denominator)
end

"""
Serialize `value` with `digits` places after the decimal point.

Return a decimal string whose exact value is at most `value`.
"""
function directed_lower_decimal(value, digits)
    target = BigFloat(value)
    isfinite(target) || error("Cannot serialize a non-finite lower bound")
    candidate = prevfloat(target)
    while true
        text = @sprintf("%.*f", digits, candidate)
        parsed_upper = setrounding(BigFloat, RoundUp) do
            parse(BigFloat, text)
        end
        parsed_upper <= target && return text
        candidate = prevfloat(candidate)
    end
end

"""
Write the two eigenvalue lower bounds into `bounds` at JSON `path`.

Both bounds are rounded downward to `digits` places. Return nothing.
"""
function write_eigenvalue_bounds(path, bounds, inset_lower, compact_lower, digits)
    bounds["lambda_1_delta_lower_bound"] =
        directed_lower_decimal(inset_lower, digits)
    bounds["lambda_1_lower_bound"] =
        directed_lower_decimal(compact_lower, digits)
    open(path, "w") do io
        JSON.print(io, bounds, 4)
        println(io)
    end
end

"""Read the fixed FEM certificate and update the two JSON eigenvalue bounds."""
function run_bound_update()
    setprecision(BigFloat, 256) do
        decimal_digits = 77
        bounds_path = normpath(
            joinpath(@__DIR__, "..", "data", "verified_bounds.json"),
        )
        bounds = JSON.parsefile(bounds_path)
        delta = parse_rational_interval(bounds["delta_inset"])
        comparison_constant =
            parse_bound_value(bounds["Liu_FEM_comparison_constant"])
        result_path = joinpath(@__DIR__, "verified_eigenvalue.mat")
        lambda_fem_lower, lambda_fem_upper =
            verified_fem_endpoints(result_path)
        lambda_inset_lower =
            liu_lower_bound(lambda_fem_lower, comparison_constant)
        lambda_compact_lower = compact_lower_bound(
            lambda_inset_lower,
            delta,
            bounds["curvature_bounds"]["ricci_lower_bound"],
        )
        write_eigenvalue_bounds(
            bounds_path,
            bounds,
            lambda_inset_lower,
            lambda_compact_lower,
            decimal_digits,
        )
        println("FEM eigenvalue enclosure: [$lambda_fem_lower, $lambda_fem_upper]")
        println("smooth inset lower bound: $lambda_inset_lower")
        println("compact-manifold lower bound: $lambda_compact_lower")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_bound_update()
end
