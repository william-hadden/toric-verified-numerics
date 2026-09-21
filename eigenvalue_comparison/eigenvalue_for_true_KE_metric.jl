using IntervalArithmetic

include(joinpath(@__DIR__, "..", "apply_fixed_point", "apply_fixed_point.jl"))

println("Begin eigenvalue bounding w.r.t. true KE metric")

setprecision(BigFloat, 256) do
    epsilon = read_bound("fixed_point_bounds")["contraction_radius_upper_bound"]
    constants = get_sobolev_multiplication_constants()
    C1_epsilon =
        interval(BigFloat(4)) *
        constants.const_emb_C4 *
        constants.const_emb_C3 *
        constants.const_emb_C2 *
        constants.const_emb_C1 *
        epsilon

    C1_epsilon = interval(sup(C1_epsilon))

    lower_factor = (exact(1) - C1_epsilon)^2 / (exact(1) + C1_epsilon)^3
    upper_factor = (exact(1) + C1_epsilon)^2 / (exact(1) - C1_epsilon)^3
    lower_bound = lower_factor * read_bound("lambda_1_lower_bound")
    upper_bound = upper_factor * read_bound("lambda_1_upper_bound")

    # Pad the relevant endpoints before nearest-decimal serialization.
    decimal_unit = interval(BigFloat(10))^-SERIALIZED_BOUND_DECIMAL_DIGITS
    lower_text = serialize_bound_value(inf(lower_bound - decimal_unit))
    upper_text = serialize_bound_value(sup(upper_bound + decimal_unit))

    println("True KE invariant lambda_1 lower bound: $lower_text")
    println("True KE invariant lambda_1 upper bound: $upper_text")
end

println("Eigenvalue bounding complete")
