"""
To speed up computation, approximate the polynomial to low degree, putting constant term in the interval.
"""
function lower_degree_approximation(coeffs_path::AbstractString, pdeg::Integer)
    step1_progress = start_progress("Step 1: Load and truncate u0", 1)
    step1 = step1_load_and_normalize_u0(coeffs_path)

    u_truncated = absorb_tail_into_constant(step1.raw_coeffs, pdeg)
    finish_progress!(step1_progress)
    println("Step 1: Load and truncate u0 ... ok")

    step2_progress = start_progress("Step 2: Compute second derivatives", 1)
    step2 = compute_second_derivatives(u_truncated)
    finish_progress!(step2_progress)
    println("Step 2: Compute second derivatives ... ok")

    step3_progress = start_progress("Step 3: Chebyshev products", 74)
    step3 = build_inverse_metric(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build inverse metric coefficient arrays ... ok")

    step4_progress = start_progress("Step 4: Coefficient bounds", 1)
    step4 = compute_inverse_bound_by_coefficient_bounds(step3)
    finish_progress!(step4_progress)
    println("Step 4: Bound inverse metric entries by coefficient sums ... ok")
    print_inverse_bound_summary(step4)

    return (; step1, step2, step3, step4)
    
    return u_truncated
end

function subdivide_and_bound_rigorous_local(
    coeffs_path::AbstractString,
    pdeg::Integer,
    num_subdivisions::Integer,
)
    step1_progress = start_progress("Step 1: Load and normalize u0", 1)
    step1 = step1_load_and_normalize_u0(coeffs_path)
    finish_progress!(step1_progress)
    println("Step 1: Load and normalize u0 ... ok")

    step2_progress = start_progress("Step 2: Compute full second derivatives", 1)
    step2 = compute_second_derivatives(step1.normalized_coeffs)
    finish_progress!(step2_progress)
    println("Step 2: Compute full second derivatives ... ok")

    step3_progress = start_progress("Step 3: Build full inverse metric coefficients", 74)
    step3 = build_inverse_metric(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build full inverse metric coefficient arrays ... ok")

    println("Step 4: Starting local subdivision bounds ...")
    step4_progress = start_progress("Step 4: Local subdivision boxes", num_subdivisions * num_subdivisions)
    subdivision_time = @elapsed begin
        step4 = compute_inverse_bound_by_local_subdivision_truncated(
            step3;
            pdeg = pdeg,
            nx = num_subdivisions,
            ny = num_subdivisions,
            progress = step4_progress,
        )
        global_step4 = step4
    end
    finish_progress!(step4_progress)

    println("Step 4: Local subdivision bounds ... ok")
    print_truncated_subdivision_inverse_bound_summary(global_step4)
    println()
    println("Subdivision step time: $(round(subdivision_time; digits = 3)) seconds")

    return (;
        step1,
        step2,
        step3,
        step4 = global_step4,
        subdivision_time,
        pdeg,
        num_subdivisions,
    )
end

function subdivide_and_bound_rigorous(
    coeffs_path::AbstractString,
    pdeg::Integer,
    num_subdivisions::Integer,
)
    step1_progress = start_progress("Step 1: Load and normalize u0", 1)
    step1 = step1_load_and_normalize_u0(coeffs_path)
    finish_progress!(step1_progress)
    println("Step 1: Load and normalize u0 ... ok")

    step2_progress = start_progress("Step 2: Compute full second derivatives", 1)
    step2 = compute_second_derivatives(step1.normalized_coeffs)
    finish_progress!(step2_progress)
    println("Step 2: Compute full second derivatives ... ok")

    step3_progress = start_progress("Step 3: Build full inverse metric coefficients", 74)
    step3 = build_inverse_metric(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build full inverse metric coefficient arrays ... ok")

    println("Step 4: Starting rigorous truncated subdivision bounds ...")
    step4_progress = start_progress("Step 4: Subdivision boxes", num_subdivisions * num_subdivisions)
    subdivision_time = @elapsed begin
        step4 = compute_inverse_bound_by_subdivision_truncated(
            step3;
            pdeg = pdeg,
            nx = num_subdivisions,
            ny = num_subdivisions,
            progress = step4_progress,
        )
    end
    finish_progress!(step4_progress)

    println("Step 4: Rigorous truncated subdivision bounds ... ok")
    print_truncated_subdivision_inverse_bound_summary(step4)
    println()
    println("Subdivision step time: $(round(subdivision_time; digits = 3)) seconds")

    return (; step1, step2, step3, step4, subdivision_time, pdeg, num_subdivisions)
end

function subdivide_and_bound_rigorous2(
    coeffs_path::AbstractString,
    pdeg::Integer,
    num_subdivisions::Integer,
)
    step1_progress = start_progress("Step 1: Load and normalize u0", 1)
    step1 = step1_load_and_normalize_u0(coeffs_path)
    finish_progress!(step1_progress)
    println("Step 1: Load and normalize u0 ... ok")

    step2_progress = start_progress("Step 2: Compute full second derivatives", 1)
    step2 = compute_second_derivatives(step1.normalized_coeffs)
    finish_progress!(step2_progress)
    println("Step 2: Compute full second derivatives ... ok")

    step3_progress = start_progress("Step 3: Build full inverse metric coefficients", 74)
    step3 = build_inverse_metric(step2; progress = step3_progress)
    finish_progress!(step3_progress)
    println("Step 3: Build full inverse metric coefficient arrays ... ok")

    println()
    println("Starting subdivision bounds...")
    step4_progress = start_progress("Step 4: Subdivision boxes", num_subdivisions * num_subdivisions)
    subdivision_time = @elapsed begin
        step4 = compute_inverse_bound_by_subdivision_truncated(
            step3;
            pdeg = pdeg,
            nx = num_subdivisions,
            ny = num_subdivisions,
            progress = step4_progress,
        )
        global_step4 = step4
    end
    finish_progress!(step4_progress)

    println("Step 4: Rigorous truncated subdivision bounds ... ok")
    print_truncated_subdivision_inverse_bound_summary(global_step4)

    println()
    println("Subdivision step time: $(round(subdivision_time; digits=3)) seconds")

    return (;
        step1,
        step2,
        step3,
        step4 = global_step4,
        subdivision_time,
        pdeg,
        num_subdivisions,
    )
end
