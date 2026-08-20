using IntervalArithmetic
using Logging

global_logger(NullLogger())

include(joinpath(@__DIR__, "..", "apply_fixed_point", "apply_fixed_point.jl"))
include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "..", "bound_metric","util", "inverse_bounds.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_subdivision_bounds.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "curvature_bounds.jl"))
include(joinpath(@__DIR__, "checkpoints.jl"))
include(joinpath(@__DIR__, "util.jl"))
include(joinpath(@__DIR__, "arithmetic_helpers.jl"))



function find_U_V_proposition_negative_hsc_at_point(
    coeffs_path::AbstractString;
    eta,
    rho,
    pdeg::Integer = 30,
    domain::PolytopeBox = PolytopeBox(0, 1, 0, 1),
)
    println("Step 1: Load u0")
    step1 = load_rational_coeffs_csv(coeffs_path)
    println("Step 1: Load u0 ... ok")

    checkpoint_context = hsc_checkpoint_context(coeffs_path)
    pdeg > 0 || error("HSC coefficient truncation requires pdeg > 0")
    grid = 24
    direction_order = 5
    directions_per_cell = 2
    max_seeds = 100
    seed_cutoff = 0
    candidate_shapes = (
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 2),
    )
    screen_depth = 0
    maxdepth = 6
    beam_width = 20
    invalid_keep = 20
    run_parameters = (
        coefficient_sha256 = checkpoint_context.coefficient_sha256,
        bigfloat_precision = checkpoint_context.bigfloat_precision,
        pdeg,
        domain = (domain.xlo, domain.xhi, domain.ylo, domain.yhi),
        eta,
        rho,
        grid,
        direction_order,
        directions_per_cell,
        max_seeds,
        seed_cutoff,
        candidate_shapes,
        screen_depth,
        maxdepth,
        beam_width,
        invalid_keep,
    )
    run_checkpoint_dir = hsc_run_checkpoint_dir(run_parameters)
    println("Run checkpoints and parameters: $run_checkpoint_dir")

    step2_parameters = (operation = :second_derivatives,)

    step2 = load_hsc_checkpoint_if_valid(
        2;
        expected_context = checkpoint_context,
        expected_parameters = step2_parameters,
    )
    if step2 !== nothing
        println("Step 2: Loaded checkpoint: $(hsc_checkpoint_path(2))")
    else
        step2_progress = start_progress("Step 2: Compute full second derivatives", 1)
        # step2 = compute_second_derivatives(step1)
        step2 = build_derivative_pack(step1)
        finish_progress!(step2_progress)
        step2_checkpoint = save_hsc_checkpoint(
            2,
            step2;
            context = checkpoint_context,
            parameters = step2_parameters,
        )
        println(
            "Step 2: Compute full second derivatives ... ok " *
            "(checkpoint: $step2_checkpoint)",
        )
    end

    step3_parameters = (operation = :inverse_metric_coefficients,)
    step3 = load_hsc_checkpoint_if_valid(
        3;
        expected_context = checkpoint_context,
        expected_parameters = step3_parameters,
    )
    if step3 !== nothing
        println("Step 3: Loaded checkpoint: $(hsc_checkpoint_path(3))")
    else
        step3_progress = start_progress("Step 3: Build inverse metric coefficients", 74)
        step3 = build_inverse_metric_coeffs(step2; progress = step3_progress)
        finish_progress!(step3_progress)
        step3_checkpoint = save_hsc_checkpoint(
            3,
            step3;
            context = checkpoint_context,
            parameters = step3_parameters,
        )
        println(
            "Step 3: Build inverse metric coefficient arrays ... ok " *
            "(checkpoint: $step3_checkpoint)",
        )
    end

    step4_parameters = (operation = :inverse_derivatives, k = 2, pdeg)
    step4 = load_hsc_checkpoint_if_valid(
        4;
        expected_context = checkpoint_context,
        expected_parameters = step4_parameters,
        checkpoint_dir = run_checkpoint_dir,
    )
    if step4 !== nothing
        println("Step 4: Loaded checkpoint: $(hsc_checkpoint_path(4; checkpoint_dir = run_checkpoint_dir))")
    else
        step4_progress = start_progress("Step 4: Compute derivatives", 144)
        step4 = compute_inverse_derivative_numerator_components_truncated_coeff_space(
            step3;
            k = 2,
            pdeg,
            progress = step4_progress,
        )
        finish_progress!(step4_progress)
        step4_checkpoint = save_hsc_checkpoint(
            4,
            step4;
            context = checkpoint_context,
            parameters = step4_parameters,
            checkpoint_dir = run_checkpoint_dir,
        )
        println("Step 4: Compute derivatives ... ok (checkpoint: $step4_checkpoint)")
    end

    inverse_value_enclosures = prepare_inverse_coeffs_for_subdivision(step3, pdeg)
    derivatives = (
        deriv_num = step4.deriv_num,
        D = enclosure_to_coeffs(inverse_value_enclosures.D),
        pack = step4.pack,
        pdeg = step4.pdeg,
    )
    step5_parameters = (
        operation = :generate_hsc_candidates,
        domain = (domain.xlo, domain.xhi, domain.ylo, domain.yhi),
        grid,
        direction_order,
        directions_per_cell,
        max_seeds,
        seed_cutoff,
        pdeg,
        shapes = candidate_shapes,
    )
    candidates = load_hsc_checkpoint_if_valid(
        5;
        expected_context = checkpoint_context,
        expected_parameters = step5_parameters,
        checkpoint_dir = run_checkpoint_dir,
    )
    if candidates !== nothing
        println(
            "Step 5: Loaded $(length(candidates)) candidate regions from " *
            "checkpoint: $(hsc_checkpoint_path(5; checkpoint_dir = run_checkpoint_dir))",
        )
    else
        seed_evaluations = grid * grid * length(primitive_directions(direction_order))
        step5_progress = start_progress("Step 5: Generate HSC candidates", seed_evaluations)
        candidates = generate_hsc_candidates(
            derivatives,
            domain;
            grid,
            direction_order,
            directions_per_cell,
            max_seeds,
            seed_cutoff,
            shapes = candidate_shapes,
            progress = step5_progress,
        )
        finish_progress!(step5_progress)
        step5_checkpoint = save_hsc_checkpoint(
            5,
            candidates;
            context = checkpoint_context,
            parameters = step5_parameters,
            checkpoint_dir = run_checkpoint_dir,
        )
        println(
            "Step 5: Generated $(length(candidates)) candidate regions ... ok " *
            "(checkpoint: $step5_checkpoint)",
        )
    end

    step6_parameters = (
        operation = :find_compatible_U_V,
        diagnostic_reporting_version = 4,
        curvature_evaluation = :local_coefficient_contraction,
        contraction_truncation = pdeg,
        eta,
        rho,
        screen_depth,
        maxdepth,
        beam_width,
        invalid_keep,
        step5_parameters,
    )
    result = load_hsc_checkpoint_if_valid(
        6;
        expected_context = checkpoint_context,
        expected_parameters = step6_parameters,
        checkpoint_dir = run_checkpoint_dir,
    )
    if result !== nothing
        println("Step 6: Loaded search result from checkpoint: $(hsc_checkpoint_path(6; checkpoint_dir = run_checkpoint_dir))")
    else
        search_evaluations = length(candidates) +
                             (maxdepth - screen_depth) * (beam_width + invalid_keep)
        step6_progress =
            start_progress("Step 6: Search compatible U and V", search_evaluations)
        result = find_compatible_U_V(
            derivatives,
            eta,
            rho,
            candidates;
            pdeg,
            screen_depth,
            maxdepth,
            beam_width,
            invalid_keep,
            progress = step6_progress,
        )
        finish_progress!(step6_progress)
        step6_checkpoint = save_hsc_checkpoint(
            6,
            result;
            context = checkpoint_context,
            parameters = step6_parameters,
            checkpoint_dir = run_checkpoint_dir,
        )
        println("Step 6: Search result saved (checkpoint: $step6_checkpoint)")
    end

    if result.found
        println(
            "Certified Proposition 5.6 with margin ",
            result.proof.margin,
        )
    elseif result.best !== nothing
        println(
            "No certificate. Best upper margin was ",
            result.best.proof.margin_upper,
        )
    elseif haskey(result, :best_unresolved) && result.best_unresolved !== nothing
        unresolved = result.best_unresolved
        println(
            "No certificate. Best unresolved candidate was direction ",
            unresolved.candidate.ξ,
            ", seed cell ",
            unresolved.candidate.seed_cell,
            ", shape ",
            unresolved.candidate.shape,
            "; failure reason: ",
            unresolved.proof.diagnostic.reason,
            ". Full diagnostic saved in the step 6 checkpoint.",
        )
    else
        println("No viable candidate was found")
    end

    println("\nHSC run parameters:")
    println("  BigFloat precision:       $(precision(BigFloat)) bits")
    println("  pdeg:                    $pdeg")
    println("  domain:                  ($(domain.xlo), $(domain.xhi)) × ($(domain.ylo), $(domain.yhi))")
    println("  eta:                     $eta")
    println("  rho:                     $rho")
    println("  seed grid:               $(grid) × $(grid)")
    println("  direction order:         $direction_order")
    println("  directions per cell:     $directions_per_cell")
    println("  maximum seeds:           $max_seeds")
    println("  seed cutoff:             $seed_cutoff")
    println("  candidate shapes:        $candidate_shapes")
    println("  generated candidates:    $(length(candidates))")
    println("  screen depth:            $screen_depth")
    println("  maximum search depth:    $maxdepth")
    println("  beam width:              $beam_width")
    println("  unresolved candidates kept: $invalid_keep")
    println("  checkpoint folder:       $run_checkpoint_dir")
    println("  parameter README:        $(joinpath(run_checkpoint_dir, "README.txt"))")

    selected = if result.found
        result
    elseif result.best !== nothing
        result.best
    elseif haskey(result, :best_unresolved) && result.best_unresolved !== nothing
        result.best_unresolved
    else
        nothing
    end

    if selected !== nothing
        println("  selected direction:      $(selected.candidate.ξ)")
        println("  selected seed cell:      $(selected.candidate.seed_cell)")
        println("  selected initial shape:  $(selected.candidate.shape)")
        println("  selected final depth:    $(selected.depth)")
        println("  selected final boxes:    $(length(selected.boxes))")
    end
end

"""
can use lemma 5.1 and lemma 5.3 to set these 
"""
function set_eta_rho()
    epsilon = read_bound("fixed_point_bounds")["contraction_radius_upper_bound"] 

    const_emb_C1, const_emb_C2, const_emb_C3, const_emb_C4 = get_sobolev_multiplication_constants()
    C_1 = const_emb_C4 * const_emb_C3 * const_emb_C2 * const_emb_C1
    eta = C_1 * epsilon

    eta < 1 ||
    throw(DomainError(C_1 * epsilon, "C_1 * epsilon must be less than 1"))

    rho = epsilon/(interval(1)-C_1*epsilon) +  (const_emb_C4^2*epsilon^2)/(interval(1)-C_1*epsilon)^2

    return eta, rho
end

function main()
    setprecision(BigFloat, 200) 
    eta, rho = set_eta_rho()
    println("Find valid U,V with eta: $eta, rho: $rho")
    find_U_V_proposition_negative_hsc_at_point(U0_PATH; eta, rho)
end

# "c:\\Users\\willi\\.vscode\\extensions\\julialang.language-julia-1.219.2\\scripts\\debugger\\run_debugger.jl"
if abspath(PROGRAM_FILE) == @__FILE__
    Base.invokelatest(main)
end
