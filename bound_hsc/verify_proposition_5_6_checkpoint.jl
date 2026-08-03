using IntervalArithmetic
using Logging
using Serialization

global_logger(NullLogger())

include(joinpath(@__DIR__, "..", "utils", "load_common.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_coefficients.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_bounds.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_truncation.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "inverse_subdivision_bounds.jl"))
include(joinpath(@__DIR__, "..", "bound_metric", "util", "curvature_bounds.jl"))
include(joinpath(@__DIR__, "checkpoints.jl"))
include(joinpath(@__DIR__, "util.jl"))
include(joinpath(@__DIR__, "arithmetic_helpers.jl"))

function read_checkpoint_envelope(path::AbstractString, expected_step::Integer)
    isfile(path) || throw(ArgumentError("Checkpoint does not exist: $path"))
    checkpoint = try
        open(deserialize, path)
    catch exception
        throw(ArgumentError("Unable to deserialize checkpoint $path: $exception"))
    end

    required = (:format_version, :step, :context, :parameters, :result)
    checkpoint isa NamedTuple && all(key -> haskey(checkpoint, key), required) ||
        throw(ArgumentError("Legacy or invalid HSC checkpoint: $path"))
    checkpoint.format_version == HSC_CHECKPOINT_FORMAT_VERSION ||
        throw(ArgumentError("Unsupported HSC checkpoint format: $path"))
    checkpoint.step == expected_step || throw(ArgumentError(
        "Expected a Step $expected_step checkpoint, but $path is Step $(checkpoint.step)",
    ))
    return checkpoint
end

function supporting_checkpoint_path(step6_path::AbstractString, step::Integer)
    run_dir = dirname(step6_path)
    candidates = (
        joinpath(run_dir, "step$(step).jls"),
        joinpath(dirname(run_dir), "step$(step).jls"),
    )
    index = findfirst(isfile, candidates)
    index === nothing && throw(ArgumentError(
        "Could not find step$(step).jls in $(run_dir) or $(dirname(run_dir))",
    ))
    return candidates[index]
end

function selected_certificate_data(result)
    result.found || error(
        "The Step 6 checkpoint does not contain a successful search result",
    )
    haskey(result, :boxes) || error("The Step 6 result has no saved final region")
    haskey(result, :candidate) || error("The Step 6 result has no saved vector field")
    return result.boxes, result.candidate.ξ
end

"""
Recompute the Proposition 5.6 bound on the region and constant vector field
saved in a successful HSC Step 6 checkpoint. This performs no point search,
candidate generation, beam search, or region refinement.
"""
function verify_proposition_5_6_checkpoint(step6_path::AbstractString)
    step6_path = abspath(step6_path)
    step6 = read_checkpoint_envelope(step6_path, 6)

    precision_bits = step6.context.bigfloat_precision
    setprecision(BigFloat, precision_bits)

    step3_path = supporting_checkpoint_path(step6_path, 3)
    step4_path = supporting_checkpoint_path(step6_path, 4)
    step3 = read_checkpoint_envelope(step3_path, 3)
    step4 = read_checkpoint_envelope(step4_path, 4)

    isequal(step3.context, step6.context) ||
        error("Step 3 and Step 6 checkpoint contexts do not match")
    isequal(step4.context, step6.context) ||
        error("Step 4 and Step 6 checkpoint contexts do not match")

    parameters = step6.parameters
    all(key -> haskey(parameters, key), (:contraction_truncation, :eta, :rho)) ||
        error("Step 6 checkpoint parameters do not contain degree, eta, and rho")
    pdeg = parameters.contraction_truncation
    step4.parameters.pdeg == pdeg ||
        error("Step 4 and Step 6 checkpoints use different pdeg values")

    inverse_value_enclosures =
        prepare_inverse_coeffs_for_subdivision(step3.result, pdeg)
    derivatives = (
        deriv_num = step4.result.deriv_num,
        D = enclosure_to_coeffs(inverse_value_enclosures.D),
        pack = step4.result.pack,
        pdeg = step4.result.pdeg,
    )

    boxes, ξ = selected_certificate_data(step6.result)
    println("Verifying the saved Proposition 5.6 candidate")
    println("  checkpoint:         $step6_path")
    println("  precision:          $precision_bits bits")
    println("  pdeg:               $pdeg")
    println("  vector field:       $ξ")
    println("  integration boxes:  $(length(boxes))")
    println("  eta:                $(parameters.eta)")
    println("  rho:                $(parameters.rho)")

    prepared_base = prepare_hsc_base_coeffs(derivatives; pdeg)
    prepared_direction =
        prepare_hsc_direction_coeffs(derivatives, prepared_base, ξ; pdeg)
    proof = bound_proposition_5_6_margin(
        derivatives,
        boxes,
        ξ,
        parameters.eta,
        parameters.rho;
        prepared = prepared_direction,
        pdeg,
    )

    println("\nFresh rigorous bound")
    println("  Q integral:         $(proof.Q_integral)")
    println("  Riemann L2 bound:   $(proof.Rnorm)")
    println("  volume:             $(proof.volume)")
    println("  correction:         $(proof.correction)")
    println("  margin:             $(proof.margin)")
    println("  margin upper bound: $(proof.margin_upper)")
    println("  certified:          $(proof.certified)")

    proof.valid || error("Proposition 5.6 evaluation was not valid")
    proof.certified || error(
        "Proposition 5.6 was not certified: margin upper bound = $(proof.margin_upper)",
    )
    return proof
end

function main(args = ARGS)
    length(args) == 1 || error(
        "Usage: julia --project=. bound_hsc/verify_proposition_5_6_checkpoint.jl " *
        "<checkpoint-directory-or-step6.jls>",
    )
    supplied_path = args[1]
    step6_path = isdir(supplied_path) ? joinpath(supplied_path, "step6.jls") : supplied_path
    verify_proposition_5_6_checkpoint(step6_path)
    println("\nProposition 5.6 certified successfully.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    Base.invokelatest(main)
end
