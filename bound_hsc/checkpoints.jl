using Serialization
using SHA
using IntervalArithmetic

isdefined(@__MODULE__, :PolytopeBox) || include(joinpath(@__DIR__, "util.jl"))

hsc_checkpoint_environment() = (
    julia_version = string(VERSION),
    interval_arithmetic_version = string(Base.pkgversion(IntervalArithmetic)),
)

const DEFAULT_HSC_CHECKPOINT_DIR = joinpath(@__DIR__, "checkpoints")
const HSC_CHECKPOINT_FORMAT_VERSION = 1

function hsc_run_checkpoint_dir(
    parameters;
    checkpoint_root::AbstractString = DEFAULT_HSC_CHECKPOINT_DIR,
)
    digest = bytes2hex(sha256(codeunits(repr(parameters))))[1:12]
    folder_name = "run_pdeg-$(parameters.pdeg)_grid-$(parameters.grid)_" *
                  "directions-$(parameters.direction_order)_" *
                  "depth-$(parameters.screen_depth)-$(parameters.maxdepth)_$digest"
    checkpoint_dir = joinpath(checkpoint_root, folder_name)
    mkpath(checkpoint_dir)

    readme_path = joinpath(checkpoint_dir, "README.txt")
    open(readme_path, "w") do io
        println(io, "HSC checkpoint run")
        println(io, "==================")
        println(io)
        println(io, "This folder contains the parameter-dependent checkpoints for one HSC run.")
        println(io, "Steps 2 and 3 are shared, parameter-independent checkpoints stored in the parent folder.")
        println(io)
        println(io, "Run parameters")
        println(io, "--------------")
        for (name, value) in pairs(parameters)
            println(io, name, " = ", value)
        end
    end

    return checkpoint_dir
end

"""Identify the coefficient data and arithmetic precision used by an HSC run."""
function hsc_checkpoint_context(coeffs_path::AbstractString)
    resolved_path = realpath(coeffs_path)
    digest = open(sha256, resolved_path)
    return (
        coefficient_sha256 = bytes2hex(digest),
        coefficient_size = filesize(resolved_path),
        bigfloat_precision = precision(BigFloat),
        coefficient_format = :rational_csv,
    )
end

function hsc_checkpoint_path(
    step::Integer;
    checkpoint_dir::AbstractString = DEFAULT_HSC_CHECKPOINT_DIR,
)
    step in 2:6 || throw(ArgumentError(
        "HSC checkpoints are supported for steps 2 through 6",
    ))
    return joinpath(checkpoint_dir, "step$(step).jls")
end

has_hsc_checkpoint(
    step::Integer;
    checkpoint_dir::AbstractString = DEFAULT_HSC_CHECKPOINT_DIR,
) = isfile(hsc_checkpoint_path(step; checkpoint_dir))

"""
    save_hsc_checkpoint(step, result; context, parameters,
                        checkpoint_dir=DEFAULT_HSC_CHECKPOINT_DIR)

Serialize the result of HSC step 2 through 6 to `checkpoint_dir/stepN.jls`.
The checkpoint directory is created if necessary. Returns the saved file path.
"""
function save_hsc_checkpoint(
    step::Integer,
    result;
    context,
    parameters = NamedTuple(),
    checkpoint_dir::AbstractString = DEFAULT_HSC_CHECKPOINT_DIR,
)
    mkpath(checkpoint_dir)
    path = hsc_checkpoint_path(step; checkpoint_dir)
    checkpoint = (
        format_version = HSC_CHECKPOINT_FORMAT_VERSION,
        step = Int(step),
        context,
        parameters,
        environment = hsc_checkpoint_environment(),
        result,
    )

    temporary_path = tempname(checkpoint_dir)
    try
        open(temporary_path, "w") do io
            serialize(io, checkpoint)
        end
        mv(temporary_path, path; force = true)
    finally
        isfile(temporary_path) && rm(temporary_path; force = true)
    end
    return path
end

"""
    read_checkpoint_envelope(path, step)

Read and validate a checkpoint envelope, reporting environment differences.
Does not change the current arithmetic precision.
"""
function read_checkpoint_envelope(path::AbstractString, step::Integer)
    isfile(path) || throw(ArgumentError("HSC checkpoint does not exist: $path"))
    checkpoint = try
        open(deserialize, path)
    catch error
        throw(ArgumentError(
            "Unable to deserialize HSC checkpoint $path. " *
            "Current environment: $(hsc_checkpoint_environment()). " *
            "Use the writer's Julia version, Manifest.toml, and repository revision. " *
            "The saved environment may be unavailable because deserialization failed. " *
            "Original error: $(sprint(showerror, error))",
        ))
    end

    valid_envelope = checkpoint isa NamedTuple &&
                     all(key -> haskey(checkpoint, key),
                         (:format_version, :step, :context, :parameters, :result))
    valid_envelope || throw(ArgumentError(
        "Legacy or invalid HSC checkpoint $path",
    ))
    checkpoint.format_version == HSC_CHECKPOINT_FORMAT_VERSION || throw(ArgumentError(
        "Unsupported HSC checkpoint format in $path",
    ))
    checkpoint.step == step || throw(ArgumentError(
        "HSC checkpoint step mismatch in $path",
    ))
    if haskey(checkpoint, :environment) &&
       !isequal(checkpoint.environment, hsc_checkpoint_environment())
        println(stderr, "HSC checkpoint environment differs: saved ",
                checkpoint.environment, "; current ", hsc_checkpoint_environment(),
                ". Use matching Julia/package versions if compatibility problems occur.")
    end
    return checkpoint
end

"""
Load a saved HSC result, with its required types already defined by this file.
For interactive loads, restore the saved global BigFloat precision. Pipeline
loads supplying expected_context retain and validate their current precision.
Older checkpoints without environment metadata remain readable.
"""
function load_hsc_checkpoint(
    step::Integer;
    expected_context = nothing,
    expected_parameters = nothing,
    checkpoint_dir::AbstractString = DEFAULT_HSC_CHECKPOINT_DIR,
    restore_precision::Bool = expected_context === nothing,
)
    path = hsc_checkpoint_path(step; checkpoint_dir)
    checkpoint = read_checkpoint_envelope(path, step)
    if expected_context !== nothing && !isequal(checkpoint.context, expected_context)
        throw(ArgumentError(
            "HSC checkpoint $path was produced from different coefficients or " *
            "BigFloat precision",
        ))
    end
    if expected_parameters !== nothing && !isequal(checkpoint.parameters, expected_parameters)
        throw(ArgumentError(
            "HSC checkpoint $path was produced with different step parameters",
        ))
    end
    restore_precision && setprecision(BigFloat, checkpoint.context.bigfloat_precision)
    return checkpoint.result
end

"""
Load a compatible checkpoint when one exists. Invalid, legacy, or stale
checkpoints are reported and treated as cache misses so the pipeline can
recompute and replace them.
"""
function load_hsc_checkpoint_if_valid(
    step::Integer;
    expected_context,
    expected_parameters,
    checkpoint_dir::AbstractString = DEFAULT_HSC_CHECKPOINT_DIR,
)
    has_hsc_checkpoint(step; checkpoint_dir) || return nothing

    try
        return load_hsc_checkpoint(
            step;
            expected_context,
            expected_parameters,
            checkpoint_dir,
        )
    catch error
        error isa ArgumentError || rethrow()
        println("Ignoring unusable HSC checkpoint: ", sprint(showerror, error))
        return nothing
    end
end
