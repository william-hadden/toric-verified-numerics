# Lightweight terminal progress reporting shared by long-running bounds.
mutable struct TextProgress
    label::String
    total::Int
    current::Int
    width::Int
    started_at::UInt64
    finished_at::Union{UInt64, Nothing}
end

function start_progress(label::AbstractString, total::Integer)
    progress = TextProgress(String(label), Int(total), 0, 30, time_ns(), nothing)
    render_progress(progress)
    return progress
end

elapsed_seconds(progress::TextProgress) =
    ((progress.finished_at === nothing ? time_ns() : progress.finished_at) - progress.started_at) / 1.0e9

function format_elapsed(seconds::Real)
    seconds < 60 && return "$(round(seconds; digits = 2))s"

    minutes = floor(Int, seconds / 60)
    remaining_seconds = seconds - 60 * minutes
    return "$(minutes)m $(round(remaining_seconds; digits = 2))s"
end

function render_progress(progress::TextProgress)
    filled = progress.total == 0 ? progress.width : fld(progress.current * progress.width, progress.total)
    percent = progress.total == 0 ? 100 : fld(progress.current * 100, progress.total)
    bar = repeat("#", filled) * repeat("-", progress.width - filled)
    elapsed = format_elapsed(elapsed_seconds(progress))
    print("\r$(progress.label) [$bar] $(percent)% ($(progress.current)/$(progress.total)) elapsed: $elapsed")
    flush(stdout)
end

function advance_progress!(progress::Union{TextProgress, Nothing}, n::Integer = 1)
    progress === nothing && return nothing
    progress.current = min(progress.current + Int(n), progress.total)
    render_progress(progress)
    return nothing
end

function finish_progress!(progress::Union{TextProgress, Nothing})
    progress === nothing && return nothing
    progress.current = progress.total
    progress.finished_at = time_ns()
    render_progress(progress)
    println()
    return nothing
end

function progress_message!(progress::Union{TextProgress, Nothing}, message::AbstractString)
    progress === nothing && return nothing
    println()
    println(message)
    render_progress(progress)
    return nothing
end
