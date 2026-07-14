# Load shared helpers once when executable scripts are included into the same module.
isdefined(@__MODULE__, :REPO_ROOT) || include(joinpath(@__DIR__, "io.jl"))
isdefined(@__MODULE__, :intervalize_coefficients) || include(joinpath(@__DIR__, "intervals.jl"))
isdefined(@__MODULE__, :TextProgress) || include(joinpath(@__DIR__, "progress.jl"))

chebyshev_dir = normpath(joinpath(@__DIR__, "..", "Chebyshev"))
isdefined(@__MODULE__, :_CHEB_HAS_FFTW) || include(joinpath(chebyshev_dir, "algebra.jl"))
isdefined(@__MODULE__, :cheb_diff_1d) || include(joinpath(chebyshev_dir, "differentiation.jl"))
isdefined(@__MODULE__, :chebyshev_tensor_tail_bound) || include(joinpath(chebyshev_dir, "truncation.jl"))
isdefined(@__MODULE__, :chebyshev_values) || include(joinpath(chebyshev_dir, "interval_evaluation.jl"))
