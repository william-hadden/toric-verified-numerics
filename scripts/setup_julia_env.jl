import Pkg

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

Pkg.activate(REPO_ROOT)
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()

println("Julia environment ready at: " * REPO_ROOT)
println("You can now run:")
println("  julia --project=. test/runtests.jl")
