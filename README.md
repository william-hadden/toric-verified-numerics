# Toric Verified Numerics

DESC (TODO)

# Setup

This repository uses a local Julia project environment.

## Requirements

- Julia 1.10 or newer

## Install dependencies

From the repository root, run:

```bash
julia --project=. scripts/setup_julia_env.jl
```

This installs the required packages declared in `Project.toml`, including:

- `IntervalArithmetic`
- `FFTW`

## Data

Download the file `coeffs-rational.csv.zip` from <https://drive.google.com/file/d/1lLDZO1lSbs4pZRi_1aa1WtDz13-kLCH4/view?usp=sharing> and put it in the folder `data/happrox`.

Extract the archive in place before running the residual-bound code.
The active data file is the extracted `data/happrox/coeffs-rational.csv`.

## Running scripts

Use:

```bash
julia --project=. path/to/script.jl
```

so scripts run inside the project environment.

## Code layout

- `Chebyshev/` contains the shared tensor-product Chebyshev algebra,
  differentiation, interval/local-box evaluation, and rigorous truncation
  helpers.
- `utils/` contains shared data I/O, interval construction, progress reporting,
  and `load_common.jl`, the common include entry point used by executable scripts.
- `bound_metric/` contains only inverse-metric and curvature certificate code.
- `bound_residual/` contains only the Monge–Ampère residual computation.

Scripts that need the shared numerical helpers should include
`utils/load_common.jl` rather than assembling their own list of utility files.

## Tests

Run the full test suite with:

```bash
julia --project=. test/runtests.jl
```

The test runner also accepts a filter argument for focused runs:

```bash
julia --project=. test/runtests.jl quotient
julia --project=. test/runtests.jl ricci
julia --project=. test/runtests.jl inverse
julia --project=. test/runtests.jl cheb
julia --project=. test/runtests.jl utils
```

- `quotient` checks the coefficient-space quotient differentiation formula in cases where `D = 1`.
- `ricci` checks Ricci assembly against a polynomial inverse metric with known exact answer.
- `inverse` runs a small rigorous end-to-end inverse/Ricci certificate on embedded test data.
- `cheb` runs the Chebyshev multiplication tests.
- `utils` checks shared CSV/JSON I/O, interval helpers, evaluation, and truncation.

## TODO
