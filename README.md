# Toric Verified Numerics

In this repository, we use a combination we produce an approximate Kähler-Einstein metric 𝑔 and also prove that the true metric 𝑔KE is close to 𝑔. As an application, we provide bounds on the first invariant eigenvalue of the Laplace-Beltrami operator. 

# Setup

This repository uses a combination of Mathematica to compute the approximate metric, Julia to compute the bounds of this metric and it's curvature and compute the bound of the Monge-Ampere residual, and MATLAB to compute the corresponding approximate eigenvalues.

## Requirements

- Mathematica 
- Julia 1.10 or newer
- MATLAB

## Install dependencies

From the repository root, run:

```bash
julia --project=. scripts/setup_julia_env.jl
```

This installs the required packages declared in `Project.toml`, including:

- `IntervalArithmetic`
- `FFTW`

# Data

Download the file `coeffs-rational.csv.zip` from <https://drive.google.com/file/d/1MeSFcmyTjvK3429lprgXJMsYQt1hi0ol/view?usp=sharing> and put it in the folder `data/happrox`.

Extract the archive in place before running the residual-bound code.
The active data file is the extracted `data/happrox/coeffs-rational.csv`.

# Code layout

- `metric_approx/` contains the notebook used to compute the approximate metric.
- `Chebyshev/` contains the shared tensor-product Chebyshev algebra,
  differentiation, interval/local-box evaluation, and rigorous truncation
  helpers.
- `utils/` contains shared data I/O, interval construction, progress reporting,
  and `load_common.jl`, the common include entry point used by executable scripts.
- `bound_metric/` contains the inverse-metric and curvature certificate code.
- `bound_residual/` contains the Monge–Ampère residual computation.
- `eigenvalues/` contains the Laplace-Beltrami eigenvalues computation.

# Running scripts

Use:

```bash
julia --project=. path/to/script.jl
```

so scripts run inside the project environment.

## Running order

If checkpoints:

bound_metric/bound_inverse -> bound_metric/bound_riem ->
bound_metric/bound_ricci -> bound_residual/bound_residual
->eigenvalues/orchestrate.jl -> apply_fixed_point/apply_fixed_point ->
bound_hsc/verify_proposition_5_6.jl -> eigenvalue_comparison/upper_bound.jl ->
eigenvalue_comparison/eigenvalue_for_true_KE_metric

If not HSC checkpoints:
bound_metric/bound_inverse -> bound_metric/bound_riem ->
bound_metric/bound_ricci -> bound_residual/bound_residual
->eigenvalues/orchestrate.jl -> apply_fixed_point/apply_fixed_point ->
bound_hsc/find_negative_curvature_region -> bound_hsc/verify_proposition_5_6 -> eigenvalue_comparison/upper_bound.jl -> eigenvalue_comparison/eigenvalue_for_true_KE_metric

The following powershell command runs the pipeline and keeps the process alive:
```powershell
$repo = "<path-to-code>\toric-verified-numerics"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

$p = Start-Process powershell.exe -WindowStyle Hidden -PassThru `
    -WorkingDirectory $repo `
    -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File ".\<run-pipeline>.ps1"' `
    -RedirectStandardOutput "$repo\pipeline_after_orchestrate_$stamp.log" `
    -RedirectStandardError "$repo\pipeline_after_orchestrate_$stamp.err.log"

$p.Id
Get-Content "$repo\pipeline_after_orchestrate_$stamp.log" -Wait -Tail 20
```

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
