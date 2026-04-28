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

## Running scripts

Use:

```bash
julia --project=. path/to/script.jl
```

so scripts run inside the project environment.

## Tests

Run tests with:

```bash
julia --project=. test/runtests.jl
```

## TODO
