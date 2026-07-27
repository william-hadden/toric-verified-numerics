# Verified Monge–Ampère residual bounds

This directory contains the interval-arithmetic computation that measures how
far the approximate toric Kähler–Einstein metric is from satisfying the
Monge–Ampère equation.  It produces rigorous \(C^0\) and Sobolev bounds for the
residual and records them in `data/verified_bounds.json`.

The computation uses the tensor-product Chebyshev representation of the
approximate symplectic potential \(u_0\).  On the polytope coordinates
\((x,y)\in[0,1]^2\), it writes the equation in the form

\[
\operatorname{MA}(u_0)=G\,e^H,
\qquad
H=2u_0-2(xu_{0,x}+yu_{0,y}),
\]

where \(G\) is the polynomial expression in the second derivatives of \(u_0\)
implemented in `eqn_coefficients_at_point`.  An exact solution, after fixing
the additive normalization, would have
\(\operatorname{MA}(u_0)=1\).  The residual bounded here is therefore

\[
E=G\,e^H-1.
\]

## Files and recommended review order

1. **`bound_residual.jl` — entry point and high-level pipeline**

   Start here.  It loads `IntervalArithmetic`, includes the shared utilities
   and the problem-specific implementation, and defines
   `compute_bound_residual`.  That function presents the calculation in its
   execution order:

   - load and normalize \(u_0\);
   - form \(G\) and \(H\) on a Chebyshev–Lobatto grid;
   - rigorously bound the first three derivatives of \(G e^H\);
   - combine the first partial-derivative bounds;
   - use the mean value theorem to bound
     \(\lVert G e^H-1\rVert_{C^0}\);
   - convert the residual and derivative estimates into
     \(L^2\), \(H^1\), \(H^2\), and \(H^3\)-type bounds.

   The final `main` function sets 200-bit `BigFloat` precision and runs the
   default calculation with a \(240\times240\) Lobatto grid.

2. **`util/problem.jl` — mathematical implementation**

   Read this second, following the functions in their file order.

   - `eqn_coefficients_at_point` implements the coefficient functions in the
     scalar equation.
   - `evaluate_raw_F_at_reference` estimates the additive normalisation from one
     interior collocation point.
   - `step1_load_and_normalize_u0` shifts the constant Chebyshev mode by the 
     additive normalisation.
   - `step2_compute_GH_values_grid` pads the coefficient array, differentiates
     on the Lobatto grid, and evaluates \(G\) and \(H\).
   - `step3_compute_MA_derivatives` differentiates the factored expression
     \(G e^H\) through third order.  Each derivative is transformed back to
     Chebyshev coefficient space and bounded by the sum of the absolute
     coefficient intervals.
   - `step4_compute_MA_derivative_bound` combines the \(x\)- and
     \(y\)-derivative bounds.
   - `step5_compute_residual_bound_by_MVT` evaluates at
     \((1/2,1/2)\) and uses the derivative bound and the maximum distance to a
     corner of \([0,1]^2\) to obtain the global residual bound via MVT.
   - `step_nabla`, `step_nabla2`, and `step_nabla3` use the certified inverse
     metric and curvature bounds to obtain the Sobolev estimates.  The latter
     two use the Laplacian formula and the project’s Bochner/commutator
     estimates.

## Inputs, dependencies, and outputs

The executable loads the rational Chebyshev coefficients from
`data/happrox/coeffs-rational.csv`.  Shared CSV/JSON helpers and all Chebyshev
operations are brought in through `utils/load_common.jl`.

The covariant-derivative stages also read the following previously certified
entries from `data/verified_bounds.json`:

- `metric_inverse`, including its first and second derivatives;
- `curvature_bounds.ricci_C0`;
- `curvature_bounds.riemann_C0`;
- `curvature_bounds.nabla_riemann_C0`.

Consequently, the inverse-metric and curvature bounds must already be present
before running the complete residual pipeline.  The computation updates these
groups in `data/verified_bounds.json`:

- `ma_first_derivatives`;
- `ma_derivatives`;
- `ma_residual_bounds`;
- `ma_sobolev_bounds`.

## Running the computation

From the repository root, install the Julia dependencies as described in the
top-level README, then run:

```bash
julia --project=. bound_residual/bound_residual.jl
```

The rational input file is large, and the default interval transforms use a
\(240\times240\) grid, so this is a substantial verified computation rather
than a quick smoke test.  For programmatic experiments, include the entry-point
file and call `compute_bound_residual(path; pdeg=...)`; smaller `pdeg` values
are useful for development but do not reproduce the default certificate.

