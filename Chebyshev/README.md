# Shared Chebyshev utilities

This directory implements the tensor-product Chebyshev operations used by the
verified residual, inverse-metric, curvature, and eigenvalue computations.  The
files are collections of Julia functions rather than a standalone module.
Executable scripts normally load them, together with the required interval and
I/O helpers, through `utils/load_common.jl`.

For an exported coefficient matrix `a`, the project convention on
\((x,y)\in[0,1]^2\) is

\[
u(x,y)=\sum_{i,j} a_{i+1,j+1}
T_i(2x-1)\,T_j(1-2y).
\]

Thus the first array index is the \(x\) mode and the second is the \(y\) mode.
The reversed affine coordinate in the \(y\) factor accounts for the minus sign
in `differentiate_coeffs_y` and in the Lobatto \(y\)-differentiation matrix.

## Files and recommended review order

1. **`algebra.jl` — core representation and transforms**

   Read this first.  It defines the basic coefficient-array operations:
   constants, zeros, padding, trimming, addition, subtraction, scaling, and
   multiplication.  It also implements the one- and two-dimensional DCT-I
   transforms between Chebyshev coefficients and values on tensor-product
   Lobatto grids.

   There are three multiplication paths:

   - `:direct` applies the Chebyshev product identity in coefficient space;
   - `:dct` is the fast floating-point Lobatto-grid/DCT calculation;
   - `:interval_dct` performs the transforms with cached interval matrices and
     outward-rounded interval arithmetic.

   `cheb_mul_fast` is the main rigorous multiplication wrapper used elsewhere
   in the repository.  Near the end of the file are coefficient-sum sup-norm
   bounds, Chebyshev differentiation matrices, and routines that assemble
   first-, second-, and third-derivative values on a Lobatto grid.

2. **`differentiation.jl` — coefficient-space derivatives**

   This implements the Chebyshev coefficient recurrence for differentiation,
   maps it to the project’s physical \(x\) and \(y\) coordinates, and assembles
   the coefficient arrays for \(u_0\), \(u_x\), \(u_y\), \(u_{xx}\),
   \(u_{xy}\), and \(u_{yy}\).  These routines are used when later calculations
   need derivative series rather than only derivative values on a grid.

3. **`truncation.jl` — rigorous finite-degree reduction**

   This file bounds discarded modes by summing their absolute interval
   coefficients.  It can absorb a tail into the constant mode or return a
   retained coefficient block together with a separate tail bound.  The latter
   form, produced by `truncate_coeffs_with_tail`, is used by the local
   subdivision arguments in the metric and eigenvalue computations.

4. **`interval_evaluation.jl` — point, box, and local evaluation**

   Read this last because it builds on the coefficient conventions and
   interval helpers used above.  It contains:

   - recurrence evaluation of \(T_0,\ldots,T_N\);
   - evaluation at points in the physical square \([0,1]^2\);
   - construction of Chebyshev grids and interval subdivisions;
   - direct evaluation on interval boxes;
   - conversion to power coefficients on a local box;
   - centered coefficient-sum enclosures for subdivision proofs.

   The local-box routines use native Chebyshev coordinates in \([-1,1]^2\);
   `evaluate_coeffs_at_point` is the routine that explicitly applies the
   project’s affine map from physical coordinates.

## How the files are loaded

`utils/load_common.jl` first loads the shared I/O, interval, and progress
helpers, then includes these files in the following dependency-safe order:

```text
algebra.jl
differentiation.jl
truncation.jl
interval_evaluation.jl
```

Loading through that common entry point is preferred to including an
individual file, because functions here rely on `IntervalArithmetic` types and
shared helpers such as `intervalize_coefficients`, `exact`, and
`symmetric_interval`.

## Tests

From the repository root, the focused Chebyshev test set can be run with:

```bash
julia --project=. test/runtests.jl cheb
```

It compares direct and DCT multiplication, checks rigorous interval
enclosures, exercises known Chebyshev identities and `BigFloat` intervals, and
includes timing and full-data cases.  The lighter shared-utility tests cover
basic evaluation and truncation:

```bash
julia --project=. test/runtests.jl utils
```
