# eigenvalues folder

## Summary

This carries out the calculations for Section 5.2 "first eigenvalue".
It computes a rigorous eigenvalue enclosure for the first D6xT^2-invariant eigenvalue w.r.t. the true KE metric.
It does this in two steps:

1. Compute a rigorous upper bound for the invariant eigenvalue w.r.t. the approximate KE metric. (A rigorous lower bound was computed earlier.)
2. Use an upper bound for the C^0-difference between true and approximate KE metric to transform upper and lower bounds to bounds for the true KE metric.


## How to run this

- `upper_bound.jl`: computes an upper bound for the invariant eigenvalue w.r.t. the approximate KE metric and writes it to the shared json file under the key `lambda_1_upper_bound` (see Latex `approxmineigenvalue` for an explanation of what's being computed).
- `eigenvalue_for_true_KE_metric.jl`: use `proposition:eigenvalue-comparison` from Latex to get bounds for the true KE metric.


## How to review this

- `upper_bound.jl`: computes the Rayleigh quotient on the square `(x,s) in [0,1]^2` where `s=-y` relative to the physical coordinates `(x,y) in [0,1]x[-1,0]`. The Chebyshev variables are `(xcheb,scheb)=(2x-1,1-2s) in [-1,1]^2`. The test function is `f=x^2+xy+y^2-5/12=x^2-xs+s^2-5/12`. On the square it has mean zero and squared L^2-norm 43/720. The square contains two of the hexagons six fundamental sectors, but a factor of 3 in numerator and denominator would cancel in the Rayleigh quotient and is therefore not implemented. Same for the factor 4pi^2 from the torus volume. The volume form is metric-independent, which the numerator depends in the approximate KE metric when measuring `grad(f)`.
- `eigenvalue_for_true_KE_metric.jl`: a short script that multiplies eigenvalue bounds with the metric difference as appropriate.