# eigenvalues folder

## Summary

This carries out the calculations for Section 3.5 "a non-sharp first eigenvalue".
It computes a rigorous lower bound for the first non-zero D6xT^2-invariant eigenvalue with respect to the approximate Kähler-Einstein metric.
The T^2-invariance reduces this to computations on the hexagon Q.
The D6-invariance further reduces this to computations on the fundamental sector P.
It does this in three steps:

1. On an inset fundamental sector P_delta, compute a smallest non-zero FEM eigenvalue using INTLAB.
2. Use a result by Xuefeng Liu to relate this to the smallest non-zero eigenvalue for smooth functions on P_delta.
3. Use a theorem from our paper to relate the eigenvalue of P_delta to the eigenvalue on P (which is equal to the eigenvalue on the smooth, closed 4D manifold).


## How to run this

- `python3 eigenvalues/triangulation/check_triangulation.py` checks the correctness of `triangulation/triangulation.mat`
- `julia --threads=4 --project=. eigenvalues/orchestrate.jl` produces mass and (a lower bound for) the stiffness matrices `eigenvalues/mass_matrix.mat` and `eigenvalues/stiff_matrix.mat` for the inset fundamental sector with respect to the approximate KE metric
- In MATLAB run `run('eigenvalues/verify_eigenvalue.m')` uses INTLAB and `veigs` to certify the first non-constant FEM eigenvalue and writes it to `eigenvalues/verified_eigenvalue.mat`
- `julia --project=. eigenvalues/update_bound.jl` applies steps 2 and 3 from above and writes `lambda_1_delta_lower_bound` and `lambda_1_lower_bound` to `data/verified_bounds.json`. The latter is the verified eigenvalue bound on the smooth, closed 4D manifold.


## How to review this

- `check_triangulation.py`: `check_lattice_data` checks basic properties of the triangulation; `check_exact_areas` checks that the triangulation has area equal to the fundamental sector; `check_geometric_conformity` checks that there are no overlapping triangles. The last two checks together prove that the triangulation is a triangulation of the fundamental sector without gaps. Caveat: all triangulation lattice points are integers. They are later scaled down into the range x in [0,1-delta] (y accordingly) to correspond to a triangulation of P_delta.
- `orchestrate.jl`: this is basic in/out and fixes parameters such as delta and truncation degree in the method `run_orchestration`.
- `assemble_matrices.jl`: when arguments `nodes` and `tri` appear, then `nodes` is the full list of vertices of the triangulation as Interval{BigFloat} (rescaled from the earlier integers) and `tri` is a triple of indices defining one triangle. `local_cr_mass_matrix` and `local_cr_stiffness_matrix` give the 3x3 mass and stiffness matrices for one fixed triangle (there are three elements that are non-zero on any given triangle). Computing them is simple, because elements have constants gradients per triangle. `build_cr_edges` creates a list of all edges. For CR FEM, one edge=one degree of freedom. `d6_edge_quotient` identifies actual reflected mesh edges across the sector bisector. It uses the smaller of the sorted endpoint pair and its reflected sorted endpoint pair as the orbit key; unmatched edges remain independent. This does not require a reflection-invariant triangulation. The returned vector records this quotient index for every edge in the unique edge list. `local_liu_constant` is the per-triangle version of `equation:liu-constant` from the paper. `certify_element` gives for each triangle a lower bound for the bilinear form G^{ij} (called B) and a smallest eigenvalue bound alpha. `assemble_matrices` constructs the mass and stiffness matrices: the lines `add_block!(mass, dofs, local_cr_mass_matrix(mesh.nodes, tri))` and `add_block!(stiffness, dofs, local_stiffness)` add the local (triangle-wise) mass and stiffness matrices to the global matrices.
- `inverse_metric_bounds.jl`: the method `inverse_metric_lower_bound` is called by the previous file. It gives a "Loewner" lower bound B for G^{ij} on a triangle, i.e. vBv^T < vG^{ij}v^T for all vectors v not 0. It obtains this bound by interval-enclosing G^{ij} on the triangle (that's `metric_matrix_hull(leaves)`), then computing its interval-midpoint `midpoint`, and then attempts to find a Loewner lower bound of the form B=theta*midpoint for theta<=1 using the method `search_lower_matrix_scaling`.
- `verify_eigenvalue.m`: apart from i/o and blocking a startup animation to enable non-interactive use, the only lines of code are: the lines starting with `hull` making the matrices symmetric, calling the verified eigenvalue computation `veigs(stiffness, mass, 1, eigenvalue_target)` and then selecting the eigenvalue with index 2 (Matlab starts counting at 1, and the first eigenfunction is the constant). `eigenvalue_target = 5` is a numerical hint for Matlab.
- `update_bound.jl`: loads the lower bound provided by Matlab; the method `liu_lower_bound` applies `theorem:liu-theorem` from the paper to get a lower bound on smooth functions on the inset hexagon; `compact_lower_bound` uses `theorem:eigenvalue-inset-comparison` from the paper to obtain a bound on the 4D manifold. (Some arithmetic is used to explain the theorem, which is explained in the docstring of `compact_lower_bound`.)
- `check_triangulation.jl`: this constructs one fixed triangulation. This generating file need not be reviewed, because the properties of the triangulation are checked by `check_triangulation.py` instead.