Run

julia --project=. eigenvalues/mini-demo/assemble_matrices.jl

to create mass_matrix.mat and stiff_matrix.mat.

On Mac, run

env BLAS_VERSION=/Applications/MATLAB_R2026a.app/bin/maca64/libmwopenblas.dylib \
  matlab -batch "run('eigenvalues/mini-demo/compute_eigenvalues.m')"

to compute eigenvalues of this triangulation.

Output:

Certified FEM eigenvalue intervals:
    1.0000   -0.0000    0.0000
    2.0000    9.8893    9.8893
    3.0000    9.8916    9.8916
    4.0000   19.7932   19.7932
    5.0000   39.7950   39.7950
    6.0000   39.8318   39.8318
    7.0000   49.7355   49.7355
    8.0000   49.7805   49.7805
    9.0000   79.8242   79.8242
   10.0000   90.4377   90.4377
Continuum values with same indices, for comparison only:
    1.0000         0
    2.0000    9.8696
    3.0000    9.8696
    4.0000   19.7392
    5.0000   39.4784
    6.0000   39.4784
    7.0000   49.3480
    8.0000   49.3480
    9.0000   78.9568
   10.0000   88.8264