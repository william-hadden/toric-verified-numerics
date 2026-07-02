dir = fileparts(mfilename('fullpath'));

addpath(fullfile(dir, 'veigs-main'));
addpath(fullfile(dir, 'INTLAB', 'Intlab_V14.1'));

oldFigureVisible = get(0, 'DefaultFigureVisible');
restoreFigureVisible = onCleanup(@() set(0, 'DefaultFigureVisible', oldFigureVisible));
set(0, 'DefaultFigureVisible', 'off');
startintlab;
set(0, 'DefaultFigureVisible', oldFigureVisible);
clear restoreFigureVisible

intvalinit('DisplayInfsup')
format long g

S = load(fullfile(dir, 'stiff_matrix.mat'));
stiff_nnz = numel(S.i);
K = infsup(sparse(S.i, S.j, S.lo, S.n, S.n), sparse(S.i, S.j, S.hi, S.n, S.n));

S = load(fullfile(dir, 'mass_matrix.mat'));
mass_nnz = numel(S.i);
M = infsup(sparse(S.i, S.j, S.lo, S.n, S.n), sparse(S.i, S.j, S.hi, S.n, S.n));

disp('Matrix sizes:')
disp(size(K))
disp('Stored entries:')
disp([stiff_nnz, mass_nnz])

k = 10;
tic
[lambda, ind] = veigs(K, M, k, 'sa');
elapsed = toc;

disp('veigs runtime in seconds:')
disp(elapsed)

disp('Certified D6-invariant FEM eigenvalue intervals:')
disp([ind(:), inf(lambda(:)), sup(lambda(:))])
