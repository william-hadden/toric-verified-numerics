dir = fileparts(mfilename('fullpath'));
addpath(fullfile(dir, '..', 'veigs-main'));
addpath(fullfile(dir, '..', 'INTLAB', 'Intlab_V14.1'));

oldFigureVisible = get(0, 'DefaultFigureVisible');
restoreFigureVisible = onCleanup(@() set(0, 'DefaultFigureVisible', oldFigureVisible));
set(0, 'DefaultFigureVisible', 'off');
startintlab;
set(0, 'DefaultFigureVisible', oldFigureVisible);
clear restoreFigureVisible

intvalinit('DisplayInfsup')

S = load(fullfile(dir, 'stiff_matrix.mat'));
K = infsup(sparse(S.i,S.j,S.lo,S.n,S.n), sparse(S.i,S.j,S.hi,S.n,S.n));

S = load(fullfile(dir, 'mass_matrix.mat'));
M = infsup(sparse(S.i,S.j,S.lo,S.n,S.n), sparse(S.i,S.j,S.hi,S.n,S.n));

k = 10;
[lambda, ind] = veigs(K, M, k, 'sa');

disp('Certified FEM eigenvalue intervals:')
disp([ind(:), inf(lambda(:)), sup(lambda(:))])

[m,n] = ndgrid(0:10,0:10);
exact = pi^2 * sort(m(:).^2 + n(:).^2);

disp('Continuum values with same indices, for comparison only:')
disp([ind(:), exact(ind(:))])
