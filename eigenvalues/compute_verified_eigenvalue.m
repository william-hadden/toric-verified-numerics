dir = fileparts(mfilename('fullpath'));

addpath(fullfile(dir, 'veigs-main'));
addpath(fullfile(dir, 'INTLAB', 'Intlab_V14.1'));

oldFigureVisible = get(0, 'DefaultFigureVisible');
restoreFigureVisible = onCleanup(@() set(0, 'DefaultFigureVisible', oldFigureVisible));
set(0, 'DefaultFigureVisible', 'off');
evalc('startintlab');
set(0, 'DefaultFigureVisible', oldFigureVisible);
clear restoreFigureVisible

evalc('intvalinit(''DisplayInfsup'')');

S = load(fullfile(dir, 'stiff_matrix.mat'));
K = infsup(sparse(S.i, S.j, S.lo, S.n, S.n), sparse(S.i, S.j, S.hi, S.n, S.n));

S = load(fullfile(dir, 'mass_matrix.mat'));
M = infsup(sparse(S.i, S.j, S.lo, S.n, S.n), sparse(S.i, S.j, S.hi, S.n, S.n));

[lambda, ~] = veigs(K, M, 2, 'sa');

lambda_lb = inf(lambda(2));

save(fullfile(dir, 'verified_eigenvalue.mat'), 'lambda_lb', '-v4')
