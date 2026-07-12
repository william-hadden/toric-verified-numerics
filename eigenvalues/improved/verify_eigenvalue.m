function verify_eigenvalue(run_dir)
% Verify the first positive generalized FEM eigenvalue for one improved run.

script_dir = fileparts(mfilename('fullpath'));
eigen_dir = fileparts(script_dir);

addpath(fullfile(eigen_dir, 'veigs-main'));
addpath(fullfile(eigen_dir, 'INTLAB', 'Intlab_V14.1'));

old_figure_visible = get(0, 'DefaultFigureVisible');
restore_figure_visible = onCleanup(@() set(0, 'DefaultFigureVisible', old_figure_visible));
set(0, 'DefaultFigureVisible', 'off');
evalc('startintlab');
set(0, 'DefaultFigureVisible', old_figure_visible);
clear restore_figure_visible

evalc('intvalinit(''DisplayInfsup'')');

S = load(fullfile(run_dir, 'stiff_matrix.mat'));
K = infsup(sparse(S.i, S.j, S.lo, S.n, S.n), ...
           sparse(S.i, S.j, S.hi, S.n, S.n));

S = load(fullfile(run_dir, 'mass_matrix.mat'));
M = infsup(sparse(S.i, S.j, S.lo, S.n, S.n), ...
           sparse(S.i, S.j, S.hi, S.n, S.n));

tic
[lambda, ind] = veigs(K, M, 2, 'sa');
elapsed = toc;

% The D6-quotient CR matrices, before imposing the mean-zero constraint,
% contain the constant zero mode.  The second generalized eigenvalue is
% therefore the first positive eigenvalue on the mean-zero quotient space.
target = find(ind == 2);
if numel(target) ~= 1
    error('VEIGS did not return a unique enclosure indexed by eigenvalue 2.');
end
lambda_fem_lb = inf(lambda(target));
lambda_fem_ub = sup(lambda(target));
lambda_fem_ind = ind(target);

save(fullfile(run_dir, 'verified_eigenvalue.mat'), ...
     'lambda_fem_lb', 'lambda_fem_ub', 'lambda_fem_ind', 'elapsed', '-v4')

fprintf('lambda_FEM(2) = [%.17g, %.17g]\n', lambda_fem_lb, lambda_fem_ub);
fprintf('veigs elapsed seconds = %.6f\n', elapsed);
end
