% Verify the first positive generalized FEM eigenvalue.

directory = fileparts(mfilename('fullpath'));
addpath(fullfile(directory, 'veigs-main'));
addpath(fullfile(directory, 'INTLAB', 'Intlab_V14.1'));

old_figure_visible = get(0, 'DefaultFigureVisible');
restore_figure_visible = onCleanup(@() set(0, 'DefaultFigureVisible', old_figure_visible));
set(0, 'DefaultFigureVisible', 'off');
evalc('startintlab');
set(0, 'DefaultFigureVisible', old_figure_visible);
clear restore_figure_visible
evalc('intvalinit(''DisplayInfsup'')');

data = load(fullfile(directory, 'stiff_matrix.mat'));
stiffness = infsup(sparse(data.i, data.j, data.lo, data.n, data.n), ...
                    sparse(data.i, data.j, data.hi, data.n, data.n));

data = load(fullfile(directory, 'mass_matrix.mat'));
mass = infsup(sparse(data.i, data.j, data.lo, data.n, data.n), ...
              sparse(data.i, data.j, data.hi, data.n, data.n));

[lambda, index] = veigs(stiffness, mass, 2, 'sa');
target = find(index == 2);
if numel(target) ~= 1
    error('VEIGS did not return a unique enclosure for generalized eigenvalue 2.');
end

lambda_fem_lb = inf(lambda(target));
lambda_fem_ub = sup(lambda(target));
lambda_fem_ind = index(target);
save(fullfile(directory, 'verified_eigenvalue.mat'), ...
     'lambda_fem_lb', 'lambda_fem_ub', 'lambda_fem_ind', '-v4')

fprintf('lambda_FEM(2) = [%.17g, %.17g]\n', lambda_fem_lb, lambda_fem_ub);
