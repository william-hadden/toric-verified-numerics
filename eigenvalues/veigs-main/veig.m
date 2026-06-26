function [eig_bounds, ind_range] = veig(A, B, ind)
%% VEIG: Compute verified eigenvalue bounds for the generalized eigenproblem A x = λ B x.
%
% INPUT:
%   A, B : Real symmetric or interval matrices of size n-by-n.
%          B must be positive definite (interval PD allowed).
%   ind  : Eigenvalue index (scalar) or continuous range of indices.
%
% OUTPUT:
%   eig_bounds : Interval enclosures of the requested eigenvalues.
%   ind_range  : Actual indices of the eigenvalues returned.
%
% NOTES:
% 1) The returned bounds may include more eigenvalues than those specified in "ind".
% 2) This function is intended for small- or medium-scale dense matrices.
%    Although the block LDL algorithm handles sparse matrices, the call to
%    the standard EIG function (to obtain approximate eigenvalues) may be expensive.
%
% REFERENCE:
%   Yamamoto N., “A simple method for error bounds of eigenvalues of symmetric matrices”,
%   Linear Algebra and its Applications, 234 (2001), 227–234.
%
% AUTHOR / HISTORY:
%   Original: Xuefeng Liu
%   Yuka Yanagisawa: Clean version (Nov.21.2025)

%% ------------------------------------------------------------
% Basic checks and setup
%% ------------------------------------------------------------

eig_bounds = [];
ind_range  = [];

% --- Size consistency check (added) ---
[nA1,nA2] = size(A);
[nB1,nB2] = size(B);
if nA1 ~= nA2 || nB1 ~= nB2 || nA1 ~= nB1
    error('A and B must be square matrices of the same size.');
end
n = nA1;

% Symmetry checks
if ~isequal(A,A')
    error('Matrix A is not symmetric.');
end
if ~isequal(B,B')
    error('Matrix B is not symmetric.');
end

% Problem scale restriction
if n > 100
    error("VEIG is designed for small/medium dense matrices. Choose another solver.");
end

% --- Process eigenvalue index range ---
if ~exist("ind","var") || isempty(ind)
    min_ind = 1;
    max_ind = n;
else
    if ~isnumeric(ind)
        error('Input parameter "ind" must be numeric.');
    end
    min_ind = min(ind);
    max_ind = max(ind);
end

% --- Range check for ind (added) ---
if min_ind < 1 || max_ind > n
    error('The index range "ind" must satisfy 1 ≤ ind ≤ n.');
end

% Convert A,B to interval (for consistency)
if ~isa(A,'intval'), A = intval(A); end
if ~isa(B,'intval'), B = intval(B); end

% --- Single computation of mid A,B (added) ---
A_mid = mid(A);
B_mid = mid(B);

%% ------------------------------------------------------------
% Compute approximate eigenvalues
%% ------------------------------------------------------------

if issparse(A) || issparse(B)
    eig_list = eig(full(A_mid), full(B_mid));
else
    eig_list = eig(A_mid, B_mid);
end

eig_bounds = intval([]);
min_eig_B  = min(eig(B_mid));  % B positive definite

index = min_ind;

%% ------------------------------------------------------------
% Main loop over requested eigenvalue indices
%% ------------------------------------------------------------
while index <= max_ind

    eig_test = eig_list(index);

    %% ------------------------------------------------------------
    % Rough lower bound
    %% ------------------------------------------------------------
    lambda_test = eig_test;
    find_lambda = false;
    k = 0;

    if index == 1
        neg_num_low     = 0;
        lambda_test_low = eig_test;
    else
        while ~find_lambda
            if k > 10
                error('Rough lower bound search exceeded iteration limit.');
            end

            E = A_mid - lambda_test*B_mid;
            [L,D,P] = ldl(E);
            [negative_num, positive_num, zero_num, F] = GetInertia(D);

            if negative_num >= index || F ~= 0
                lambda_test = eig_test - max(eps(eig_test),abs(eig_test))*10^(k-10);
                k = k + 1;
            else
                find_lambda = true;
            end
        end
        neg_num_low     = negative_num;
        lambda_test_low = lambda_test;
    end

    %% ------------------------------------------------------------
    % Rough upper bound
    %% ------------------------------------------------------------
    lambda_test = eig_test;
    find_lambda = false;
    k = 0;

    if index == n
        neg_zero_num_upper = n;
        lambda_test_upper  = eig_test;
    else
        while ~find_lambda
            if k > 10
                error('Rough upper bound search exceeded iteration limit.');
            end

            E = A_mid - lambda_test*B_mid;
            [L,D,P] = ldl(E);
            [negative_num, positive_num, zero_num, F] = GetInertia(D);

            if negative_num + zero_num < index || F ~= 0
                lambda_test = eig_test + max(eps(eig_test),abs(eig_test))*10^(k-10);
                k = k + 1;
            else
                find_lambda = true;
            end
        end
        neg_zero_num_upper = negative_num + zero_num;
        lambda_test_upper  = lambda_test;
    end

    %% ------------------------------------------------------------
    % Verified lower bound
    %% ------------------------------------------------------------
    lambda_test = lambda_test_low;
    E_mid = A_mid - lambda_test*B_mid;
    [L,D,P] = ldl(E_mid);
    E      = A - lambda_test*B;

    diff_m     = P*L*D*(L')*P' - E;
    diff_m_inf = norm(sup(abs(diff_m)),'inf');

    if diff_m_inf == 0
        lambda_lower = lambda_test;
    else
        err_est = max(eps(eig_test), diff_m_inf / min_eig_B);
        is_positive = 0;
        k = 0;

        while k < 53 && is_positive < 1
            err_est = err_est * 2;

            tmp = intval(err_est)*B - diff_m;
            is_positive = isspd(hull(tmp,tmp'));

            tmp = intval(err_est)*B + diff_m;
            is_positive = is_positive * isspd(hull(tmp,tmp'));
            k = k+1;
        end

        if is_positive < 1
            error('Failure in LDL-based positive definiteness check (lower bound).');
        end

        lambda_lower = lambda_test - err_est;
    end

    %% ------------------------------------------------------------
    % Verified upper bound
    %% ------------------------------------------------------------
    lambda_test = lambda_test_upper;
    E_mid = A_mid - lambda_test*B_mid;
    [L,D,P] = ldl(E_mid);
    E      = A - lambda_test*B;

    diff_m     = P*L*D*(L')*P' - E;
    diff_m_inf = norm(sup(abs(diff_m)),'inf');

    if diff_m_inf == 0
        lambda_upper = lambda_test;
    else
        err_est = max(eps(eig_test), diff_m_inf / min_eig_B);
        is_positive = 0;
        k = 0;

        while k < 53 && is_positive < 1
            err_est = err_est * 2;

            tmp = intval(err_est)*B - diff_m;
            is_positive = isspd(hull(tmp,tmp'));

            tmp = intval(err_est)*B + diff_m;
            is_positive = is_positive * isspd(hull(tmp,tmp'));
            k = k+1;
        end

        if is_positive < 1
            error('Failure in LDL-based positive definiteness check (upper bound).');
        end

        lambda_upper = lambda_test + err_est;
    end

    %% Store bound
    eig_bound = hull(lambda_lower, lambda_upper);

    index_range = (neg_num_low+1):neg_zero_num_upper;

    if neg_num_low+1 < min_ind
        min_ind = neg_num_low+1;
    end

    eig_bounds(index_range - min_ind + 1) = eig_bound;

    index = neg_zero_num_upper + 1;

end

ind_range = min_ind:(index-1);

end