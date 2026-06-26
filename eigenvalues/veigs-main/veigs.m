function [lambda,ind_range] = veigs(A, B, varargin)
%% VEIGS(A,B,NumOfEigs,SIGMA)
% Finds verified bounds for eigenvalues of the generalized eigenproblem A x = λ B x.
%
% ------------- Input Parameters: A, B, SIGMA ----------------------------
%
% A and B must be symmetric (real or interval) matrices, and B must be positive definite.
% If B is not positive definite, the program will fail to continue.
% A and B can be in sparse matrix format.
% The scheme for calculating approximate eigenvalues depends on the format of A and B.
%
% NumOfEigs is optional and specifies the minimum number of eigenvalues to be returned.
% Default value is 1.
%
% SIGMA is optional. The default value of SIGMA is 'largestabs'.
%
% VEIGS(A,B,NumOfEigs,SIGMA) returns bounds for eigenvalue(s) as below:
%   SIGMA = 'sa' or 'smallestreal' : smallest eigenvalue
%   SIGMA = 'la' or 'largestreal' : largest eigenvalue
%   SIGMA = 'sm' or 'smallestabs': eigenvalue of smallest magnitude
%   SIGMA = 'lm' or 'largestabs' : eigenvalue of largest magnitude
% If SIGMA is a real scalar, VEIGS(A,B,SIGMA) finds the bound for eigenvalue(s) near SIGMA.
%
% ----------------------- Output -----------------------------------------
%
% [lambda] = VEIGS(A,B) returns the bound(s) of eigenvalues in INTVAL format.
% The returned lambda can represent one eigenvalue or several eigenvalues in a cluster.
%
% [lambda, ind] = VEIGS(A,B) returns both eigenvalue bounds and corresponding index/indices.
%
% If either A or B is in sparse matrix format, VEIGS first calls EIGS to compute approximate
% eigenvalues.  If EIGS fails or A and B are full matrices, the code turns to EIG,
% which may take longer for large matrices.
%----------------------- Dependencies -------------------------------------
%   INTLAB (required):
%     - intval, mid, inf, sup, hull, isspd, midrad, mag, in0, getround/setround
%   Local helpers (must be on MATLAB/Octave path):
%     - GetInertia.m   : inertia counter for 1x1/2x2 LDL blocks
%     - veig.m         : interval solver for small generalized EVP used inside
%                        the Lehmann–Behnke step (solves (SA, SB) subproblems)
%----------------------- References -------------------------------------
% 1) H. Behnke, Clausthal, "The calculation of Guaranteed bounds for eigenvalues using complementary variational principles",
%     Computing 47, 11-27 (1991).
%  2) Yamamoto N., "A simple method for error bounds of eigenvalues of symmetric matrices",
%      Linear Algebra and its Application, 234(2001), 227-234
%
% Xuefeng Liu (2011) : Original implementation.
% Yuka Yanagisawa: Clean version (Nov.7.2025)
% Yuka Yanagisawa: Add eigs compatibility wrapper for R2024a name-value syntax (Nov.13.2025)

% ---- Defaults (backward compatible) ----
NumOfEigs  = 1;      % default count
SIGMA      = 'lm';   % default target (legacy mnemonic, mapped below)

if nargin < 2
    error('veigs requires at least A and B.');
end

% ---- Lightweight dispatcher by type (minimal change) ----
switch numel(varargin)
    case 0
        % use defaults
    case 1
        v3 = varargin{1};
        if ischar(v3) || (isstring(v3) && isscalar(v3))
            % Old style: veigs(A,B,SIGMA), NumOfEigs  = 1
            SIGMA = char(v3);
        elseif isnumeric(v3) && isscalar(v3) && isinteger(v3)
            % New style: veigs(A,B,NumOfEigs), SIGMA = 'lm'
            NumOfEigs = v3;
        else
            error('3rd arg must be SIGMA (char/string) or NumOfEigs (numeric scalar).');
        end
    case 2
        v3 = varargin{1}; v4 = varargin{2};
        % ---------------------------
        % Case 1: both numeric
        %   veigs(A,B,NumOfEigs,K)
        % ---------------------------
        if isnumeric(v3) && isscalar(v3) && isnumeric(v4) && isscalar(v4)
            NumOfEigs = v3; SIGMA = v4;
            % ---------------------------
            % Case 2: (numeric , string)
            %   New style: veigs(A,B,5,'sa')
            % ---------------------------
        elseif isnumeric(v3) && isscalar(v3) && (ischar(v4) || (isstring(v4)&&isscalar(v4)))
            NumOfEigs = v3; SIGMA = char(v4);
        else
            error('3rd and 4th args must be NumOfEigs,SIGMA.');
        end
    otherwise
        error('Too many input arguments.');
end

%% --------- Configuration of parameters ---------------------------------

lambda_ratio = 1E-5;   % Value used to determine ρ and σ.
min_dist     = 1E-8;   % Minimum relative distance among eigenvalues in a cluster.
do_shift     = 1;      % Whether eigenvalue shifting is used.

%%% ----------- Step 1: Parsing parameters -------------------------------
lambda       = nan;
ind_range    = nan;
eig_list     = [];
SIGMA_max    = 0;
SIGMA_min    = 0;

% Number of approximate eigenvalues required in calling EIGS.
if NumOfEigs == 1
    EigNum = min(10, size(A,1));
else
    EigNum = min(size(A,1), NumOfEigs+2);
end

% Normalize old shorthand -> new keywords (for name-value usage)
% Old: 'lm', 'sm', 'la', 'sa'.
switch SIGMA
    case 'lm', SIGMA = 'largestabs';
    case 'sm', SIGMA = 'smallestabs';
    case 'la', SIGMA = 'largestreal';
    case 'sa', SIGMA = 'smallestreal';
end

%% -------- Approximate eigenvalue computation ---------------------------
if ischar(SIGMA) || isstring(SIGMA)
    if any(strcmpi(char(SIGMA), {'smallestabs','largestabs','largestreal','smallestreal'}))
        [V,D,F] = eigs(mid(A), mid(B), EigNum, char(SIGMA));
        if F ~= 0
            warning('EIGS failed to converge; switching to full EIG for approximate eigenvalues.');
            [V,D] = eig(full(mid(A)), full(mid(B)));
        end
        [eig_list,p] = sort(diag(D));
    else
        error(['Parameter SIGMA not available: ', char(SIGMA)]);
    end

    switch lower(char(SIGMA))
        case 'smallestreal'
            SIGMA_min = 1; [~,ind] = min(eig_list);
        case 'largestreal'
            SIGMA_max = 1; [~,ind] = max(eig_list);
        case 'smallestabs'
            [~,ind] = min(abs(eig_list));
        otherwise % 'largestabs'
            [~,ind] = max(abs(eig_list));
    end
else
    [V,D,F] = eigs(mid(A), mid(B), EigNum, SIGMA);
    if F ~= 0
        warning('EIGS failed to converge; switching to full EIG.');
        [V,D] = eig(full(mid(A)), full(mid(B)));
    end
    [eig_list,p] = sort(diag(D));
    [~,ind] = min(abs(eig_list - SIGMA));
end

V = V(:,p);
m = size(V,2);  % eig_list(m) has the maximum value of eig_list

%% -------- Initialize global bounds ------------------------------------
global_sigma = inf;
global_rho   = -inf;
global_s = 0;
global_r = size(A,1)+1;

% Compute rough lower bound for B and norm of A
[~,lambda_B_min] = eigs(mid(B),1,'smallestabs');
lambda_B_min = veigs_RoughLower(B, speye(size(B)), lambda_B_min, lambda_B_min/2);
A_inf_norm = norm(sup(abs(A)),'inf');
lambda_rough_min = -A_inf_norm / lambda_B_min;
lambda_rough_max =  A_inf_norm / lambda_B_min;
var_rough_bounds = [lambda_B_min, lambda_rough_min, lambda_rough_max];

%%% ----------- Step 2: Determination of r and s -------------------------
% According to Behnke’s paper:
% λ_{r−1} < λ_r ≈ λ_s < λ_{s+1}
% The (ind)-th eigenvalue is the approximate eigenvalue of interest.
s = ind;
direction_left = 1;
direction_right = 1;

if SIGMA_max == 1  % largest eigenvalue
    s = m;
    global_sigma = veigs_RoughUpper(A,B,eig_list(s),lambda_rough_max);
    global_s = size(A,1);
    direction_right = 0; % no eigenvalue on right of maximum eigenvalue
end

if SIGMA_min == 1  % smallest eigenvalue
    r = 1;
    global_rho = veigs_RoughLower(A,B,eig_list(r),lambda_rough_min);
    global_r = 1;
    direction_left = 0; % no eigenvalue on left of minimum eigenvalue
end

%%% ----------- Step 3: Verification loop for clusters -------------------
do_next = 1;
ind_range = [];
lambda = [];

while do_next
    if isempty(ind_range) % first call of veigs_EigLocalBound
        para = [global_r, global_s, global_rho, global_sigma, ...
            SIGMA_min, SIGMA_max, direction_left, direction_right];
        [lambda, local_ind_range, global_ind_range] = ...
            veigs_EigLocalBound(A,B,eig_list,V,ind,var_rough_bounds,para);
        ind_range = global_ind_range;
    else
        ind_left = min(local_ind_range)-1;
        ind_right = max(local_ind_range)+1;
        local_ind_range_left = [];
        local_ind_range_right = [];
        global_s = ind_range(1)-1;
        global_r = ind_range(end)+1;

        if direction_right == 1
            global_rho = max(inf(lambda));
            para = [global_r,global_s,global_rho,global_sigma, ...
                SIGMA_min,SIGMA_max,0,direction_right];
            [local_lambda,local_ind_range_right,global_ind_range] = ...
                veigs_EigLocalBound(A,B,eig_list,V,ind_right,var_rough_bounds,para);
            lambda = [lambda, local_lambda];
            ind_range = [ind_range, global_ind_range];
        end

        if direction_left == 1
            global_sigma = min(sup(lambda));
            para = [global_r,global_s,global_rho,global_sigma, ...
                SIGMA_min,SIGMA_max,direction_left,0];
            [local_lambda,local_ind_range_left,global_ind_range] = ...
                veigs_EigLocalBound(A,B,eig_list,V,ind_left,var_rough_bounds,para);
            lambda = [local_lambda, lambda];
            ind_range = [global_ind_range, ind_range];
        end
        local_ind_range = [local_ind_range_left, local_ind_range_right];
    end

    g_r = min(ind_range);
    g_s = max(ind_range);

    do_next = 0;
    if g_s-g_r+1 < NumOfEigs
        if min(local_ind_range)>1 && direction_left==1
            do_next = 1;
        else
            direction_left = 0;
        end
        if max(local_ind_range)<m && direction_right==1
            do_next = 1;
        else
            direction_right = 0;
        end
    end
end

lambda = lambda';
end


%% =======================================================================
%%  Subfunction: Local bound computation (core)
%% =======================================================================
function [lambda, local_ind_range,ind_range] = veigs_EigLocalBound( A,B, eig_list, V, ind, var_rough_bounds, para)
% The parameter lambda_ratio and do_shift are for Behnke's method.
%
% The value of lambda_ratio is used to determine rho and sigma.
%   rho = eig_list(r-1)*(1-lambda_ratio) + eig_list(r)*lambda_ratio;
%   sigma = eig_list(s)*lambda_ratio + eig_list(s+1)*(1-lambda_ratio);
% where eig_list is a local list of eigenvalues.

lambda_ratio = 1E-5;

% The minimum relative distance among eigenvalues in a cluster.
%   lambda_r - lambda_{r+1} > min_dist * ( lambda_r + lambda_{r+1} )
% (Note: not used explicitly below; retained here conceptually in comments.)

% Whether eigenvalue shift is used or not.
do_shift = 1;

% --- locals from rough bounds vector ---
lambda_B_min     = var_rough_bounds(1);
lambda_rough_min = var_rough_bounds(2);
lambda_rough_max = var_rough_bounds(3);

% Obtain parameter values from 'para'
global_r     = para(1);
global_s     = para(2);
global_rho   = para(3);
global_sigma = para(4);
SIGMA_min    = para(5);
SIGMA_max    = para(6);
direction_left  = para(7);
direction_right = para(8);

r = ind;
s = ind;
m = length(eig_list);

%% -------------------------------------------------------------------------
%  Right-side (upper) verification
%  Determine s and an upper separator sigma so that
%  lambda_{s} ~ target cluster and lambda_{s+1} > sigma.
% --------------------------------------------------------------------------
if direction_right == 1

    while s < m && (eig_list(s) > eig_list(s+1) - lambda_ratio*abs(eig_list(s+1)) )
        s = s + 1;
    end

    % Select the next approximate eigenvalue to the right
    if s == m
        eig_next = eig_list(s) + abs(eig_list(s))*0.01;
    else
        eig_next = eig_list(s+1);
    end

    % Interpolate a test value between eig_list(s) and eig_next
    lambda = lambda_ratio*eig_list(s) + (1-lambda_ratio)*eig_next;
    tmpM   = intval(A) - lambda*intval(B);

    [L,D,P] = ldl( mid(tmpM) );
    % inertia of D; (neg_num + zero_num) counts non-positive directions
    [neg_num,~,zero_num] = GetInertia(D,1);

    if (neg_num + zero_num == size(A,1))
        % This means A - lambda*B is (almost) non-positive definite everywhere:
        % treat as the "largest-eigenvalue" side and set a rough lower bound for s+1-th eigenvalue
        SIGMA_max    = 1;
        s            = m;
        global_sigma = veigs_RoughUpper( A, B, eig_list(s), lambda_rough_max);
        global_s     = size(A,1);
    else
        % Compare PLDL'P' vs. (A - lambda B) to estimate the interval radius
        diff_m  = P*intval(L)*intval(D)*intval(L')*P' - tmpM; % (not exactly symmetric, hull used)
        err_est = norm( sup(abs(diff_m)),'inf') / lambda_B_min;

        % Reduce the error estimate iteratively, ensuring positivity of
        %   err_est * B - ( PLDL'P' - (A - lambda B) )
        % This guarantees the perturbation is dominated by err_est * B.
        if ( err_est > (1-lambda_ratio) * ( eig_next - eig_list(s) ) )
            is_positive = 1;
            while ( is_positive == 1 )
                err_est_test = err_est/2;
                % Check isspd( err_est*B - ( PLDL'P' - (A - sigma B)) )
                tmpMatrix  = intval(err_est_test)*intval(B) - diff_m;
                is_positive = isspd( hull(tmpMatrix, tmpMatrix') );
                clear tmpMatrix ;
                if ( ~is_positive )
                    break;
                end
                err_est = err_est_test;
            end
        end

        % Check feasibility of the bound:
        is_positive = 1;
        if ( err_est > (1-lambda_ratio) * ( eig_next - eig_list(s) ) )
            is_positive = 0;
        end

        if ( ~is_positive )
            if ( neg_num == 1 )
                % Turn to a rough lower bound by a direct isspd test.
                SIGMA_min = 1;
                r = 1;
                global_rho = veigs_RoughLower(A,B, eig_list(r), lambda_rough_min );
                global_r   = 1;
            else
                error( ['LDL factorization: failed to find a lower bound of eigenvalue lambda_{',num2str(s+1),'}.' ] );
            end
        else
            % From computation above, we can declare that the (global_s + 1)-th eigenvalue > global_sigma.
            global_s     = neg_num;
            global_sigma = inf( lambda - intval(err_est) );
        end
    end
end

%% -------------------------------------------------------------------------
%  Left-side (lower) verification
%  Determine r and a lower separator rho so that
%  lambda_{r-1} < rho < lambda_r ~ target cluster.
% --------------------------------------------------------------------------
if direction_left == 1

    r = ind;
    while r > 1 && ( eig_list(r) < eig_list(r-1) + lambda_ratio*abs(eig_list(r-1)) )
        r = r - 1;
    end

    if r == 1
        eig_pre = eig_list(r) - abs(eig_list(r))*0.01;
    else
        eig_pre = eig_list(r-1);
    end

    lambda = lambda_ratio*eig_list(r) + (1-lambda_ratio)*eig_pre;
    tmpM   = intval(A) - lambda*intval(B);

    [L,D,P] = ldl( mid(tmpM) );
    neg_num = GetInertia(D);

    if ( neg_num == 0 )
        % Treat as the "smallest-eigenvalue" side:
        SIGMA_min = 1;
        r = 1;
        global_rho = veigs_RoughLower(A,B, eig_list(r), lambda_rough_min);
        global_r   = 1;
    else
        % Error radius via difference of factorization and original
        diff_m  =  tmpM - P*intval(L)*intval(D)*intval(L')*P';
        err_est = norm( sup(abs(diff_m)),'inf') / lambda_B_min;

        if ( err_est > (1-lambda_ratio) * ( eig_list(r) - eig_pre ) )
            is_positive = 1;
            while ( is_positive == 1 )
                err_est_test = err_est/2;
                % Check isspd( err_est*B - ( PLDL'P' - (A - sigma B)) )
                tmpMatrix  = intval(err_est_test)*intval(B) - diff_m;
                is_positive = isspd( hull(tmpMatrix, tmpMatrix') );
                clear tmpMatrix ;
                if( ~is_positive )
                    break;
                end
                err_est = err_est_test;
            end
        end

        is_positive = 1;
        if ( err_est > (1-lambda_ratio) * ( eig_list(r) - eig_pre ) )
            is_positive = 0;
        end

        if ( ~is_positive ) % Turn to a rough upper bound by a direct isspd test
            if ( neg_num == size(A,1)-1 )
                % It arrived here because it failed to give the lower bound for the maximum eigenvalue
                SIGMA_max    = 1;
                s            = m;
                global_sigma = veigs_RoughUpper( A, B, eig_list(s), lambda_rough_max);
                global_s     = size(A,1);
            else
                error(['LDL factorization: failed to find an upper bound of eigenvalue lambda_{',num2str(r-1),'}.']) ;
            end
        else
            % Selection of lambda and err_est_base ensures global_rho < eig(global_r).
            global_r  = neg_num + 1;
            global_rho = sup( lambda + intval(err_est) );
        end

    end
end

%% -------------------------------------------------------------------------
%  LDL Result check
%  Check whether LDL failed or not.
%  If failed to separate properly, resort to Rayleigh-type bounds.
% -------------------------------------------------------------------------
if ( global_s - global_r ~= s - r  || global_s < 0 || global_r > size(A,1) )
    if ( SIGMA_max == 1 && SIGMA_min == 1 ) % Rayleigh bounds for both ends
        lower_bound = V(:,m)'*intval(A)*V(:,m)/(V(:,m)'*intval(B)*V(:,m));
        upper_bound = V(:,1)'*intval(A)*V(:,1)/(V(:,1)'*intval(B)*V(:,1));
        lambda    = [ hull( global_rho, upper_bound);  hull( lower_bound, global_sigma) ];
        ind_range = [1, size(A,1)];
        local_ind_range = [1, size(A,1)];
        return
    end

    if ( SIGMA_max == 1 ) % Rayleigh bound for maximum eigenvalue
        lower_bound = V(:,m)'*intval(A)*V(:,m)/(V(:,m)'*intval(B)*V(:,m));
        lambda    = infsup( lower_bound.inf, global_sigma);
        ind_range = size(A,1);
        local_ind_range = ind_range;        
        return
    end

    if ( SIGMA_min == 1 ) % Rayleigh bound for minimum eigenvalue
        upper_bound = V(:,1)'*intval(A)*V(:,1)/(V(:,1)'*intval(B)*V(:,1));
        lambda    = infsup( global_rho, upper_bound.sup);
        ind_range = 1;
        local_ind_range = ind_range;        
        return
    end
end

if ( global_s - global_r > s - r )
    error('Computation failed in finding upper and lower bounds for clustered eigenvalues around the given value. Please consider increasing the value of EigNum in this code.')
end

if ( global_s - global_r < s - r )  % This happens in very rare cases when approximate eigenvalues are in poor precision
    error('Computation failed in finding rough bounds  rho < lambda_s ~ lambda_r < sigma ')
end

% When we apply Lehmann's theorem, it is better to have sigma and rho separated from the target eigenvalue (cluster).
if ( SIGMA_max == 1 )
    global_sigma = max( global_sigma, eig_list(m) + abs(eig_list(m)) );
end
if ( SIGMA_min == 1 )
    global_rho   = min( global_rho, eig_list(r) - abs( eig_list(r)) );
end

%%% global_sigma: lower bound of (s+1)-th eigenvalue.
%%% global_rho  : upper bound of (r-1)-th eigenvalue.

V = V(:,r:s);

%%% Lehmann–Behnke's method (default).
lambda = Lehmann_Behnke__(A,B, eig_list, V, global_rho, global_sigma, lambda_B_min, r, s, do_shift);

ind_range       = global_r:1:global_s;
local_ind_range = r:s;

end


%% =======================================================================
%%  Subfunction: Rough lower bound via isspd( A - λB )
%% =======================================================================
function [bound] = veigs_RoughLower(A,B,eig_test,eig_test_min)
% Roughly finds a lower bound for the minimum eigenvalue by testing
% the positive definiteness of A - λB iteratively, decreasing λ.
k = 1;
delta  = max(eps, (eig_test - eig_test_min)/2^32);
lambda = eig_test - delta*2^(k-1);
is_pd  = false;

while (lambda >= eig_test_min - eps) || (k == 1)
    tmpMatrix = intval(A) - lambda * intval(B);
    is_pd = isspd( hull(tmpMatrix, tmpMatrix') );
    if is_pd
        break
    end
    k = k + 1;
    lambda = eig_test - delta*2^(k-1);
end

if is_pd
    bound = lambda;
else
    error('Failed to find a rough lower bound for the minimum eigenvalue.');
end
end


%% =======================================================================
%%  Subfunction: Rough upper bound via isspd( λB - A )
%% =======================================================================
function [bound] = veigs_RoughUpper(A,B,eig_test,eig_test_max)
% Roughly finds an upper bound for the maximum eigenvalue by testing
% the positive definiteness of λB - A iteratively, increasing λ.
k = 1;
delta  = max(eps, (eig_test_max - eig_test)/2^32);
lambda = eig_test + delta*2^(k-1);
is_pd  = false;

while (lambda <= eig_test_max + eps) || (k == 1)
    tmpMatrix = lambda * intval(B) - intval(A);
    is_pd = isspd( hull(tmpMatrix, tmpMatrix') );
    if is_pd
        break
    end
    k = k + 1;
    lambda = eig_test + delta*2^(k-1);
end

if is_pd
    bound = lambda;
else
    error('Failed to find a rough upper bound for the maximum eigenvalue.');
end
end


%% =======================================================================
%%  Subfunction: Lehmann–Behnke inclusion theorem
%% =======================================================================
function lambda = Lehmann_Behnke__(A,B, eig_list, V, global_rho, global_sigma, lambda_B_min, r, s, do_shift)
% Computes verified bounds for clustered eigenvalues using Lehmann–Behnke
% complementary variational principles.

% --- Step 1: (Optional) shift to improve conditioning ---
c = lambda_B_min;
lambda_shift = 0;
if do_shift > 0
    lambda_shift = eig_list(r);
    A = intval(A) - lambda_shift * intval(B);
    global_rho   = sup( global_rho   - intval(lambda_shift) );
    global_sigma = inf( global_sigma - intval(lambda_shift) );
end

% --- Step 2: Construct submatrices in Lehmann’s formulation ---
A0 = V' * intval(B) * V;      % interval
A1 = V' * intval(A) * V;      % interval

% nV := B^{-1} A V, computed with midpoints (rough inverse for residual)
nV = mid(B) \ ( mid(A) * mid(V) );
err_vec = intval(B) * intval(nV) - intval(A) * intval(V);
Err     = (err_vec' * err_vec) / intval(c);   % conservative residual energy

% A2 gathers coupling and residual terms (interval-symmetric via hull where needed)
A2 = V' * A * intval(nV) - intval(nV)' * ( B * intval(nV) - A * V ) + Err;

% --- Step 3: Lower bound via Lehmann (sigma side) ---
sigma = global_sigma;
SA = (- A2 + sigma * A1);
SB = (- A1 + sigma * A0);

if ~isspd( hull(SB, SB') )
    error('Lehmann theorem failed: SB is not positive definite (lower bound side).');
end

if (s == r)
    low = SA / SB;                        % scalar Rayleigh-type ratio with intervals
else
    low = veig( hull(SA,SA'), hull(SB,SB') );  % verified generalized eigen bound
end

% --- Step 4: Upper bound via Lehmann (rho side) ---
rho = global_rho;
SA = ( A2 - rho * A1 );
SB = ( A1 - rho * A0 );

if ~isspd( hull(SB, SB') )
    error('Lehmann theorem failed: SB is not positive definite (upper bound side).');
end

if (s == r)
    upper = SA / SB;
else
    upper = veig( hull(SA,SA'), hull(SB,SB') );
end

% --- Step 5: Compose the verified interval (undo shift if applied) ---
lambda = infsup( inf( low   + intval(lambda_shift) ), ...
    sup( upper + intval(lambda_shift) ) );
end
