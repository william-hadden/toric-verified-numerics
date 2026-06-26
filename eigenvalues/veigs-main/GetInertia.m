function [neg_n, pos_n, zero_n, F] = GetInertia(D, AbortOnErr)
% Compute the inertia of a block-diagonal symmetric matrix D.
%
% D is a real symmetric matrix composed of 1-by-1 or 2-by-2 diagonal blocks.
%
% In case the inertia cannot be determined, if AbortOnErr == 0,
% then F = 1 will be returned (failure flag); otherwise, the program
% exits with an error.
%
% Xuefeng Liu (2011) : Original implementation.
% Yuka Yanagisawa: Clean version (comments and robustness updated) (Nov.14.2025)

neg_n  = 0;
zero_n = 0;
pos_n  = 0;
F      = 0;

% Check AbortOnErr as a local variable and initialize if needed
if ~exist('AbortOnErr','var') || isempty(AbortOnErr)
    AbortOnErr = 0;
end

% Ensure we work with INTLAB interval data
if ~isa(D, 'intval')
    D = intval(D);
end

n = size(D, 1);
i = 1;

while i <= (n - 1)

    % Case 1: 1-by-1 diagonal entry
    if D(i,i+1) == 0
        if D(i,i) > 0
            pos_n = pos_n + 1;
        elseif D(i,i) < 0
            neg_n = neg_n + 1;
        end
        i = i + 1;
        continue;
    end

    % Case 2: 2-by-2 symmetric block
    % Interval determinant of the 2x2 block
    my_det = D(i,i) * D(i+1,i+1) - D(i,i+1) * D(i+1,i);

    if sup(my_det) < 0
        % Two eigenvalues have different signs.
        pos_n = pos_n + 1;
        neg_n = neg_n + 1;
        i     = i + 2;
        continue;
    end

    if inf(my_det) > 0
        % Two eigenvalues have the same sign and are non-zero.
        % cond is the (interval) trace of the 2x2 block.
        % It is used to decide whether both eigenvalues are positive
        % or both are negative.
        cond = D(i,i) + D(i+1,i+1);

        if inf(cond) > 0
            pos_n = pos_n + 2;
            i     = i + 2;
            continue;
        elseif sup(cond) < 0
            neg_n = neg_n + 2;
            i     = i + 2;
            continue;
        end
    end

    % If the inertia cannot be determined:
    if AbortOnErr == 1
        error('Cannot determine the inertia of D.');
    else
        % F = 1 indicates failure to determine the inertia.
        F = 1;
        return;
    end
end

% Last remaining 1-by-1 block if n is odd
if i == n
    if D(i,i) > 0
        pos_n = pos_n + 1;
    elseif D(i,i) < 0
        neg_n = neg_n + 1;
    end
end

zero_n = n - pos_n - neg_n;
F      = 0;   % Success case

end
