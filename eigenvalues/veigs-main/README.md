# veigs

Verified eigenvalue solver for the generalized eigenproblem $A x = \lambda B x$ (MATLAB).

This toolbox provides verified (interval) bounds for eigenvalues of the symmetric generalized eigenproblem using LDL-based separation and the Lehmann–Behnke inclusion. It is intended for small-to-medium dense or sparse problems.

Authors:
- Xuefeng Liu
- Yuuka Yanagisawa


Key files:
- [`veigs`](veigs.m) — main high-level routine for cluster-aware verified eigenvalue bounds. See [veigs.m](veigs.m).
- [`veig`](veig.m)  — interval solver for small generalized eigenproblems used in Lehmann–Behnke steps. See [veig.m](veig.m).
- [`GetInertia`](GetInertia.m) — inertia counter for 1×1 and 2×2 diagonal blocks from LDL. See [GetInertia.m](GetInertia.m).
- License: [LICENSE](LICENSE)

Requirements
- INTLAB (interval arithmetic toolbox for MATLAB/Octave): `intval`, `mid`, `inf`, `sup`, `hull`, `isspd`, etc.

Quick usage
- Call the main function as:
```matlab
[lambdas, ind_range] = veigs(A, B, NumOfEigs, SIGMA)
```

Exmaple 1 - Hilbert matrix
```matlab
% Example 1:
n = 8;
A = eye(n);
B = infsup(hilb(n)-1E-13, hilb(n)+1E-13);

try
    % Eigenvalues near 3.5 (request 3 eigenvalue near sigma = 3.5)
    [bounds, ind] = veigs(A, B, 3, 3.5);
    fprintf("Eigenvalue(s) near 3.5 (indices: %s)\n", mat2str(ind));
    arrayfun(@(a,b) fprintf("  lower = %.15g, upper = %.15g\n", a, b), inf(bounds), sup(bounds));

    % Smallest real eigenvalue
    [bounds, ind] = veigs(A, B, 1, 'sa');
    fprintf("Smallest real (indices: %s)\n", mat2str(ind));
    arrayfun(@(a,b) fprintf("  lower = %.15g, upper = %.15g\n", a, b), inf(bounds), sup(bounds));

    % Largest real eigenvalue
    [bounds, ind] = veigs(A, B, 1, 'la');
    fprintf("Largest real (indices: %s)\n", mat2str(ind));
    arrayfun(@(a,b) fprintf("  lower = %.15g, upper = %.15g\n", a, b), inf(bounds), sup(bounds));
catch ME
    fprintf("Error running example: %s\n", ME.message);
    fprintf("Ensure INTLAB is installed and initialized.\n");
end
```

Output
```console
Eigenvalue(s) near 3.5 (indices: [1 2 3])
  lower = 0.589643850288603, upper = 0.589643850289016
  lower = 3.35429531632129, upper = 3.35429531633695
  lower = 38.1492376817517, upper = 38.1492376835712
Smallest real (indices: 1)
  lower = 0.589643850288603, upper = 0.589643850289016
Largest real (indices: 8)
  lower = 8963150992.40563, upper = 9029922118.19569
```

Example 2 — banded test
```matlab
% Example 2:
n = 10;
A = diag(ones(n,1))*6; A(1,1)=5; A(n,n)=5;
A = A + diag( ones(n-1,1),1)*(-4) + diag( ones(n-1,1),-1)*(-4);
A = A + diag( ones(n-2,1),2) + diag( ones(n-2,1),-2);
B = hilb(n) * 232792560;

% Largest eigenvalue verified bounds
[bounds, ind] = veigs(A, B, 5,'la');
arrayfun(@(a,b) fprintf("  lower = %.15g, upper = %.15g\n", a, b), inf(bounds), sup(bounds));
```
Output
```console
  lower = 0.00333832070468501, upper = 0.00333832070499474
  lower = 0.191995342376582, upper = 0.19199534288305
  lower = 15.6094796620753, upper = 15.6094817202553
  lower = 2014.63063143566, upper = 2014.65296932523
  lower = 550115.445510997, upper = 551048.301519021
```

Notes
- The algorithm builds on LDL factorizations of A - λB (midpoint) and uses verified interval arithmetic to certify bounds. For clustered eigenvalues it applies Lehmann–Behnke complementary variational principles.
- Designed for small-to-medium problems; performance depends on problem size and whether sparse routines (`eigs`) are applicable.
- The code uses INTLAB interval objects and utilities (e.g. intval, mid, inf, sup, hull, isspd).

References
- H. Behnke, "The calculation of Guaranteed bounds for eigenvalues using complementary variational principles", Computing 47, 11–27 (1991).
- N. Yamamoto, "A simple method for error bounds of eigenvalues of symmetric matrices", Linear Algebra and its Applications, 234 (2001), 227–234.
