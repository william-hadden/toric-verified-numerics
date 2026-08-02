# Holomorphic sectional curvature search

The HSC search certifies the curvature of the holomorphic plane spanned by a
real constant torus direction `V` and `JV`.  Directions are represented by
real pairs `xi = (xi1, xi2)`; the current search enumerates primitive rational
pairs, up to an overall real sign.

## Real-direction assumption

The contraction implemented in `compute_Q_rigorously` and in the compact
coefficient-space construction assumes that the components of `xi` are real.
In particular, it uses

```text
u^{ij} xi_i xi_j
```

for the squared length and performs the curvature contraction without complex
conjugation.  For real `xi`, this is exactly

```text
Riem(V, JV, JV, V) / |V|^4,
```

where the division normalizes `V` to unit length.

Do not pass a direction with genuinely complex components to the current
implementation.  For a complex direction, the Hermitian contractions require
complex conjugates, schematically

```text
g_{i j-bar} xi_i conjugate(xi_j)
R_{i j-bar k l-bar} xi_i conjugate(xi_j) xi_k conjugate(xi_l),
```

and the definitions of the coefficient-space polynomials `S`, `W`, `H`, and
`P` would have to be changed accordingly.  Allowing complex numeric
types without adding these conjugations would not compute holomorphic
sectional curvature correctly.

## The coefficient-space formula for `Q`

Step 6 forms the curvature contraction in Chebyshev coefficient space before
evaluating it on boxes. This section defines the intermediate objects `S`,
`W`, `H`, and `P` and derives the formula used in the code.

Write the inverse metric entries as

```text
u^{ij} = A^{ij} / D0,
```

where `A^{ij}` and the common denominator `D0` are represented by rigorous
Chebyshev coefficient enclosures. For their second derivatives, Step 4 has
already used the quotient rule to construct numerators `N^{jl}_{ab}` such
that

```text
partial_a partial_b u^{jl} = N^{jl}_{ab} / D0^3.
```

Fix a real constant torus direction

```text
xi = (xi_1, xi_2).
```

The unnormalised vector in this direction is denoted by `W` geometrically
below; the code's coefficient-space quantities are defined as follows.

### `S`: numerator of the squared direction length

The squared metric length of the unnormalised direction is

```text
D = sum_{i,j} u^{ij} xi_i xi_j.
```

Substituting `u^{ij} = A^{ij}/D0` gives

```text
D = S / D0,
S = sum_{i,j} A^{ij} xi_i xi_j.
```

Thus `S` is the numerator of the squared direction length. The certification
code proves both `D0 > 0` and `S > 0` on every box. It follows that `D > 0`,
so the direction is nonzero and can be normalized.

### `W[a]`: numerator of the metric-raised direction

Define the geometric contraction

```text
w_a = sum_i u^{ai} xi_i.
```

After substituting the rational form of the inverse metric,

```text
w_a = W[a] / D0,
W[a] = sum_i A^{ai} xi_i.
```

The code stores the two coefficient-space numerators as `W[1]` and `W[2]`.
The name `W` in the implementation refers to these numerators, whereas `w_a`
denotes the corresponding rational geometric quantities in this derivation.

### `H[a,b]`: numerator of the contracted second derivative

Define

```text
h_ab = sum_{j,l} (partial_a partial_b u^{jl}) xi_j xi_l.
```

Using the Step 4 derivative numerators gives

```text
h_ab = H[a,b] / D0^3,
H[a,b] = sum_{j,l} N^{jl}_{ab} xi_j xi_l.
```

Thus `H` is a `2 x 2` array of coefficient-space numerator enclosures.

### `P`: numerator of the curvature contraction

For the unnormalised direction, the numerator in the holomorphic sectional
curvature formula is

```text
numerator = -(1/2) sum_{a,b} w_a w_b h_ab.
```

Substituting `w_a = W[a]/D0` and `h_ab = H[a,b]/D0^3` yields

```text
numerator
    = -(1/2) sum_{a,b}
        (W[a]/D0) (W[b]/D0) (H[a,b]/D0^3)
    = P / D0^5,
```

where

```text
P = -(1/2) sum_{a,b} W[a] W[b] H[a,b].
```

The products and sum defining `P` are carried out with the structured
`(coeffs, tail, pdeg)` enclosures. Products involving omitted Chebyshev tails
and any new truncation tails are therefore included rigorously.

### Why `Q = P / (D0^3 S^2)`

The unit vector is the unnormalised direction divided by `sqrt(D)`. Since the
curvature expression contains four copies of the vector, normalization divides
the unnormalised contraction by `D^2`. Therefore

```text
Q = numerator / D^2.
```

Using

```text
numerator = P / D0^5
D = S / D0
```

gives

```text
Q = (P / D0^5) / (S^2 / D0^2)
  = P / (D0^3 S^2).
```

This is exactly the expression evaluated by
`compact_curvature_ranges_on_box`. On each box, the program obtains rigorous
interval enclosures for `D0`, `S`, and `P`, verifies that the denominator is
strictly positive, and then performs interval division:

```julia
Q = P / (D0^3 * S^2)
```

Consequently the returned interval contains `Q(x,y)` for every point in the
box, not merely at a sample point. Subdivision makes the interval enclosure
tighter without changing the formula.

In summary:

```text
S      = sum_{i,j} A^{ij} xi_i xi_j
W[a]   = sum_i A^{ai} xi_i
H[a,b] = sum_{j,l} N^{jl}_{ab} xi_j xi_l
P      = -(1/2) sum_{a,b} W[a] W[b] H[a,b]

D      = S / D0
Q      = P / (D0^3 S^2).
```

## Coordinate convention

The numerical coefficient arrays use internal coordinates `(x, y)` in
`[0,1]^2`.  They correspond to Toby's physical square
`(x, Y) in [0,1] x [-1,0]` through `y = -Y`.

Checkpoint details are documented in [`checkpoints/README.md`](checkpoints/README.md).

## Run parameters

The public entry point is:

```julia
prove_proposition_5_6(
    coeffs_path;
    eta,
    rho,
    pdeg = 30,
    domain = PolytopeBox(0, 1, 0, 1),
)
```

The parameters have the following meanings:

- `coeffs_path` is the rational CSV containing the Chebyshev coefficients of
  the approximate symplectic potential. The checkpoint context stores a
  SHA-256 hash of the file contents, not of its path, so a checkpoint is not
  reused after the coefficient data changes.

- `eta` is the nonnegative multiplicative metric/curvature error appearing in
  Proposition 5.6. It must satisfy `0 <= eta < 1`.

- `rho` is the nonnegative additive curvature error appearing in Proposition
  5.6.

- `pdeg` is the coefficient truncation size used while forming
  coefficient-space products and curvature contractions. A value of 30 keeps
  a `30 x 30` Chebyshev block, corresponding to modes 0 through 29 in each
  coordinate. Coefficients outside this block are not simply discarded: they
  are accumulated into rigorous tail enclosures. Increasing `pdeg` generally
  tightens the result but increases time and memory use. It must be positive.

- `domain` is the rational base-coordinate rectangle searched for a suitable
  region. Its default is the whole internal coordinate square `[0,1]^2`. The
  final manifold region is a subset of `domain x T^2`.

The correction added to the curvature integral is

```text
sqrt(volume) * (eta * Rnorm + (1 + eta) * rho).
```

Consequently, larger `eta` or `rho` makes certification harder. Setting both
to zero removes this correction, but is appropriate only when the mathematical
hypotheses actually give zero errors.

The current entry point also sets the following search parameters internally
near the beginning of `prove_proposition_5_6`. They are recorded in the
run-specific checkpoint directory's `README.txt`, even though they are not
currently keyword arguments.

### Candidate-generation parameters

- `grid = 24` divides each axis of `domain` into 24 equal rational pieces,
  producing a `24 x 24` grid of seed cells. Step 5 evaluates candidate
  directions at each cell centre. A seed with indices `(ix, iy)` refers to
  zero-based cell indices, so `(22, 22)` is the 23rd cell on each axis.
  Candidate rectangle boundaries lie on this rational grid.

  `grid` is only a search-resolution parameter. The cell-centre calculation is
  used to find and rank promising seeds; it is not itself the proof. Step 6
  certifies the full candidate rectangle using interval bounds. Increasing
  `grid` can locate smaller or more localized negative-curvature regions, at a
  seed-search cost proportional to `grid^2`.

- `direction_order = 5` generates all primitive integer directions `(p,q)`,
  up to overall sign, satisfying
  `max(abs(p), abs(q)) <= direction_order`. For example, `(1,1)` and `(1,-1)`
  are included, while scalar duplicates such as `(2,2)` are reduced to
  `(1,1)`. Increasing the order searches more real torus directions and makes
  both seed generation and later candidate evaluation more expensive.

- `directions_per_cell = 2` retains at most the two directions with the most
  negative pointwise seed scores in each grid cell.

- `max_seeds = 100` limits the total number of retained seeds after candidates
  from all cells have been sorted by their pointwise scores.

- `seed_cutoff = 0` keeps a point-direction pair only if its pointwise
  heuristic score is below zero. The score is used solely to select and order
  candidates; it is never treated as a certificate.

- `candidate_shapes` is a tuple of `(rx, ry)` values. For a seed cell, `rx`
  and `ry` specify how many additional grid cells are included on each side in
  the two coordinate directions. Thus `(0,0)` selects just the seed cell,
  `(1,0)` can span up to three cells horizontally and one vertically, and
  `(1,1)` can span up to a `3 x 3` block. Shapes are clipped at the domain
  boundary. The current shapes are:

  ```julia
  (
      (0, 0),
      (1, 0), (0, 1),
      (1, 1),
      (2, 0), (0, 2),
      (2, 1), (1, 2),
  )
  ```

### Certification-search parameters

- `screen_depth = 0` is the number of uniform subdivisions applied to every
  initial candidate before its first Step 6 evaluation. A depth increment
  splits every box into four. Starting at zero first screens candidates on
  their original grid-aligned rectangles.

- `maxdepth = 6` is the largest subdivision depth considered. One initial box
  has `4^d` boxes at depth `d`, so depth 6 can produce 4096 boxes for a single
  surviving candidate. The search stops earlier as soon as it finds a
  certified margin.

- `beam_width = 20` is the maximum number of valid but not-yet-certified
  candidates retained after each depth. They are ranked by `margin_upper`;
  smaller is better. Only the retained candidates are subdivided for the next
  depth.

- `invalid_keep = 20` is the maximum number of unresolved candidates retained
  after each depth. An unresolved candidate is one for which a required
  positivity or interval condition could not be proved on a coarse box.
  Subdivision may make that condition provable, so these candidates are not
  all discarded immediately.

The beam search is deliberately not exhaustive: `max_seeds`, `beam_width`, and
`invalid_keep` prune the search. A successful result is nevertheless a
rigorous certificate because the winning candidate is checked on its entire
region. An unsuccessful result means that this search did not find a
certificate; it is not a proof that no suitable region and direction exist.

### Checkpoint identity parameters

The run folder also records:

- `coefficient_sha256`, a hash of the coefficient file's bytes;
- `bigfloat_precision`, the active `BigFloat` precision in bits;
- the domain, error bounds, truncation degree, and every search parameter
  above.

These values determine the run-specific checkpoint folder. The loader compares
the full stored context and step parameters before reusing a checkpoint, which
prevents results computed from different coefficient data, precision, or
search settings from being silently mixed.

## Inspecting a saved Step 6 result

Start Julia from the repository root:

```powershell
julia --project=.
```

Load the HSC code, select the checkpoint directory printed by the run, and
load the Step 6 result:

```julia
include("bound_hsc/bound_hsc.jl")

checkpoint_dir =
    "bound_hsc/checkpoints/run_pdeg-30_grid-24_directions-5_depth-0-6_95ef054545ec"
result = load_hsc_checkpoint(6; checkpoint_dir);
```

The run-specific directory name depends on the search parameters. To list the
available Step 6 checkpoints:

```julia
filter(isfile, [
    joinpath(root, file)
    for (root, _, files) in walkdir("bound_hsc/checkpoints")
    for file in files
    if file == "step6.jls"
])
```

The top-level checkpoint, when present, can instead be loaded with:

```julia
result = load_hsc_checkpoint(6);
```

Use the following fields for a successful result:

```julia
result.found
result.candidate.ξ
result.candidate.region
result.depth

result.proof.certified
result.proof.margin
result.proof.margin_upper
result.proof.Q_integral
result.proof.R2_integral
result.proof.Rnorm
result.proof.volume
result.proof.correction
result.proof.rhs

length(result.boxes)
result.boxes[1]
result.boxes[end]
```

The test is

```julia
result.proof.certified == true
result.proof.margin_upper < 0
```

`margin` is a rigorous interval enclosure for

```text
Q_integral + correction
```

where

```text
correction =
    sqrt(volume) * (eta * Rnorm + (1 + eta) * rho).
```

The certificate succeeds only when the upper endpoint of the complete margin
interval is strictly negative. A negative midpoint alone would not be enough.
When `eta == rho == 0`, `correction` is zero and `margin == Q_integral`.

`Q_integral` is obtained by enclosing `Q` uniformly on every rational
`PolytopeBox`, multiplying each enclosure by that box's exact area, summing
the resulting intervals, and multiplying by the rigorously enclosed torus
factor `4pi^2`. Subdivision tightens these enclosures; at depth `d`, one
initial box produces `4^d` final boxes.

`R2_integral` similarly encloses the integral of `|Riem|^2`, and `Rnorm` is
formed from the square root of its nonnegative upper enclosure. To print every
final integration box without Julia abbreviating the middle of the array:

```julia
for (i, box) in enumerate(result.boxes)
    println(i, ": ", box)
end
```

For an unsuccessful search, inspect:

```julia
result.found
result.best
result.best_unresolved
```

`best` is the valid candidate with the smallest upper margin found by the beam
search. `best_unresolved` records a candidate for which positivity or another
required interval condition could not be certified on a box.

To inspect checkpoint metadata as well as the stored result:

```julia
using Serialization

checkpoint = open(deserialize, joinpath(checkpoint_dir, "step6.jls"))
checkpoint.context
checkpoint.parameters
checkpoint.result
```

The convenience function `load_hsc_checkpoint` returns only
`checkpoint.result`. The pipeline additionally checks the stored coefficient
hash, `BigFloat` precision, format version, and search parameters before
reusing a checkpoint.
