# HSC checkpoints

Running `prove_proposition_5_6` checks this directory before steps 2, 3, and 4.
An existing `stepN.jls` is loaded; otherwise the step is computed and its result
is saved here. Step 5 saves the generated candidate regions (the initial boxes),
and step 6 saves the complete compatible-U-and-V search result. Each checkpoint
records the SHA-256 hash of the rational CSV,
the `BigFloat` precision, the step number, and step-specific parameters. A stale
or legacy checkpoint is ignored, recomputed, and atomically replaced. The
generated `.jls` files are intentionally ignored by Git.

Reload one from Julia after including `bound_hsc/bound_hsc.jl`, for example:

```julia
step4 = load_hsc_checkpoint(4)
initial_boxes = load_hsc_checkpoint(5)
search_result = load_hsc_checkpoint(6)
```

The pipeline itself supplies expected metadata when loading, so the unchecked
form above is intended only for manual inspection.
