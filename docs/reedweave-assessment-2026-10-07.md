# ReedWeave assessment for zkIE (2026-10-07)

ReedWeave (eprint 2026/2147) is a faster RS polynomial commitment: it
interleaves a degree-d polynomial into m components committed over a smaller
domain, and its first "row combination" is exactly an arity-m FRI fold, avoiding
the per-round domain-shifting / re-encoding / extra FFTs that WHIR pays.
Measured in the paper: ReedWeave prover ~0.59 s vs WHIR ~3.44 s at rate 1/4,
i.e. ~5-6x faster commit/open.

## Why it does NOT move the needle for zkIE's verify time

Our GPT-2 512 (13 shards) committed bench measured:

```
WHIR open_stats  = (70, 0.51 s)   # all openings, whole proof
WHIR verify_stats= (70, 0.03 s)
verify total     = 184 s
```

So the entire WHIR open+verify is ~0.54 s of 184 s (0.3%). Even if ReedWeave
made open+verify 10x faster, it saves ~0.5 s — nothing against the ~183 s the
verifier spends recomputing the forward pass.

ReedWeave (like batch-opening) optimizes Layer 2 (the PCS unit cost). The
zkIE verify bottleneck is Layer 1: the verifier recomputes the forward pass.

## Where ReedWeave *would* matter

- In a future "committed + real security parameter" configuration (not
  `new_testing`), where commit/open is large and GPU-bound. There ReedWeave's
  ~5-6x commit/open win is real, but it is still a Layer-2 win.

## Conclusion

- Priority 1 (73x): make the verifier claim-driven — open boundary/weight/I-O
  anchors instead of recomputing the forward. Cuts verify ~183 s -> ~seconds.
- Priority 3 (<2.5 s): ReedWeave / batch-opening — Layer-2 only.

Doing ReedWeave before the claim-driven verifier optimises the 0.3% tail while
the 99.7% head is untouched.
