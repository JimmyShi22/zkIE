# ReedWeave assessment for zkIE (2026-10-07)

ReedWeave (eprint 2026/2147) is a faster RS polynomial commitment: it
interleaves a degree-d polynomial into m components committed over a smaller
domain, and its first "row combination" is exactly an arity-m FRI fold, avoiding
the per-round domain-shifting / re-encoding / extra FFTs that WHIR pays.
Measured in the paper: ReedWeave prover ~0.59 s vs WHIR ~3.44 s at rate 1/4,
i.e. ~5-6x faster commit/open.

## Why it does NOT move the needle for zkIE's verify time

Our GPT-2 512 (13 shards) committed bench measured (after the parallel-verify
fix in the batch-opening note):

```
WHIR open_stats   = (70, 0.53 s)   # all openings, whole proof
WHIR verify_stats = (70, 0.03 s)
verify total      = 22.14 s
```

So the entire WHIR open+verify is ~0.56 s of 22.14 s (~2.5%). Even if ReedWeave
made open+verify 10x faster, it saves ~0.5 s - nothing against the ~20 s the
verifier spends recomputing the forward pass once.

ReedWeave (like batch-opening) optimizes Layer 2 (the PCS unit cost). The
zkIE verify bottleneck is Layer 1: the verifier recomputes the forward pass.

## Where ReedWeave *would* matter

- In a future "committed + real security parameter" configuration (not
  `new_testing`), where commit/open is large and GPU-bound. There ReedWeave's
  ~5-6x commit/open win is real, but it is still a Layer-2 win.

## Conclusion

- Priority 1 (~20x): make the verifier claim-driven - open boundary/weight/I-O
  anchors instead of recomputing the forward. Cuts verify ~20 s -> ~seconds.
- Priority 3 (<1 s): ReedWeave / batch-opening - Layer-2 only.

Doing ReedWeave before the claim-driven verifier optimises the ~2.5% tail while
the ~90% head (the forward recompute) is untouched.

## Update: real security parameters flip the conclusion - 2026-10-07

The "WHIR open ~0.5 s / ~2.5%" premise is at `new_testing` (pow_bits 10). At
`Whir::new(19)` (security 90, pow_bits 32) the open's PoW grind is 2^32, and
GPT-2 512 prove did not finish within ~2 h (testing: ~36 s). Under real
parameters the WHIR open IS the dominant cost - exactly what ReedWeave
optimises. ReedWeave moves from "Priority 3 (tail)" to "Priority 1" once real
security parameters are used.
