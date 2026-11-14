# lookup-chain

- Scheme: lookup-chain (the lookup challenges come from the segment chain).
- Based on: the range limbs (code 81432017, trace 952bfd69).
- Purpose: the lookup's challenge source was a standalone 64-bit LCG seeded
  from a value the caller derived; the production path must draw and absorb on
  the enclosing segment's formal chain instead.
- Change:
  - `LookupChallenges` is now a view over `zkie_engine`'s `Chain`: every
    absorbed message (the plan digest, the table identity, the multiplicity
    message root, the mask root, the hiding sum, each round and the end values)
    is its own chain round under a fixed domain tag, and the challenges are the
    chain's extension-field draws. The standalone `u64` state, its rotate/add/
    multiply mixing and the `Fp3::from_u64` draw are gone.
  - `prove_lookup*` and `verify_lookup` take the chain instead of a seed; the
    bundled `prove_lookup` keeps its private commitment seed for the message
    randomness and takes the chain for the transcript.
  - The norm session no longer derives a seed from the chain for its two
    lookups: the prover and the verifier both run the lookup transcript on the
    same chain, so the message absorption and the challenge order match by
    construction. The two seed helpers and their tags are removed.
- Key numbers: workspace 1106 passed / 0 failed (workspace.log); lookup 14
  passed (lookup.log); norm 87 passed (norm.log); hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256).
- Conclusions: the lookup transcript is the segment's chain transcript; a
  changed message changes the later challenges, and the prover and verifier
  share one order. The arithmetic and relations are unchanged.
- Boundaries: this step formalizes the challenge source only; the digit
  lookups over the range object's ZZ message (R10z/R8d parameters, sources,
  padding, multiplicity bits and their consumption by the segment proximity
  group) are not wired yet, so no smallness/range closure is claimed. Not
  pushed.
