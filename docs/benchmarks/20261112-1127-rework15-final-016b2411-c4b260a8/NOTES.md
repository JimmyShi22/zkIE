# rework15 final: per-group stage facts and the corrected FOLD/proximity terms

## Scheme and baseline

- Scheme: `rework15-final`.
- Code: document commit `016b2411` (v13); previous round head `c4b260a8`.
- This round changes only `docs/lin-bind-closure-security.md` (v12 -> v13).
  No production code, test path, plan, parameters, commitments, challenge
  order or proof format change was made. The multi-stage IOPP is not
  implemented; the blocker is the missing trusted stage list/per-stage
  parameters for the actual fixture shapes, not a non-power-of-two main code.
- Fixture for the reused performance numbers: `/data/lin-fixture`, formal
  hidden session, GPT-2 output head, 16 positions.

## N1 facts (current tree)

- Main W/closure/shared/own proximity code: `(log_n,ell,r) =
  (17, 65,536, 8,192)`; ell is a power of two. Mask code `(17, 512, 256)`.
- Table code `(17, 65,539, 8,192)` is only the reader/column-opening object
  (`GuShape.index`, `HiddenSendPlan.table_shape`, 603 column draws); it is not
  a BAT/proximity member and does not enter the IOPP shape check. The three
  pads and the sample semantics are unchanged and the layout is not moved.
- Shared proximity members: Gu (2 rows) + Omega (1 row) -> k=2, h=0; own:
  16 W (2 rows) + Omega (1) + Dot X (1) + index mask (2 mask rows) +
  16 slack (2) + 16 bits (128) + 16 map (128) + closure mask (2 mask rows)
  -> main rows 4,162 -> k=13, mask rows 4 -> h=2. Sample answers 1 each;
  W leaf class 1, other message leaf class 2, mask leaf class 3.
- Actual FOLD terms: shared `2k/p^3 = 4/p^3`; own `2k/p^3 = 26/p^3`. The
  earlier `22/28` (k=11/14) figures came from a different plan and are
  withdrawn.
- `lm_head_compiled` has main ell=4,096/r=962, mask 64/256, k=6/5 and three
  stages; it is not the hidden fixture's run plan and cannot be used directly.

## N2 status

- Reference shape conditions: non-last k>=1, h<=k; power-of-two message
  lengths; `ell_i = 2^{k_{i+1}} ell_{i+1}`; next mask room
  `ell_zk_{i+1} 2^{h_{i+1}} >= 2 ell_zk_i + r_i + 2 r_zk_i + answers`; last
  k=h=0.
- Shape-feasible candidates exist for the actual stage-0 shapes (for example
  k'=2 -> ell'=16,384 with next mask ell=4,096 and room 16,384 >= 9,729), but
  the fixture plan has no trusted stage list, per-stage codes, positions,
  samples or work. The production compiled plan is a different shape.
- The trusted planner output for the actual shapes therefore does not exist in
  the tree; choosing those parameters is a plan/parameter change. The M2
  implementation is not claimed.

## N3/N4

- v13 §8.7 gives the c-difference/`Im(A)` statement: different secrets have
  the same affine image only when the c difference lies in `Im(A)`; the block
  covers put the secret-difference directions into the fresh columns. The
  A=0 values are winner/valid/tie and the public statement constants, listed as
  the approved public boundary. SEND/PSC/mask/star/cross-tree openings are
  covered by the same affine maps; a fixed root is an input and a single
  tree's q<=r is not a substitute for the joint root/RO history.
- The table is outside the IOPP; the 3 pads/sample semantics are documented
  and no layout/asset/proof contract change is made.

## Numeric corrections

- Actual main W/closure code: delta=(131072-73727)/131072=0.4375076294;
  t=2 miss=(73727/131072)^2=0.3163976670=2^-1.6601. Mask miss unchanged
  (3.424292663e-5). The rework13 value 0.3164234164 used the table ell=65,539
  and is corrected; the qualitative conclusion (no effective proximity bound
  at the current parameters) is unchanged. The 0.3164/155 figure remains a
  single conditional event, not a demonstrated attack or an overall bound.

## Workspace

`cargo test --workspace`: 48 test binaries, 751 passed, 0 failed, 17 ignored.
The complete log is `workspace-test.txt`.

## Reused seq16x3 numbers

| run | hidden prove ms | hidden verify ms | whole test s | max RSS KiB | proof sha256 | accepts | rejects |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 | 359,802 | 50,538 | 646.80 | 169,394,600 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 2 | 358,777 | 50,224 | 644.51 | 169,315,672 | f8f512c0…3215f5 | 276,343,251 | 0 |
| 3 | 358,806 | 49,829 | 644.82 | 169,404,380 | f8f512c0…3215f5 | 276,343,251 | 0 |

These are the rework10 measurements, reused with provenance; they were not
re-measured because the protocol was not changed.

## Open items

- `GAP`: the multi-stage IOPP is not implemented; the trusted stage
  list/per-stage parameters for the actual shapes are missing. The
  `lm_head_compiled` plan is a different shape.
- `GAP`: the direct FOLD-to-BASE proximity is ineffective at the current
  parameters (main miss 0.3163976670); a single conditional event, not a
  demonstrated attack.
- `UNPROVEN`: kernel coverage, witness-free simulator, concrete-hash RO/CR,
  deployment randomness, exact final-verifier negative.
- No push: origin remains `831e7134`.

## Ledger

- Tests passed: 751 workspace tests, 0 failed, 17 ignored.
- Proof bytes, wall time and VmHWM: not re-measured; reused rework10 figures
  with provenance above.
