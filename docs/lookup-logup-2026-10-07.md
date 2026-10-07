# Lookup LogUp benchmark (2026-10-07)

TimesFM-scale generic root-bound lookup LogUp (`extension_lookup_logup`),
run through the real IR (`Store` + `Op::Lookup`), at 32-bit testing security
(`SECURITY_LEVEL=32 POW_BUDGET=10`, matching the existing `new_testing`
model benches), on rows=2^15 / table=2^20:

```
prove 2.905 s / verify 0.028 s / accepted
phase: initial_commit 0.174 s, derived_inverse_commit 0.176 s,
       relation_prove 0.925 s, terminal_open 0.610 s
rows_whir    open=14 (0.050 s)  verify=14 (0.011 s)
entries_whir open=14 (0.519 s)  verify=14 (0.017 s)
```

Conclusion: the lookup LogUp primitive is cheap at model scale. The dominant
cost under 90-bit `Whir::new` is the 32-bit PoW grind, not the lookup relation
(already noted in `reedweave-assessment-2026-10-07.md`).
