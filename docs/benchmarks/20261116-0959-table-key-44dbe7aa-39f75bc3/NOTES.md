# table-key: real-instance evidence attempt and range finding

## Scheme and provenance

- Scheme: table-key. Code: `44dbe7aa` (evidence driver) on `39f75bc3` (public
  verifier), after `c4ab23af` (candidate instance) and `b1cbac0b` (plan
  derivation). Production range declaration remains R18; this trace records a
  diagnostic attempt and a parameter finding, not an accepted run.
- Purpose: run the complete key chain over the registered candidate instance
  under the release profile, verify it through the independent public entry,
  and record proof bytes, wall time and peak memory.

## Range finding

The registered L0_k table's real i32 values span `-186360 .. 183214`. The
approved R18 declaration covers `-131072 .. 131071`, so 48 of 1,048,576
entries (0.0046%) lie outside it. `range_multiplicities` correctly refuses
such keys (`BoundViolation`); no proof was produced. This is the mechanism
working as specified, and it means the declaration does not cover the real
asset.

The next approved candidate width, R19 (`d=19`, offset `262144`,
`bit_entries=2^24`, equal to the approved `max_bit_entries`), covers the whole
table. R20 would exceed the approved multiplicity-message limit. Switching
the production declaration from R18 to R19 changes the public range
declaration and plan derivation; the table instance itself is unchanged. The
production configuration was not changed here.

## Diagnostic runs

| root | declaration | outcome | ledger |
|---|---|---|---|
| `601d921c…` (registered candidate) | R18 | refused at multiplicity: 48 keys outside the range | untouched (0) |
| `601d921c…` (registered candidate) | R19 | first attempt charged the batch and failed later | spent (52); retry refused |
| `r19-a…` | R19 | plan derives; first part reaches the key group proof; failed | spent (52) |
| `r19-b…` | R19 | same, after finer first-part probes | spent (52) |
| `r19-c…` | R19 | same, after group/proximity probes | spent (52) |

The first R19 attempt over the registered candidate root charged its
52-column durable budget before the failure. A retry on that root is refused;
the count is never refunded. Later attempts used fresh diagnostic roots in
this trace; each charged its own 52 columns. Old instances and ledgers are
untouched.

With R19 the plan derives with shared/own main code `{18, 16384, 130}`, mask
code `{18, 256, 66}`, `3/3` stages and BASE positions `66/53`. The key first
part completes the index mask, lookup, member, W values/commit, Eval row and
group members/samples/plain rows/fold/IOPP derivation. The failure is inside
the multi-stage IOPP proof of the key group, after `proximity_prove` prints
the FOLD output and before BASE: `ShapeMismatch` after about 128 s, at first
stage `k=10`, `h=4` (group mask rows 9, fresh mask 16 rows), three stages.
The fixture-scale key group uses the same code path and passes; the failure
is scale-dependent. No complete proof bytes, wall time or VmHWM are claimed.

## Open items

1. Confirm the declared range width (R19 or another approved declaration)
   before any production configuration change.
2. Investigate the real-scale multi-stage IOPP shape mismatch in the key
   group; it is independent of the range width and is a scheme-level issue.

Logs: `key-chain-evidence.log` (R18 finding), `key-chain-evidence-r19.log`
(first R19 attempt), the diagnostic `instance/*/commit*.log` and
`commit-run/result.json` files.
