# at-start-ledger-parameters

- Scheme: at-start-ledger-parameters (B2 production projection consumer,
  TURN 524/530/532/536 follow-through on the member plan).
- Purpose: take the at_start mask slots once per run, reject repeated and
  out-of-range slots, re-derive the request-level parameters from the trusted
  shapes, and register the request W as an own object of the planner-checked
  assembly.
- Change: MaskSlotLedger take in prove_first and verify_first (Eval/tilde
  allocation slots and the pending claim's slots, each once);
  HiddenLinPlan::validate rejects a repeated mask slot; RequestParameters
  derives row/column bits, sample form, openings, tree columns and number, mask
  slots and rows from the plan shapes and checks the declared member bits;
  the planner-checked assembly gains the request W own object.
- Tests: request parameters derivation and bit-mismatch negative; member
  fixture negatives extended with a repeated mask slot, a slot past the free
  entries, a changed fold round, a changed activation root and a changed
  reduction coefficient.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb;
  merged tests 104 passed; workspace numbers in workspace.log.
- Conclusions: the request-level member relation now takes its mask slots
  through one runtime ledger and derives every parameter it can from shapes,
  with no frozen numbers; repeated or out-of-range slots and a changed fold
  round, table identity or reduction coefficient are rejected by the actual
  chain.
- Open items: the request chain section (token/stack/merge) is not yet in the
  session transcript: the verifier entry currently resolves mask values through
  a caller closure, and per TURN 536 the terminal relations must be checked
  from the received end values (BAT rows), which is the next structural piece.
  The W layout stays unfrozen per TURN 538.
