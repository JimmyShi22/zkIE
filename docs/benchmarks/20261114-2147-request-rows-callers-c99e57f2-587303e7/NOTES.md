# request-rows-callers

- Scheme: request-rows-callers (B2 production projection consumer, TURN 538
  first part: the request rows reach the second session).
- Code: c99e57f2; baseline: 587303e7 (request-w-member).
- Purpose: build the request rows bundle from the plan member claim and the
  own members and pass it into own_plain_rows at both session callers, with
  the message layouts made explicit and the LIN Dot row sign fixed.
- Change: RequestRows owns its vectors and carries two TableLayouts
  (main_rows, ell). request_rows_bundle resolves the request W and Omega by
  source oracle, masks by the member index oracle through
  entry_virtual_index, flattens the pending weights with pending_row_weights
  and takes psi* = shared_point[..k_col]. request_eval_row scatters eq(psi) on
  the W layout and adds the claim mask units against the target;
  request_dot_row scatters the row weights on the Omega layout and the
  negated eq(psi*) on the W layout, target zero, so the row is
  sum g*Omega - W(psi*) = 0. scatter_message_coefficients now delegates to
  TableLayout::scatter (entry j -> row j mod main_rows, place j div
  main_rows).
- Key numbers: hard gate 45,080 B, sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb
  (hard-gate.log, hard-gate.sha256); workspace 51 suites, 1078 passed,
  0 failed, 19 ignored (workspace.log); focused request_rows and request_w
  tests green.
- Conclusions: the request W oracle, Omega and the member index oracle all
  resolve to own members; the two request rows are built and appended to the
  own plain rows by both callers. Plans without request W are unchanged.
- Open items: a member-carrying session fixture (plan.member + request_w +
  request_reads) with positive and negative runs; shared/second IOPP tests;
  layout freeze; request chain section wiring; at_start ledger; parameter
  re-derivation.
