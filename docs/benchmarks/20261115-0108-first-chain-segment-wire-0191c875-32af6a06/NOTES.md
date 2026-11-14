# first-chain-segment-wire

- Scheme: first-chain-segment-wire (B2 request chain, TURN 540-C1 V1: the segment
  runs inside the first chain at the ruled position).
- Purpose: execute token/stack/merge on the first chain before the LIN first W
  root and the seal, with the section message fixed in the first message.
- Change: prove_first/verify_first take Option<&ChainInputs>; prove_first runs
  the segment after the reduction root absorb, stores the section in
  HiddenSendMessages.request before the W root/sample and the seal; verify_first
  replays it at the same point. Inner entries gate presence: plan section
  without inputs or inputs without a section are refused. Existing entries pass
  None.
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; merged
  tests 110 passed; workspace numbers in workspace.log.
- Conclusions: the first-chain segment has an execution point and a transcript
  fixation in the ordered position; no silent omission is possible.
- Open items: a caller input bundle matching the session plan (reduction/claims/
  masks) to exercise the path positively, the relation rows wiring, negatives,
  W layout freeze, batch report.
