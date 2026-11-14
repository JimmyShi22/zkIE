# chain-mask-plan

- Scheme: chain-mask-plan (B2 request chain, TURN 540-C1 V2: the sub-proof mask
  table is its own committed object, distinct from the request index-mask).
- Purpose: fix the owner and the slot-to-entry layout of the token, stack and
  merge mask slots before the segment rows are bound.
- Change: request_chain::ChainMaskPlan (oracle, group, mask code, mask rows,
  entries, runtime tree) with fault checks and an entry layout mapping a slot
  to the group mask table virtual index; chain_mask_entries returns the owner
  binding and refuses duplicate slots and slots past the free entries (pad
  excluded).
- Key numbers: hard gate 45,080 B sha256
  7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb; focused
  test green; workspace numbers in workspace.log.
- Conclusions: the sub-proof mask slots have one owner, one layout and one
  ledger; slot numbers are not shared with the index-mask object; a plan whose
  tree does not carry exactly its mask rows or whose entries exceed the free
  rows is refused.
- Open items: the table's commitment message and sampling in the first chain,
  the capacity/read ledger, the row bindings of the four remaining relations,
  then the first-chain segment order and the transcript/version.
