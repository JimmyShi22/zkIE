# Lookup key claim registration

- Scheme: lookup-key-claim-registration; code 21773d0e; previous 43a28314.
- The key plan builds the BAT row for the expanded key claim: each source term
  on its physical member table, the public constant in the target, mask terms as
  units on the mask member; a unit test checks the cell coefficient, mask path and
  target.
- workspace 48/852/0/17; focused lookup 6 passed; norm-free 45,080 B sha256
  7a5da6...de4cfb identical; no push.
- Missing: wiring the row into the hidden second BAT session and replacing the
  synthetic key rows with the real round messages. No byte proof or rounding
  claim yet.
