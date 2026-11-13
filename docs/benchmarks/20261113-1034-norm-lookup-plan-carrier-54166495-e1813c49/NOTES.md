# Norm lookup plan carrier

- Scheme: norm-lookup-plan-carrier; code 54166495; previous e1813c49.
- NormPlan now carries optional NormLookupPlan (lookup plan + committed bits/mask
  plans + key map); validation checks the counts and the plan digest binds the
  lookup and key map; version rises to 5; existing plans carry no lookup.
- workspace 48/853/0/17; focused norm 27 passed; norm-free 45,080 B sha256
  7a5da6...de4cfb identical; no push.
- Missing: first/second message carriers, BAT registration in hidden second and
  the real round key rows. No byte proof or rounding claim yet.
