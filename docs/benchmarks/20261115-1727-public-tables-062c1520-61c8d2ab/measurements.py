import json
import os
import sys

trace = sys.argv[1]
for case in ["gpt2-seq16", "gemma3-seq16", "timesfm-16patch"]:
    for side in ["before", "after"]:
        run = os.path.join(trace, side, case)
        for stage in ["commit-run", "prove", "verify"]:
            path = os.path.join(run, stage, "result.json")
            if not os.path.exists(path):
                continue
            with open(path) as handle:
                result = json.load(handle)
            print(
                case,
                side,
                stage,
                result.get("outcome"),
                result.get("stage_wall_ms"),
                result.get("vmhwm_kib"),
                result.get("proof_bytes"),
            )
        proof = os.path.join(run, "proof", "proof.bin")
        if os.path.exists(proof):
            print(case, side, "proof.bin", os.path.getsize(proof))
