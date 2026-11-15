#!/bin/bash
set -u
cd /data/zkIE
CUR=$(git rev-parse --short=8 HEAD)
PREV=9b05474e
TS=$(date +%Y%m%d-%H%M)
D=docs/benchmarks/$TS-raw-dedup-$CUR-$PREV
BASE=/data/raw-baseline-zkie
REG16=docs/benchmarks/20261114-0637-model-table-instance-d1824d49-a5de37d8/smoke/gpt2-seq16/commit
REG512=docs/benchmarks/20261114-0637-model-table-instance-d1824d49-a5de37d8/smoke/gpt2-seq512/commit
mkdir -p "$D"
echo "TRACE $D" | tee "$D/run-summary.txt"

run_one() {
  side=$1
  case_name=$2
  bin=$3
  policy=$4
  case_file=$5
  input=$6
  model_id=$7
  commit_policy=$8
  commit_dir=$9
  run=$D/$side/$case_name
  rm -rf "$run"
  mkdir -p "$run"
  printf 'profile = "case"\n' > "$run/parameters.toml"
  args=(run --commit-policy "$commit_policy")
  if [ "$commit_policy" = "reuse" ]; then
    args+=(--commit-dir "$commit_dir")
  fi
  args+=(--case "$case_file" --assets-manifest "$case_file"
    --parameters "$run/parameters.toml" --input "$input"
    --model-id "$model_id" --policy "$policy" --run-dir "$run")
  start=$(date +%s%3N)
  ZKIE_CASE_ID="$case_name" timeout 3600 "$bin" "${args[@]}" > "$D/$side-$case_name.log" 2>&1
  code=$?
  end=$(date +%s%3N)
  echo "$side $case_name EXIT $code WALL_MS $((end-start))" | tee -a "$D/run-summary.txt"
  cat "$run/commit-run/result.json" 2>/dev/null | tee -a "$D/run-summary.txt"; echo | tee -a "$D/run-summary.txt"
  cat "$run/prove/result.json" 2>/dev/null | tee -a "$D/run-summary.txt"; echo | tee -a "$D/run-summary.txt"
  cat "$run/verify/result.json" 2>/dev/null | tee -a "$D/run-summary.txt"; echo | tee -a "$D/run-summary.txt"
  stat -c "%s %n" "$run/proof/proof.bin" 2>/dev/null | tee -a "$D/run-summary.txt"
}

GEMMA16=39b607c9f6156a8df155ca1a97e1fa70f0adf74730cc670ec67b5c8fd6410e34
GEMMA512=d5114631bccfb6e029cfeb1ad2c7b2a65d8a23827551f69424b07f87cbe332ec
TIMESFM=c87a951844e832fcfc8022688ab86f6c7f634c86874a6b6758c1c88104460742
GPT2_16=c2971c0179463bf12aecaefc3488417228008c1314e6905c49a55078242c1268
GPT2_512=6d35b05739f33e2d2c969f55fafd79c22a96d2264683a4d9a63ecb842438d271

run_one after gemma3-seq16 target/release/zkie models/gemma3/case.policy \
  models/gemma3/case.toml /data/zkie-assets/cases/gemma3/tokens_16_i32.bin "$GEMMA16" create ""
run_one before gemma3-seq16 "$BASE" models/gemma3/case.policy \
  models/gemma3/case.toml /data/zkie-assets/cases/gemma3/tokens_16_i32.bin "$GEMMA16" reuse "$D/after/gemma3-seq16/commit"
run_one after gemma3-seq512 target/release/zkie models/gemma3/case.policy \
  models/gemma3/case.toml /data/zkie-assets/cases/gemma3/tokens_512_i32.bin "$GEMMA512" create ""
run_one before gemma3-seq512 "$BASE" models/gemma3/case.policy \
  models/gemma3/case.toml /data/zkie-assets/cases/gemma3/tokens_512_i32.bin "$GEMMA512" reuse "$D/after/gemma3-seq512/commit"
run_one after timesfm-16patch target/release/zkie models/timesfm/case.policy \
  models/timesfm/case.toml /data/zkie-assets/cases/timesfm/input_ts_i32.bin "$TIMESFM" create ""
run_one before timesfm-16patch "$BASE" models/timesfm/case.policy \
  models/timesfm/case.toml /data/zkie-assets/cases/timesfm/input_ts_i32.bin "$TIMESFM" reuse "$D/after/timesfm-16patch/commit"

run_one before gpt2-seq16 "$BASE" models/gpt2/case.policy \
  models/gpt2/case.toml /data/zkie-assets/cases/gpt2/tokens_16_i32.bin "$GPT2_16" reuse "$REG16"
run_one after gpt2-seq16 target/release/zkie models/gpt2/case.policy \
  models/gpt2/case.toml /data/zkie-assets/cases/gpt2/tokens_16_i32.bin "$GPT2_16" reuse "$REG16"
run_one before gpt2-seq512 "$BASE" models/gpt2/case.policy \
  models/gpt2/case.toml /data/zkie-assets/cases/gpt2/tokens_512_i32.bin "$GPT2_512" reuse "$REG512"
run_one after gpt2-seq512 target/release/zkie models/gpt2/case.policy \
  models/gpt2/case.toml /data/zkie-assets/cases/gpt2/tokens_512_i32.bin "$GPT2_512" reuse "$REG512"

echo "DONE $D" | tee -a "$D/run-summary.txt"
