# Builds a local test policy for the smoke only: the same case policy with the
# model version raised and the seq16 expected anchor replaced by the test
# instance anchor. It is not the registered production policy.
# The instance anchor is the first 32 bytes of the serialized public anchor
# record inside model-commitment.bin.
ANCHOR=$(xxd -p -l 32 "$1/commit/public/model-commitment.bin" | tr -d '\n')
sed -e 's/^model_version = 1$/model_version = 2/' \
    -e "s/dd08df8bcbba41adea6182709e00adb2f897fde4abc241d1ba09c552fd77ae36/$ANCHOR/g" \
    models/gpt2/case.policy > "$1/policy-v2.toml"
