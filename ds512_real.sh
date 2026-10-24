cd /data/jimmyshi/ie
source /data/jimmyshi/.env
docker exec -e HF_TOKEN=$HF_TOKEN -e M=512 mlgen python3 /work/gen_ds_real.py
docker exec -e M=512 mlgen python3 /work/gen_ds_x0.py
docker exec -e M=512 mlgen python3 /work/gen_ds_rope.py
docker exec -e DS_SEQ=512 mlgen python3 /work/scripts/fp_deepseek.py
M=512 cargo run --release -p zkie-models-deepseek-v2-lite --example prove

