
export CUDA_VISIBLE_DEVICES=7

FORGET_RATIO=5
METHOD="NPO"
CACHE_PATH="/playpen-shared/haochenz/UMU-Bench-result/ckpts/llava_${METHOD}_forget${FORGET_RATIO}"
OUT_PATH="/playpen-shared/haochenz/UMU-Bench-result/results"
python eval.py \
  --model_id "llava-hf/llava-1.5-7b-hf" \
  --cache_path ${CACHE_PATH} \
  --forget_ratio ${FORGET_RATIO} \
  --data_split_dir "/playpen-shared/haochenz/UMU-Bench" \
  --output_path ${OUT_PATH} \
  --output_file "${METHOD}_forget${FORGET_RATIO}.json"