MODEL_PT="CompVis/stable-diffusion-v1-4"
MODEL_FT="./checkpoint/model_ft"
TV_EDIT_ALPHA=1.75
OUTPUT_DIR="./checkpoint/model_ft_tv-${TV_EDIT_ALPHA}"

python3 save_tv.py \
    --model_pretrained $MODEL_PT \
    --model_finetuned $MODEL_FT \
    --output_dir $OUTPUT_DIR \
    --tv_edit_alpha $TV_EDIT_ALPHA