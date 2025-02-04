MODEL_NAME=""
TRAIN_DIR=""
OUTPUT_DIR=""

accelerate launch train_text_to_image.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$TRAIN_DIR \
        --use_ema \
        --resolution=512 --center_crop --random_flip \
        --train_batch_size=4 \
        --gradient_accumulation_steps=4 \
        --gradient_checkpointing \
        --max_train_steps=200 \
        --learning_rate=1e-05 \
        --max_grad_norm=1 \
        --lr_scheduler="constant" --lr_warmup_steps=0 \
        --output_dir=$OUTPUT_DIR \
        --mixed_precision fp16


