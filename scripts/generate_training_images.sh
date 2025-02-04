# generating training images for english springer
OUTPUT_DIR=""
PROMPT=""
MODEL_PATH=""
NUM_TRAIN_IMAGES=200

python3 generate_training_images.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt "$PROMPT" \
    --mode test \
    --num_train_images $NUM_TRAIN_IMAGES

# inference model
OUTPUT_DIR=""
PROMPT=""
MODEL_PATH=""
NUM_TRAIN_IMAGES=200

python3 generate_training_images.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --prompt "$PROMPT" \
    --mode test \
    --num_train_images $NUM_TRAIN_IMAGES
