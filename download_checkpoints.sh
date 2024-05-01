#!/bin/bash

# Define the base URL of the Hugging Face model repository
BASE_URL="https://huggingface.co/Jha-Pranav/blm-lab/resolve/main/"

# Get the model names from command line arguments
IFS=',' read -ra MODEL_ARRAY <<< "$1"

# Iterate over each model name
for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
    # Construct the complete checkpoint URL
    CHECKPOINT_URL="${BASE_URL}${MODEL_NAME}/last.bin?download=true"
    echo $CHECKPOINT_URL

    # Directory where the checkpoint will be saved
    CHECKPOINT_DIR="checkpoint/${MODEL_NAME}"

    # Create the directory if it doesn't exist
    mkdir -p "$CHECKPOINT_DIR"

    # Download the checkpoint using wget
    wget -O "$CHECKPOINT_DIR/last.ckpt" "$CHECKPOINT_URL"
done
# ./download_checkpoints.sh "blm-fine-tuned-imdb-v2,blm-fine-tuned-imdb-v3,blm-fine-tuned-imdb,blm-instruct-maths-problem,blm-instruct-maths-problem,blm-instruct-v2,blm-instruct-v3,blm-medium-instruct-tinnystories,blm-medium"