#!/bin/bash
# Run ColabFold batch predictions for GPCR-Gα complexes
# Supports both Docker (recommended) and local installation

INPUT_DIR=$(readlink -f results/alphafold_validation/fasta_inputs)
MULTIMER_DIR=${INPUT_DIR}/multimer_only
OUTPUT_DIR=$(readlink -f results/alphafold_validation/af_predictions)
CACHE_DIR=$(readlink -f cache_colabfold)

mkdir -p $MULTIMER_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $CACHE_DIR

# Filter only multimer inputs
find $INPUT_DIR -maxdepth 1 -type f -name '*.fasta' ! -name '*_separate.fasta' -exec cp {} $MULTIMER_DIR/ \;

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Using Docker (ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2)..."
    echo "Input: $MULTIMER_DIR"
    echo "Output: $OUTPUT_DIR"
    echo "Cache: $CACHE_DIR"
    
    docker run --gpus all --rm \
      -v "${CACHE_DIR}:/cache" \
      -v "${MULTIMER_DIR}:/input" \
      -v "${OUTPUT_DIR}:/output" \
      ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2 \
      colabfold_batch \
      --model-type alphafold2_multimer_v3 \
      --num-recycle 3 \
      --num-models 5 \
      --amber \
      --use-gpu-relax \
      /input /output
      
    # Fix permissions (Docker runs as root)
    if [ $? -eq 0 ]; then
        echo "Fixing permissions..."
        sudo chown -R $(id -u):$(id -g) $OUTPUT_DIR $CACHE_DIR 2>/dev/null || true
    fi
else
    echo "Docker not found. Trying local colabfold_batch..."
    colabfold_batch \
      --model-type alphafold2_multimer_v3 \
      --num-recycle 3 \
      --num-models 5 \
      --amber \
      --use-gpu-relax \
      $MULTIMER_DIR $OUTPUT_DIR
fi
