#!/bin/bash
# Run ColabFold batch predictions for GPCR-Gα complexes
# Supports both Docker (recommended) and local installation
# PARALLELIZED for 3 GPUs

INPUT_DIR=$(readlink -f results/alphafold_validation/fasta_inputs)
MULTIMER_DIR=${INPUT_DIR}/multimer_only
OUTPUT_DIR=$(readlink -f results/alphafold_validation/af_predictions)
CACHE_DIR=$(readlink -f cache_colabfold)

mkdir -p $MULTIMER_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $CACHE_DIR

# Filter only multimer inputs
echo "Preparing inputs..."
find $INPUT_DIR -maxdepth 1 -type f -name '*.fasta' ! -name '*_separate.fasta' -exec cp -u {} $MULTIMER_DIR/ \;

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "Using Docker (ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2)..."
    echo "Parallel execution on GPUs 0, 1, 2"

    # Create temporary directories for split inputs
    for i in 0 1 2; do
        mkdir -p ${MULTIMER_DIR}/gpu_${i}
        rm -f ${MULTIMER_DIR}/gpu_${i}/*
    done

    # Distribute pending tasks
    count=0
    for fasta in ${MULTIMER_DIR}/*.fasta; do
        filename=$(basename "$fasta")
        basename="${filename%.*}"
        
        # Check if already done (simple check for existing PDBs)
        # Adjust pattern based on your output naming
        if ls ${OUTPUT_DIR}/${basename}*rank_001*.pdb 1> /dev/null 2>&1; then
            echo "Skipping $basename (already done)"
            continue
        fi
        
        gpu_id=$((count % 3))
        cp "$fasta" "${MULTIMER_DIR}/gpu_${gpu_id}/"
        ((count++))
    done

    echo "Tasks distributed. Launching containers..."

    # Launch containers
    for i in 0 1 2; do
        GPU_INPUT="${MULTIMER_DIR}/gpu_${i}"
        
        # Skip if empty
        if [ -z "$(ls -A $GPU_INPUT)" ]; then
            echo "No tasks for GPU $i"
            continue
        fi

        echo "Launching GPU $i processing $(ls $GPU_INPUT | wc -l) files..."
        
        docker run --gpus "\"device=${i}\"" --rm \
          -v "${CACHE_DIR}:/cache" \
          -v "${GPU_INPUT}:/input" \
          -v "${OUTPUT_DIR}:/output" \
          ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2 \
          colabfold_batch \
          --model-type alphafold2_multimer_v3 \
          --num-recycle 3 \
          --num-models 5 \
          --amber \
          --use-gpu-relax \
          /input /output &
          
        pids[${i}]=$!
    done

    # Wait for all background processes
    echo "Waiting for all GPU tasks to complete..."
    for pid in ${pids[*]}; do
        wait $pid
    done
    
    echo "All parallel tasks completed."

    # Fix permissions (Docker runs as root)
    echo "Fixing permissions..."
    sudo chown -R $(id -u):$(id -g) $OUTPUT_DIR $CACHE_DIR 2>/dev/null || true
    
    # Cleanup temp dirs
    rm -rf ${MULTIMER_DIR}/gpu_*

else
    echo "Docker not found. Trying local colabfold_batch (Single GPU)..."
    colabfold_batch \
      --model-type alphafold2_multimer_v3 \
      --num-recycle 3 \
      --num-models 5 \
      --amber \
      --use-gpu-relax \
      $MULTIMER_DIR $OUTPUT_DIR
fi
