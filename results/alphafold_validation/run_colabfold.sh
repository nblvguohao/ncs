#!/bin/bash
# Run ColabFold batch predictions for GPCR-G¦Á complexes
# Requires: pip install colabfold[alphafold]
# Expected runtime: ~2-8 hours per complex on A100

INPUT_DIR=E:\study\thailand\ncs\results\alphafold_validation\fasta_inputs
OUTPUT_DIR=E:\study\thailand\ncs\results\alphafold_validation\af_predictions

colabfold_batch \
  --model-type alphafold2_multimer_v3 \
  --num-recycle 3 \
  --num-models 5 \
  --amber \
  --use-gpu-relax \
  $INPUT_DIR $OUTPUT_DIR
