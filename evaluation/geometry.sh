#!/bin/bash

# Your mesh
MESH_DIR="--"

# Ground_truth
GT_DIR="--"

echo "Ground Truth: $GT_DIR"
echo "-----------------------------------------------------"

python evaluate_geometry.py \
    --mesh_dir "$MESH_DIR" \
    --ground_truth_dir "$GT_DIR"

echo "-----------------------------------------------------"
echo "Finished!!"