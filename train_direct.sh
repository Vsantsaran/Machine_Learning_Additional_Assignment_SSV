#!/bin/bash

# ============================================================================
# Cotton Leaf Disease Detection - Direct Training Script (No SLURM)
# ============================================================================

echo "=========================================================================="
echo "Cotton Leaf Disease Detection - Optimized MobileViT"
echo "=========================================================================="
echo "Start Time: $(date)"
echo "=========================================================================="

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset path (MODIFY THIS TO YOUR DATASET PATH)
DATA_ROOT="ds"

# Check if data path exists
if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: Data directory does not exist: ${DATA_ROOT}"
    echo "Please modify DATA_ROOT in this script to point to your dataset"
    exit 1
fi

# Output directories
EXPERIMENT_NAME="cotton_mobilevit_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./experiments/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
LOG_DIR="${OUTPUT_DIR}/logs"

# Create directories
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output Directory: ${OUTPUT_DIR}"

# ============================================================================
# GPU CONFIGURATION - Using GPUs 4, 5, 6, 7
# ============================================================================
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

echo "Using GPUs: 4, 5, 6, 7"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ============================================================================
# PYTHON ENVIRONMENT CHECK
# ============================================================================
echo "=========================================================================="
echo "Environment Check"
echo "=========================================================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

echo "Python: $(python3 --version)"

# Check PyTorch
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch not installed"
    exit 1
fi

echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"

# Check required packages
for pkg in torchvision PIL numpy sklearn; do
    python3 -c "import ${pkg}" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "ERROR: ${pkg} not installed"
        echo "Please install: pip install ${pkg}"
        exit 1
    fi
done

echo "All dependencies found!"
echo "=========================================================================="

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Model Architecture
IMAGE_SIZE=224
WIDTH_MULTIPLIER=1.0
EXPANSION_RATIO=4
TRANSFORMER_DEPTH=2
NUM_HEADS=4
PATCH_SIZE=2
DROPOUT=0.3
STOCHASTIC_DEPTH=0.1

# Training
EPOCHS=10
BATCH_SIZE=16
ACCUMULATION_STEPS=8
LR=0.001
LR_MIN=1e-6
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=3

# Scheduler
T_0=5
T_MULT=2

# Data
TRAIN_RATIO=0.7
VAL_RATIO=0.2
TEST_RATIO=0.1

# Other
SEED=42
NUM_WORKERS=8
SAVE_FREQ=2

# ============================================================================
# TRAINING
# ============================================================================

echo "=========================================================================="
echo "Starting Training"
echo "=========================================================================="
echo "Configuration:"
echo "  Data Root: ${DATA_ROOT}"
echo "  Image Size: ${IMAGE_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size (per GPU): ${BATCH_SIZE}"
echo "  Accumulation Steps: ${ACCUMULATION_STEPS}"
echo "  Effective Batch Size: $((BATCH_SIZE * 4 * ACCUMULATION_STEPS))"
echo "  Learning Rate: ${LR}"
echo "  Weight Decay: ${WEIGHT_DECAY}"
echo "=========================================================================="

# Save configuration
cat > ${OUTPUT_DIR}/run_config.txt << EOF
Experiment: ${EXPERIMENT_NAME}
Start Time: $(date)
Data Root: ${DATA_ROOT}
Output Directory: ${OUTPUT_DIR}

Model Configuration:
  Image Size: ${IMAGE_SIZE}
  Width Multiplier: ${WIDTH_MULTIPLIER}
  Expansion Ratio: ${EXPANSION_RATIO}
  Transformer Depth: ${TRANSFORMER_DEPTH}
  Number of Heads: ${NUM_HEADS}
  Patch Size: ${PATCH_SIZE}
  Dropout: ${DROPOUT}
  Stochastic Depth: ${STOCHASTIC_DEPTH}

Training Configuration:
  Epochs: ${EPOCHS}
  Batch Size (per GPU): ${BATCH_SIZE}
  Accumulation Steps: ${ACCUMULATION_STEPS}
  Effective Batch Size: $((BATCH_SIZE * 4 * ACCUMULATION_STEPS))
  Initial Learning Rate: ${LR}
  Minimum Learning Rate: ${LR_MIN}
  Weight Decay: ${WEIGHT_DECAY}
  Warmup Epochs: ${WARMUP_EPOCHS}
  T_0: ${T_0}
  T_mult: ${T_MULT}

Data Configuration:
  Train Ratio: ${TRAIN_RATIO}
  Val Ratio: ${VAL_RATIO}
  Test Ratio: ${TEST_RATIO}

Other:
  Random Seed: ${SEED}
  Number of Workers: ${NUM_WORKERS}
  Save Frequency: ${SAVE_FREQ}
EOF

echo "Starting training with nohup..."
# Run training
nohup python3 mobileVIT.py \
    --data_root ${DATA_ROOT} \
    --image_size ${IMAGE_SIZE} \
    --train_ratio ${TRAIN_RATIO} \
    --val_ratio ${VAL_RATIO} \
    --test_ratio ${TEST_RATIO} \
    --width_multiplier ${WIDTH_MULTIPLIER} \
    --expansion_ratio ${EXPANSION_RATIO} \
    --transformer_depth ${TRANSFORMER_DEPTH} \
    --num_heads ${NUM_HEADS} \
    --patch_size ${PATCH_SIZE} \
    --patch_sizes 2 4 8 \
    --dropout ${DROPOUT} \
    --stochastic_depth ${STOCHASTIC_DEPTH} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --accumulation_steps ${ACCUMULATION_STEPS} \
    --lr ${LR} \
    --lr_min ${LR_MIN} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --T_0 ${T_0} \
    --T_mult ${T_MULT} \
    --use_mixup \
    --use_cutmix \
    --use_focal_loss \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --log_dir ${LOG_DIR} \
    --save_freq ${SAVE_FREQ} \
    --seed ${SEED} \
    --num_workers ${NUM_WORKERS}
    > ${LOG_DIR}/out.log 2> ${LOG_DIR}/err.log &

echo $! > ${OUTPUT_DIR}/pid.txt
echo "Training started in background. PID saved to ${OUTPUT_DIR}/pid.txt"

EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# POST-TRAINING
# ============================================================================

echo "=========================================================================="
echo "Training Completed"
echo "=========================================================================="
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "SUCCESS! Training completed successfully."
    echo ""
    echo "Results Location:"
    echo "  ${OUTPUT_DIR}"
    echo ""
    echo "Generated Files:"
    echo "  - Best Model: ${CHECKPOINT_DIR}/best_model.pth"
    echo "  - Training Log: ${LOG_DIR}/out.log"
    echo "  - Error Log: ${LOG_DIR}/error.log"
    echo "  - Training Metrics: ${LOG_DIR}/training_metrics.json"
    echo "  - Test Metrics: ${LOG_DIR}/test_metrics.json"
    echo "  - Classification Report: ${LOG_DIR}/classification_report.txt"
    echo "  - Confusion Matrix: ${LOG_DIR}/confusion_matrix.npy"
    echo ""
    echo "To view training progress:"
    echo "  cat ${LOG_DIR}/out.log"
    echo ""
    echo "To view test results:"
    echo "  cat ${LOG_DIR}/classification_report.txt"
    echo ""
else
    echo ""
    echo "ERROR! Training failed with exit code: ${EXIT_CODE}"
    echo ""
    echo "Check logs:"
    echo "  Error Log: ${LOG_DIR}/error.log"
    echo "  Training Output: ${OUTPUT_DIR}/training_output.log"
    echo ""
fi

echo "=========================================================================="

exit ${EXIT_CODE}
