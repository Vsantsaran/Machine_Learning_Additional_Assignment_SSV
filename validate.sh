#!/bin/bash

# ============================================================================
# Setup and Validation Script for Cotton Leaf Disease Detection
# ============================================================================

echo "=========================================="
echo "Cotton Leaf Disease Detection - Setup"
echo "=========================================="

# ============================================================================
# CHECK PYTHON VERSION
# ============================================================================

echo ""
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "ERROR: Python 3.8 or higher is required!"
    exit 1
fi
echo "Python version OK"

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies!"
    exit 1
fi
echo "Dependencies installed"

# ============================================================================
# CHECK PYTORCH AND CUDA
# ============================================================================

echo ""
echo "Checking PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch not installed correctly!"
    exit 1
fi
echo "PyTorch installed"

echo ""
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "WARNING: CUDA not available! Training will be very slow on CPU."
else
    echo "CUDA available"
fi

# ============================================================================
# CHECK GPU AVAILABILITY
# ============================================================================

echo ""
echo "Checking GPU information..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    echo "GPU information retrieved"
else
    echo "WARNING: nvidia-smi not found!"
fi

# ============================================================================
# CHECK DATASET STRUCTURE
# ============================================================================

echo ""
echo "Checking dataset structure..."

DATA_ROOT="ds"
CLASSES=("Bacterial_Blight" "Curl_Virus" "Healthy_Leaf" "Herbicide_Growth_Damage" "Leaf_Hopper_Jassids" "Leaf_Redding" "Leaf_Variegation")

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data root directory '$DATA_ROOT' not found!"
    echo "Please create the 'ds' directory and organize your dataset as follows:"
    echo "ds/"
    echo "├── aug_ds/"
    echo "│   ├── Bacterial_Blight/"
    echo "│   ├── Curl_Virus/"
    echo "│   └── ..."
    echo "└── orig_ds/"
    echo "    ├── Bacterial_Blight/"
    echo "    ├── Curl_Virus/"
    echo "    └── ..."
    exit 1
fi
echo "Data root directory found"

# Check augmented dataset
if [ ! -d "$DATA_ROOT/aug_ds" ]; then
    echo "WARNING: Augmented dataset directory '$DATA_ROOT/aug_ds' not found!"
else
    echo "Augmented dataset directory found"

    for class in "${CLASSES[@]}"; do
        if [ -d "$DATA_ROOT/aug_ds/$class" ]; then
            count=$(find "$DATA_ROOT/aug_ds/$class" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
            echo "  - $class: $count images"
        else
            echo "  WARNING: Class directory '$class' not found in aug_ds"
        fi
    done
fi

# Check original dataset
echo ""
if [ ! -d "$DATA_ROOT/orig_ds" ]; then
    echo "WARNING: Original dataset directory '$DATA_ROOT/orig_ds' not found!"
else
    echo "Original dataset directory found"

    for class in "${CLASSES[@]}"; do
        if [ -d "$DATA_ROOT/orig_ds/$class" ]; then
            count=$(find "$DATA_ROOT/orig_ds/$class" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
            echo "  - $class: $count images"
        else
            echo "  WARNING: Class directory '$class' not found in orig_ds"
        fi
    done
fi

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================

echo ""
echo "Creating output directories..."
mkdir -p logs
mkdir -p checkpoints
echo "Output directories created"

# ============================================================================
# MAKE SCRIPTS EXECUTABLE
# ============================================================================

echo ""
echo "Making training scripts executable..."
chmod +x train_cluster.sh
chmod +x train_direct.sh
echo "Scripts are now executable"

# ============================================================================
# VERIFY TRAINING SCRIPT
# ============================================================================

echo ""
echo "Verifying training script..."
if [ ! -f "mobileVIT.py" ]; then
    echo "ERROR: mobileVIT.py not found!"
    exit 1
fi
echo "Training script found"

# ============================================================================
# FINAL CHECK
# ============================================================================

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. For cluster training: sbatch train_cluster.sh"
echo "2. For local training: ./train_direct.sh"
echo "3. To customize parameters, edit the training scripts"
echo ""
echo "For monitoring:"
echo "  - Training log: tail -f logs/TIMESTAMP/out.log"
echo "  - Error log: tail -f logs/TIMESTAMP/error.log"
echo ""
echo "=========================================="
