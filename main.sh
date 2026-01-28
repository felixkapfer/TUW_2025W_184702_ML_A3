#!/bin/bash
# ============================================================================
# Main execution script for ML Exercise 3.2 - Deep Learning for Image Tasks
# 
# This script runs all approaches:
#   1. CNN (cnn.py)
#   2. ViT (vit.py)
#   3. Shallow (shallow.py)
#
# Usage:
#   chmod +x main.sh
#   ./main.sh                    # Run all with default settings
#   ./main.sh --dry-run          # Quick test run
#   ./main.sh --dataset cifar10  # Specify dataset
# ============================================================================

set -e  # Exit on error

# Default settings
DATASET="fashionmnist"
DRY_RUN=""
OUTPUT_DIR="./outputs"
GRID_SEARCH=""
NO_PLOTS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --grid-search)
            GRID_SEARCH="--grid-search"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-plots)
            NO_PLOTS="--no-plots"
            shift
            ;;
        -h|--help)
            echo "Usage: ./main.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET    Dataset to use: fashionmnist or cifar10 (default: fashionmnist)"
            echo "  --dry-run            Quick test with small data subset"
            echo "  --grid-search        Run grid search for hyperparameter tuning"
            echo "  --output-dir DIR     Output directory (default: ./outputs)"
            echo "  --no-plots           Skip visualization generation"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print header
echo "============================================================"
echo "ML Exercise 3.2 - Deep Learning for Image Tasks"
echo "============================================================"
echo "Dataset:     $DATASET"
echo "Output dir:  $OUTPUT_DIR"
echo "Dry run:     ${DRY_RUN:-No}"
echo "Grid search: ${GRID_SEARCH:-No}"
echo "============================================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# ----------------------------------------------------------------------------
# Run CNN approach
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[1/2] Running CNN approach..."
echo "============================================================"
python cnn.py --dataset "$DATASET" --output-dir "$OUTPUT_DIR" $DRY_RUN $GRID_SEARCH $NO_PLOTS

# ----------------------------------------------------------------------------
# Run ViT approach
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[2/2] Running ViT approach..."
echo "============================================================"
python vit.py --dataset "$DATASET" --output-dir "$OUTPUT_DIR" $DRY_RUN $GRID_SEARCH $NO_PLOTS

# ----------------------------------------------------------------------------
# Run other approaches (add your scripts here)
# ----------------------------------------------------------------------------

# Example for approach 3 (uncomment and modify):
# echo ""
# echo "============================================================"
# echo "[3/3] Running Traditional ML approach..."
# echo "============================================================"
# python traditional.py --dataset "$DATASET" --output-dir "$OUTPUT_DIR" $DRY_RUN

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "============================================================"
echo "Total runtime: ${MINUTES}m ${SECONDS}s"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  No JSON files found"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  No CSV files found"
ls -la "$OUTPUT_DIR"/*.png 2>/dev/null || echo "  No PNG files found"
echo "============================================================"
