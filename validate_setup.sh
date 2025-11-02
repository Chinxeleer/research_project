#!/bin/bash

##############################################################################
# Setup Validation Script
##############################################################################
# Run this script BEFORE running experiments to ensure everything is ready
##############################################################################

echo "##############################################################################"
echo "# Financial Forecasting Research - Setup Validation"
echo "##############################################################################"
echo ""

ERRORS=0
WARNINGS=0

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

##############################################################################
# Check 1: Data preprocessing
##############################################################################
echo "Checking data preprocessing..."
if [ -d "dataset/processed_data" ]; then
    NUM_FILES=$(ls dataset/processed_data/*_normalized.csv 2>/dev/null | wc -l)
    if [ $NUM_FILES -gt 0 ]; then
        echo -e "${GREEN}✓${NC} Found $NUM_FILES processed data files"
        echo "  Files:"
        ls dataset/processed_data/*_normalized.csv | sed 's/^/    /'
    else
        echo -e "${RED}✗${NC} No normalized CSV files found in dataset/processed_data/"
        echo "  ACTION: Run 'cd dataset && python prepare_data_eden_method.py'"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}✗${NC} Directory dataset/processed_data/ does not exist"
    echo "  ACTION: Run 'cd dataset && python prepare_data_eden_method.py'"
    ERRORS=$((ERRORS + 1))
fi
echo ""

##############################################################################
# Check 2: Python dependencies
##############################################################################
echo "Checking Python dependencies..."
python -c "import torch; import numpy; import pandas" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Core Python packages available (torch, numpy, pandas)"
else
    echo -e "${YELLOW}⚠${NC} Some Python packages might be missing"
    echo "  ACTION: Run 'pip install -r forecast-research/requirements.txt'"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

##############################################################################
# Check 3: Model files exist
##############################################################################
echo "Checking model implementations..."
REQUIRED_MODELS=("Mamba" "Informer" "Autoformer" "FEDformer" "iTransformer")
MISSING_MODELS=()

for model in "${REQUIRED_MODELS[@]}"; do
    if [ -f "forecast-research/models/${model}.py" ]; then
        echo -e "${GREEN}✓${NC} Model found: ${model}.py"
    else
        echo -e "${RED}✗${NC} Model missing: ${model}.py"
        MISSING_MODELS+=($model)
        ERRORS=$((ERRORS + 1))
    fi
done

if [ ${#MISSING_MODELS[@]} -gt 0 ]; then
    echo "  Missing models: ${MISSING_MODELS[*]}"
fi
echo ""

##############################################################################
# Check 4: Training scripts are executable
##############################################################################
echo "Checking training scripts..."
SCRIPTS=("run_experiments.sh" "run_mamba.sh" "run_informer.sh" "run_autoformer.sh" "run_fedformer.sh" "run_itransformer.sh")
for script in "${SCRIPTS[@]}"; do
    if [ -x "forecast-research/$script" ]; then
        echo -e "${GREEN}✓${NC} Script is executable: $script"
    elif [ -f "forecast-research/$script" ]; then
        echo -e "${YELLOW}⚠${NC} Script exists but not executable: $script"
        echo "  ACTION: Run 'chmod +x forecast-research/$script'"
        WARNINGS=$((WARNINGS + 1))
    else
        echo -e "${RED}✗${NC} Script missing: $script"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

##############################################################################
# Check 5: GPU availability (if using GPU)
##############################################################################
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} GPU detected and available"
    else
        echo -e "${YELLOW}⚠${NC} nvidia-smi command failed"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not found (CPU mode will be used)"
    echo "  Note: Training will be much slower on CPU"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

##############################################################################
# Check 6: Disk space
##############################################################################
echo "Checking disk space..."
AVAILABLE_SPACE=$(df -h . | tail -1 | awk '{print $4}')
echo "  Available space: $AVAILABLE_SPACE"

# Warning if less than 10GB available (rough estimate)
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ $AVAILABLE_GB -lt 10 ]; then
    echo -e "${YELLOW}⚠${NC} Low disk space (< 10GB)"
    echo "  Experiments may generate several GB of results"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}✓${NC} Sufficient disk space available"
fi
echo ""

##############################################################################
# Check 7: Results directories
##############################################################################
echo "Checking results directories..."
DIRS=("forecast-research/checkpoints" "forecast-research/results" "forecast-research/test_results")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        NUM_ITEMS=$(ls -A "$dir" | wc -l)
        if [ $NUM_ITEMS -gt 0 ]; then
            echo -e "${YELLOW}⚠${NC} Directory exists with $NUM_ITEMS items: $dir"
            echo "  Note: Old results may be overwritten"
        else
            echo -e "${GREEN}✓${NC} Directory exists and is empty: $dir"
        fi
    else
        echo "  Directory will be created: $dir"
    fi
done
echo ""

##############################################################################
# Summary
##############################################################################
echo "##############################################################################"
echo "# Validation Summary"
echo "##############################################################################"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Ready to run experiments.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. cd forecast-research"
    echo "  2. ./run_mamba.sh              # Test with Mamba first"
    echo "  3. ./run_experiments.sh        # Run all experiments"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) found${NC}"
    echo "  You can proceed, but review warnings above"
    echo ""
    echo "To proceed:"
    echo "  cd forecast-research && ./run_mamba.sh"
else
    echo -e "${RED}✗ $ERRORS error(s) found${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s) found${NC}"
    fi
    echo ""
    echo "Please fix errors before running experiments."
    echo "See error messages above for required actions."
fi

echo ""
echo "##############################################################################"

exit $ERRORS
