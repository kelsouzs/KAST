#!/bin/bash

# KAST Setup Script for Linux
# Simple setup script to create and configure the environment

echo "=================================================="
echo " K-talysticFlow Setup Script"
echo "=================================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found!"
    echo "Please install Conda first: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html"
    exit 1
fi

echo "✅ Conda found"
echo ""

# Create conda environment from environment.yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml -y

if [ $? -eq 0 ]; then
    echo "✅ Environment created successfully!"
else
    echo "❌ Failed to create environment"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ktalysticflow"
echo ""
echo "To start using KAST:"
echo "  python main.py"
echo ""
echo "To verify installation:"
echo "  python bin/check_env.py"
echo ""
