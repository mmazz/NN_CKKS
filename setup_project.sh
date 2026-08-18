#!/bin/bash

# setup_project.sh - Script to set up the project structure

set -e

PROJECT_ROOT=$(pwd)

echo "=== Setting up project structure ==="

mkdir src
cd src

# Clonar HEAAN_PRNG si no existe
if [ ! -d "HEAAN-PRNG-Control" ]; then
    echo "Cloning HEAAN-PRNG-Control..."
    git clone git@github.com:mmazz/HEAAN-PRNG-Control.git
else
    echo "HEAAN-PRNG-Control already exists"
fi

# Clonar openfhe-PRNG-Control si no existe
if [ ! -d "openfhe-PRNG-Control" ]; then
    echo "Cloning openfhe-PRNG-Control..."
    git clone git@github.com:mmazz/openfhe-PRNG-Control.git
else
    echo "openfhe-PRNG-Control already exists"
fi

echo ""
echo "=== Building libraries ==="

# Build HEAAN if library doesn't exist
if [ ! -f "HEAAN-PRNG-Control/lib/libHEAAN.a" ]; then
    echo "Building HEAAN library..."
    cd HEAAN-PRNG-Control
    make
    cd ..
else
    echo "HEAAN library already built (libHEAAN.a found)"
fi

# Build OpenFHE if library doesn't exist
if [ ! -f "openfhe-PRNG-Control/install/lib/libOPENFHEpke.so.1" ]; then
    echo "Building OpenFHE library..."
    cd openfhe-PRNG-Control
    make
    cd ..
else
    echo "OpenFHE library already built (libOPENFHEpke.so.1 found)"
fi

cd "$PROJECT_ROOT"
if [ ! -f "NN_confing/data" ]; then
    echo "Building NN weights..."
    cd NN_config
    ./nn_training.sh
    cd ..
else
    echo "NN Weights already calculated"
fi


echo "=== Setup completed successfully ==="

