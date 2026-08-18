#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo ${PROJECT_ROOT}
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$PROJECT_ROOT/src/openfhe-PRNG-Control/install/lib/OpenFHE" \
      -DBUILD_STATIC=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-g -O3" ..
make -j$(nproc)
