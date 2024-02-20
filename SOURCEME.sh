#!/usr/bin/bash

orig_dit="$(pwd)"

## Compile SmartRedis libs
export _REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$_REPO_ROOT" || exit 1
export FOAM_SMARTREDIS="$_REPO_ROOT/smartredis"
if [ ! -d "$FOAM_SMARTREDIS" ]; then
    git clone https://github.com/CrayLabs/SmartRedis "$FOAM_SMARTREDIS"
else
    cd "$FOAM_SMARTREDIS" || exit 1
    git pull origin develop
    cd "${_REPO_ROOT}" || exit 1
fi
cd "${FOAM_SMARTREDIS}" || exit 1
make lib
cd "$_REPO_ROOT" || exit 1
export SMARTREDIS_INCLUDE=$FOAM_SMARTREDIS/install/include
export SMARTREDIS_LIB=$FOAM_SMARTREDIS/install/lib
export LD_LIBRARY_PATH=$SMARTREDIS_LIB:$LD_LIBRARY_PATH
export FOAM_CODE_TEMPLATES=$_REPO_ROOT/etc/dynamicCode/

## Compile OpenFOAM libs
wmake libso src/smartRedis
wmake libso src/functionObjects
wmake libso src/displacementSmartSimMotionSolver

## Compile OpenFOAM utilities
wmake applications/utilities/foamSmartSimSvd
wmake applications/utilities/foamSmartSimSvdDBAPI
wmake applications/utilities/svdToFoam

cd "$orig_dir" || exit 1
