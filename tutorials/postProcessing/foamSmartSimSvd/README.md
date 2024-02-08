# Computing the SVD of OpenFOAM fields

This folder contains several workflows to compute the singular value decomposition (SVD) of OpenFOAM field data:

1. **foam-smartsim-svd.ipynb**: works with pre-computed snapshots; requires the *foamSmartSimSvd* utility; requires running the *cavity* test case before starting the notebook
2. **foam-smartsim-svd-db-api.ipynb**: same as *foam-smartsim-svd.ipynb* but with alternative SmartRedis API usage
3. **partitioned-svd-cylinder.ipynb**: employs the *fieldsToSmartRedis* function object to write OpenFOAM data directly into SmartRedis; requires the *svdToFoam* utility

## Requirements

For general dependencies, refer to the [README.md](/README.md) in the top-level folder.
To compile all utilities and function objects, go to the top-level folder and run:
```
source SOURCEME.sh
```

## Executing the notebooks

All notebooks are executed locally and expect port 8000 to be free. To run workflows 1. or 2., first go to the *cavity* folder and execute the *Allrun* script. To run any of the notebooks, open Jupyter Lab and click on *Run->Run all cells*. To reset workflow 3., use the *Allclean* script located in this folder.

## Additional information

The notebook *foam-smartsim-svd.ipynb* and the corresponding OpenFOAM application *foamSmartSimSvd* use the [standard SmartRedis API](https://www.craylabs.org/docs/smartredis.html), while the notebook *foam-smartsim-svd-db-api.ipynb* and the corresponding application *foamSmartSimSvdDBAPI* use the OpenFOAM-SmartRedis database API from [src/smartRedis](/src/smartRedis) to interact with the SmartRedis database. 