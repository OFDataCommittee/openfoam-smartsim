# Compile the foamSmartSimSvd application 

Refer to [README.md](/README.md) for requirements.

```bash
# source SOURCEME.sh from the root folder of this repo
foamSmartSimSvd> source ../../../SOURCEME.sh
```

# Running the `foamSmartSimSvd` application

The notebooks expect port 8000 to be free on your local machine.

```
foamSmartSimSvd/cavity> ./Allrun.pre # Creates the mesh and decomposes the domain

foamSmartSimSvd> jupyter notebook foam-smartsim-svd.ipynb
# or
foamSmartSimSvd> jupyter notebook foam-smartsim-svd-db-api.ipynb
```

Click on `Run->Run all cells`.

The notebook `foam-smartsim-svd.ipynb` and the corresponding OpenFOAM application `foamSmartSimSvd` use the [standard SmartRedis API](https://www.craylabs.org/docs/smartredis.html), while the notebook `foam-smartsim-svd-db-api.ipynb` and the corresponding application `foamSmartSimSvdDBAPI` use the OpenFOAM-SmartRedis database API from [src/smartRedis](/src/smartRedis) to interact with the SmartRedis database. 
