# Compile the foamSmartSimSvd application 

Refer to [README.md](/README.md) for requirements.

```bash
# source SOURCEME.sh from the root folder of this repo
foamSmartSimSvd> source ../../../SOURCEME.sh
```

# Running the `foamSmartSimSvd` application

The notebooks expect port 8000 to be free on your local machine.

```
foamSmartSimSvd> jupyter notebook foam-smartsim-svd.ipynb
# or
foamSmartSimSvd> jupyter notebook foam-smartsim-svd-db-api.ipynb
```

Click on `Run->Run all cells`.

The only difference between `foamSmartSimSvd` and `foamSmartSimSvdDBAPI` is that
the second one uses the standardized SmartRedis database API from [src/smartRedis](/src/smartRedis)
to interact with SmartRedis while the first one uses the raw SmartRedis API.
