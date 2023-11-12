# Compile the foamSmartSimSvd application 

1. [Install smartredis + smartsim](https://www.craylabs.org/docs/installation_instructions/basic.html#smartredis) 
1. Install OpenFOAM and source OpenFOAM environment. 
2. Activate conda environment.  

```
foamSmartSimSvd> source set-smertredis-env.sh 
foamSmartSimSvd> wmake foamSmartSimSvd 
```

# Running the foamSmartSimSvd application

```
foamSmartSimSvd> jupyter notebook foam-smartsim-dmd.ipynb
```

Click on `Run->Run all cells`.
