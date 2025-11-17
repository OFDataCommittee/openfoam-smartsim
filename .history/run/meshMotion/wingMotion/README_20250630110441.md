# Running 

The 2D wing is rapidly translated and rotated, causing significant mesh deformation. The deformation is implemented as 

1. Laplace deformation available in OpenFOAM 
2. MLP Machine-Learning deformation, combining Pytorch and OpenFOAM via SmartSim/SmartRedis.  

To run Laplace deformation make sure SmartSim env and OpenFOAM env are sourced, then  

```
wingMotion> ./Allrun.LaplaceMeshMotion
```

this creates `mesh-motion_Laplace` - a folder that is an OpenFOAM simulation, for Laplace deformation, we don't need SmartSim/SmartRedis.

Tor run the SmartSim MLP mesh deformation, run

```
wingMotion> ./Allrun.MachineLearningMeshMotion
```

which creates `mesh-motion_MachineLearning`, which has sub-folders for the SmartSim orchestrator, for the openfoam model (`of_model`, OpenFOAM simulation case), and the MLP training script (`training_app`). 


Both `Allrun.LaplaceMeshMotion` and `Allrun.MachineLearningMeshMotion` will compute mesh quality metrics (most important ones are non-orthogonality and skewness), and `.foam` files that ParaView needs to recognize OpenFOAM folders. A paraview state file is prepared that compares the decrease in non-orthogonality, visualizing the difference between Laplace non-orthogonality and MLP non-orthogonality, run it as

```
wingMotion> paraview --state=visualize-non-orth-difference.pvsm
```

This will show how the Laplace causes an increase of non-orthogonality at the worst possible place - next to the airfoil. The increase is up to 35 degrees, w.r.t a simple MLP. 
