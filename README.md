# openfoam-smartsim 

## General Description

Sub-module for [OpenFOAM][OpenFOAM] that provides a solver for embedding [SmartSim][SmartSim]
and its external dependencies (i.e. [SmartRedis][SmartRedis]) into arbitrary OpenFOAM simulations.

The sub-module provides examples for coupling OpenFOAM with SmartSim 
    - pre-processing application 
    - function object
    - mesh motion solver
    - parameter estimation 

## License

The source code license: GPL-3.0-or-later

## Requirements

1. [OpenFOAM-v2312] or newer, or a recent [development version][OpenFOAM-git]
   from [OpenFOAM.com][OpenFOAM]. 
2. [SmartSim] 0.6.2 [installation instructions](https://www.craylabs.org/docs/installation_instructions/basic.html)
3. [SmartRedis] latest, installed automatically by `./Allwmake`
4. [PyFoam,pandas,matplotlib]: pip install PyFoam pandas matplotlib

## Building

### Tested compilers 

* `WM_COMPILER=Gcc` , compilers: g++ 11.4.0
* `WM_COMPILER=Icx`, compilers: Intel(R) oneAPI DPC++/C++ Compiler 2023.2.4

### Compiling and installation 

The OpenFOAM-SmartSim coupling functions over a connection that OpenFOAM as a client maintains with the SmartRedis database. This means that OpenFOAM elements (application, solver, function object, boundary condition, etc.) must be able to include SmartRedis source folders and link with a SmartRedis library. To facilitate this, an `./Allwmake` Bash script is provided. 

To build the project, you need to have a working OpenFOAM environment: 

```
openfoam-smartsim > ./Allwmake
```

- This will fetch and compile the latest-n-greatest [SmartRedis](https://github.com/CrayLabs/SmartRedis) for you.
- SmartRedis libraries will be installed into `$FOAM_USER_LIBBIN`.
- OpenFOAM+SmartSim libraries and applictaions will also be installed into `$FOAM_USER_LIBBIN`.

## Running 

OpenFOAM+SmartSim workflows are implemented in Python programs where SmartSim "governs" the computational workflow. The workflows can be implemented in Jupyter Notebooks or as Python programs. 

Examples that use Jupyter Notebooks set some requirements: 

- Make sure port 8000 is free. `ss -plant  | grep 8000` should return nothing!

Head to one of the tutorials, and run the jupyter notebook or a python program you find there.

## Authors / Contributors

| Name | Affiliation | Email
|------|-------|-----------|
| Alessandro Rigazzi | HPE | |
| Andrew Shao | HPE | |
| Andre Weiner | TU Dresden | |
| Matt  Ellis | HPE | |
| Mohammed Elwardi Fadeli | TU Darmstadt | |
| Tomislav Maric | TU Darmstadt | |

----

[OpenFOAM]: https://www.openfoam.com
[OpenFOAM-v2312]: https://www.openfoam.com/releases/openfoam-v2312/
[OpenFOAM-git]: https://develop.openfoam.com/Development/openfoam

[SmartSim]: https://github.com/CrayLabs/SmartSim 
[SmartSim-Installation]: https://www.craylabs.org/docs/installation_instructions/basic.html
[SmartRedis]: https://www.craylabs.org/docs/installation_instructions/basic.html

## References

Maric, T., Fadeli, M. E., Rigazzi, A., Shao, A., & Weiner, A. (2024). Combining Machine Learning with Computational Fluid Dynamics using OpenFOAM and SmartSim. arXiv preprint [https://doi.org/10.1007/s11012-024-01797-z](https://doi.org/10.1007/s11012-024-01797-z).
