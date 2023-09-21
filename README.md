# openfoam-smartsim 

## General Description

Sub-module for [OpenFOAM][OpenFOAM] that provides a solver for embedding [SmartSim][SmartSim]
and its external dependencies (i.e. [SmartRedis][SmartRedis]) into arbitrary OpenFOAM simulations.

The sub-module provides examples for coupling OpenFOAM with SmartSim 
    - pre-processing application 
    - function object
    - mesh motion solver

## License

The source code license: GPL-3.0-or-later

## Requirements

1. [OpenFOAM-v2212] or newer, or a recent [development version][OpenFOAM-git]
   from [OpenFOAM.com][OpenFOAM]. 
2. [SmartSim] 0.5.1 
2. [SmartRedis] N.N.N 

## Building

The OpenFOAM-SmartSim coupling functions over a connection that OpenFOAM as a client maintains with the SmartRedis database. This means that OpenFOAM elements (application, solver, function object, boundary condition, etc.) must be able to include SmartRedis source folders and link with a SmartRedis library. To facilitate this, ensure that the OpenFOAM environment is active and that SmartRedis can be found. Check that the `PETSC_ARCH_PATH` environment variable is properly set. If the variable is empty, source the `configure-smartredis.sh` script.

Using the supplied `Allwmake` script without arguments:

```
./Allwmake
```

will install the example OpenFOAM-SmartSim applications and libraries into `FOAM_USER_LIBBIN`. 

## How to use it

TODO

## Authors / Contributors

| Name | Affiliation | Email
|------|-------|-----------|
| Alessandro Rigazzi | HPE | |
| Andrew Shao | HPE | |
| Andre Weiner | TU Dresden | |
| Matt  Ellis | HPE | |
| Tomislav Maric | TU Darmstadt | |

----

[OpenFOAM]: https://www.openfoam.com
[OpenFOAM-v2212]: https://www.openfoam.com/releases/openfoam-v2212/
[OpenFOAM-git]: https://develop.openfoam.com/Development/openfoam

[SmartSim]: https://github.com/CrayLabs/SmartSim 
[SmartSim-Installation]: https://www.craylabs.org/docs/installation_instructions/basic.html
[SmartRedis]: https://www.craylabs.org/docs/installation_instructions/basic.html

## References


