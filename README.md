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
2. [SmartSim] 0.6.0 
2. [SmartRedis] N.N.N 

## Building

The OpenFOAM-SmartSim coupling functions over a connection that OpenFOAM as a client maintains with the SmartRedis database. This means that OpenFOAM elements (application, solver, function object, boundary condition, etc.) must be able to include SmartRedis source folders and link with a SmartRedis library. To facilitate this, a Bash script is provided:

```Bash
# This will set up your environment, get smartredis, and compile all libs and apps 
source SOURCEME.sh
```

## How to use it

- Run `source SOURCEME.sh`
  - This will fetch latest-n-greatest (and compile) [SmartRedis](https://github.com/CrayLabs/SmartRedis) for you
  - It will compile the OpenFOAM libs provided in `src` into your `$FOAM_USER_LIBBIN`
- Make sure port 8000 is free. `ss -plant  | grep 8000` should return nothing!
- Head to one of the tutorials, and run the python script you find there

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


