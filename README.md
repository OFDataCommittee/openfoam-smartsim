# OpenFOAM-SmartRedis interactions through function objects

## Requirements

- Install [SmartSim](https://www.craylabs.org/docs/installation_instructions/basic.html#smartsim)
- Install [OpenFOAM](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/)
- Run `source SOURCEME.sh`
  - This will fetch latest-n-greatest (and compile) [SmartRedis](https://github.com/CrayLabs/SmartRedis) for you
  - It will compile the OpenFOAM libs provided in `src` into your `$FOAM_USER_LIBBIN`
  - It will set `SSDB=localhost:8000` in your environment.
- Make sure port 8000 is free. `ss -plant  | grep 8000` should return nothing!
- Head to one of the tutorials, and run the python script you find there

## Objectives

- Define a standard way to interact with SmartRedis database from OpenFOAM's function objects.

## Documentation

There is some [documentation](docs.md) to get you started.
