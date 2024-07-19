## Computing the SVD of OpenFOAM fields

Running `openfoam-svd.py` will perform an SVD on OpenFOAM fields that results off
of a `pitzDaily` case. The case runs with `simpleFoam` and communicates its fields
to the `smartRedis` database through a `fieldsToSmartRedis` function object.

## Usage of the Function Object with SmartSim ensembles

The same function object is used in an ensemble setting to showcase a dummy parameter
variation. The only requirement is that the solver has access to an environment variable:
```bash
export SSKEYIN=${SSKEYOUT}
```
This ensures each ensemble member reads and writes fields to a prefixed datasets to prevent
race conditions and  members reading data from each other.
