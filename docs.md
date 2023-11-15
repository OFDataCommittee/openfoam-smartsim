# Function Objects for SmartRedis-OpenFOAM interaction

## `smartRedisFunctionObject`

This class is the building block for all SmartRedis-OpenFOAM interactions. And it's not usable on its own
as it's an abstract class.

1. It opens a connection to the SmartRedis database through a client.
2. It provides a standardized interface structuring the storage and retrieval of OpenFOAM fields
   as datasets in the SmartRedis database. The exact way datasets and fields should be named
   on the database is left to the users of this class. But at the moment, this is restricted by
   the signature of `smartRedisFunctionObject::datasetName()` and `smartRedisFunctionObject::fieldName()`
   abstract functions.
3. It provides a standardized interface to send/receive OpenFOAM fields and lists to/from the
   SmartRedis database as efficiently as possibly.
4. It posts a unique metadata dataset to the database to allow clients written in other language
   to interact with the provided data without exposing any bindings.

### Technical notes

The metadata dataset is posted to the database at construction of the function object 
through `postMetadata()` member method.
This method posts everything it finds in `namingConvention_` member (a `HashTable<string>`). Hence,
Any class inheriting from `smartRedisFunctionObject` can add its own/override metadata to the database
by populating `namingConvention_` and calling `postMetadata()`.

> All classes deriving from `smartRedisFunctionObject` will use a single client to interact with the Database
> per MPI rank.

## `fieldsToSmartRedisFunctionObject`

This class is the simplest example of a fully-implemented `smartRedisFunctionObject`.
It is used to pack and send (one-way communication to DB) selected OpenFOAM fields once per time step.
 
To send `p`, `U` and `phi` "internal" fields to the database every time step, one could use:
```
libs ("libSmartRedisFunctionObjects.so");
functions
{
    pUPhiTest
    {
        type fieldsToSmartRedis;
        clusterMode off;
        fields (p U phi);
    }
}
```

### Technical notes

This class defines the following naming conventions, which are found in the `pUPhiTest_metadata` dataset:
```
DataSetNaming   "pUPhiTest_timeindex_{{ timestep }}_mpirank_{{ processor }}"
FieldNaming     "{{ field }}_{{ patch }}"
```
After running the Jinja2 templating engine on these conventions, a full field name on the database for the
following data:
```
timestep = 200
field = p
patch = internal
processor = 0
```
would look like `{pUPhiTest_timestep_200_mpirank_0}.p_internal`.

At the moment, internal fields of volume and surface fields with components of the following types are supported:
- scalar
- vector
- tensor, symmetric tensor, and spherical tensor 

## Unit tests

The Unit Tests present in `tests` folder are meant to be run with [foamUT](https://github.com/FoamScience/foamUT)
as done in corresponding CI jobs. If you want to run them locally, you'll need a RedisAI server running on `localhost:8000`
(or whatever `SSDB` is for you; this is the default value set in `SOURCEME.sh`).

For example you can simply:
```bash
docker run -p 8000:6379 --rm redislabs/redisai
```

The unit tests show the intended use of important bits of the library and, in general, what is tested there can be
trusted not to change often (at least in terms of API) during the development of this toolkit.
