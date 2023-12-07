# Function Objects for SmartRedis-OpenFOAM interaction

## `smartRedisClient`

This class is the building block for all SmartRedis-OpenFOAM interactions. It provides an interface to SmartRedis
databases in three API levels:

1. **Service API calls,** which take only a list of OpenFOAM field names as an argument. Interactions with the DB
    are then handled automatically. Examples of these calls are
   - `void smartRedisClient::sendGeometricFields(const wordList& fields)`
   - `void smartRedisClient::getGeometricFields(const wordList& fields)`
2. **Developer API calls,** which take at least a `DataSet` object as an argument. These calls only handle local DataSet
   objects, and do not interact with the DB.  Examples of these calls are
   - `void smartRedisClient::packFields<Type>(DataSet&, const wordList&)`
   - `void smartRedisClient::getFields<Type>(DataSet&, const wordList& fields)`
3. **Generic-interaction API calls,** which deal with send and receiving a `List<Type>` of data elements to/from
   the database directly, without packing things into datasets. These are great for one-time interactions and are
   aware of MPI ranks. Examples of these calls include:
   - `void smartRedisClient::sendList<Type>(List<Type>& lst, const word& newName)`
   - `void smartRedisClient::getList<Type>(List<Type>& lst, const word& name)`

This class also manages:

- A naming convention of tensors on the Database which correspond to OpenFOAM fields (or parts of them)
- A shared client between all `smartRedisClient` objects
- A metadata `DataSet` which holds the naming convention templates and any arbitrary data a user
  of this class may deem important

### Technical notes

- The metadata dataset is posted to the database at construction of the `smartRedisClient` object
  through `postMetadata()` member method.
  - This method posts everything it finds in `namingConvention_` member (a `HashTable<string>`). Hence,
    Any class inheriting from `smartRedisClient` can add its own/override metadata to the database
    by populating `namingConvention_` and calling `postMetadata()`.
  - To fetch the name of the dataset, one can
    ```cpp
    // db is a smartRedisClient object
    db.updateNamingConventionState();
    word dsName = db.extractName("dataset", db.namingConventionState())
    ```
- By default, all field-related methods treat the "internal" field
  - But a list of boundary patches can be supplied
  - And if an MPI rank does not have the particular patch allocated (size == 0), it will not be
    communicated with the database. Clients from other languages need to take this into consideration.

## `fieldsToSmartRedisFunctionObject`

This function object is used to pack and send (one-way communication to DB) selected OpenFOAM fields once per time step.
 
To send `p`, `U` and `phi` "internal" fields to the database every time step, one could use:
```
functions
{
    pUPhiTest
    {
        type fieldsToSmartRedis;
        libs ("libsmartredisFunctionObjects.so");
        clusterMode off;
        fields (p U phi);
        patches (internal);
    }
}
```

### Technical notes

- This class defines (and brag about) the following naming conventions, which are found in the `pUPhiTest_metadata` dataset:
  ```
  The following Jinja2 templates define the naming convention:
  {
      field           "field_name_{{ name }}_patch_{{ patch }}";
      dataset         "pUPhiTest_time_index_{{ time_index }}_mpi_rank_{{ mpi_rank }}";
  }
  ```
  - After running the Jinja2 templating engine on these conventions, a full field name on the database for the
    following data:
    ```
    time_index = 200
    name = p
    patch = internal
    mpi_rank = 0
    ```
    would look like `{pUPhiTest_time_index_200_mpi_rank_0}.field_name_p_patch_internal`.
  - At the moment, internal and boundary patch parts of volume and surface fields with
    components of the following types are supported:
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
