/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::smartRedisDatabase

Description
    Provides an interface to interact with SmartRedis databases
    from OpenFOAM code.

See also
    Foam::functionObjects::fieldsToSmartRedisFunctionObject

SourceFiles
    smartRedisDatabase.C
    smartRedisDatabaseTemplates.C

\*---------------------------------------------------------------------------*/

#ifndef smartRedisDatabase_H
#define smartRedisDatabase_H

#include "fvMesh.H"
#include "smartRedisAdapter.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class smartRedisDatabase Declaration
\*---------------------------------------------------------------------------*/

class smartRedisDatabase
{
protected:

    // Protected Data
        
        //- Database name
        word name_;

        //- Region name
        word region_;

        //- A const-ref to the mesh
        const fvMesh& mesh_;

        //- Client name (for debug output)
        word clientName_;

        //- The RedisAI database client
        tmp<smartRedisAdapter> redisDB_;

        //- Naming schemes for tensors in the DB
        HashTable<wordList> namingConvention_;

        //- Running arguments to the naming convention (ie. values)
        dictionary namingConventionState_;

    // Protected Member Functions
        
        //- Post metadata dataset to DB
        //  Currently, this has to be called in dervied constructors if 
        //  they introduce new Jinja2 naming
        dictionary postMetadata();

        //- Get metadata Dataset for this function object
        DataSet getMetadata();

        //- Update naming convention state
        virtual void updateNamingConventionState();

        //- Check patch names
        void checkPatchNames
        (
            const wordList& patchNames
        ) const;

        //- check if fields are registered
        template<class... Types>
        bool checkAllFields
        (
            const wordList& fieldNames,
            const objectRegistry& obr
        );

        //- send all fields in the list
        template<class... Types>
        void sendAllFields
        (
            DataSet& ds,
            const wordList& fieldNames,
            const wordList& patchNames = wordList{"internal"}
        );
        
        //- get all fields in the list
        template<class... Types>
        void getAllFields
        (
            DataSet& ds,
            const wordList& fieldNames,
            const wordList& patchNames = wordList{"internal"}
        );
        
        //- No copy construct
        smartRedisDatabase(const smartRedisDatabase&) = delete;

        //- No copy assignment
        void operator=(const smartRedisDatabase&) = delete;


public:

    //- Runtime type information
    TypeName("smartRedis");


    // Constructors

        //- Construct from Time and dictionary
        smartRedisDatabase
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~smartRedisDatabase() = default;


    // Member Functions

        //- Return name
        const word& name() const {
            return name_;
        }

        //- Return ref to mesh
        const fvMesh& mesh() const {
            return mesh_;
        }

        //- Return naming convention
        SmartRedis::Client& client() {
            return redisDB_->client();
        }

        //- Return naming convention
        const HashTable<wordList>& namingConvention() const {
            return namingConvention_;
        }

        //- Return consst-ref to current state of naming convention
        const dictionary& namingConventionState() const {
            return namingConventionState_;
        }

        //- Return current state of naming convention
        dictionary& namingConventionState() {
            return namingConventionState_;
        }

        //- Create scheme values from flat dict
        //  Scheme values can be incomplete, but must be enough to construct
        //  a name for "key"
        dictionary createSchemeValues
        (
            const dictionary& values,
            bool allowIncomplete = false
        ) const;

        //- Get entity name on the database from values for the scheme
        word extractName
        (
            const word key,
            const dictionary& schemeValue
        ) const;

        //- Get full name on the DB for an OpenFOAM field
        word getDBFieldName
        (
            const word fName
        ) const;

        //- Add a string to metadata dataset
        void addToMetadata
        (
            const word& key,
            const word& value
        );

        //- Send fields of type T to SmartRedis Dataset
        template<class T>
        void packFields
        (
            DataSet& ds,
            const wordList& fieldNames,
            const wordList& patchNames = wordList{"internal"}
        );

        //- Get fields of type T from SmartRedis DB
        template<class T>
        void getFields
        (
            DataSet& ds,
            const wordList& fieldNames,
            const wordList& patchNames = wordList{"internal"}
        );

        //- Send a set of OpenFOAM fields of any type as SmartRedis tensors
        // @todo const-functions should also be enforced on the SmartRedis API
        // @body This is the best example for a `void sendFields(const wordList& fieldNames) const;`
        void sendGeometricFields
        (
            const wordList& fieldNames,
            const wordList& patchNames = wordList{"internal"}
        );

        //- Get a set of OpenFOAM fields of any type from SmartRedis tensors
        void getGeometricFields
        (
            const wordList& fieldNames,
            const wordList& patchNames = wordList{"internal"}
        );

        //- Send a list of objects to SmartRedis DB
        template<class T>
        void sendList(const List<T>& lst, const word& listName);

        //- Get a list of objects from SmartRedis DB
        template<class T>
        void getList(List<T>& lst, const word& listName);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#define supportedFieldTypes \
    Foam::volScalarField, \
    Foam::volVectorField, \
    Foam::volTensorField, \
    Foam::volSphericalTensorField, \
    Foam::volSymmTensorField, \
    Foam::surfaceScalarField, \
    Foam::surfaceVectorField, \
    Foam::surfaceTensorField, \
    Foam::surfaceSphericalTensorField, \
    Foam::surfaceSymmTensorField

#ifdef NoRepository
    #include "smartRedisDatabaseTemplates.C"
#endif

#endif

// ************************************************************************* //
