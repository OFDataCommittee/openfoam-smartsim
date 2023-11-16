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
    Foam::functionObjects::smartRedisFunctionObject

Description
    Provides an abstract interface to interact with SmartRedis databases
    from OpenFOAM code.

See also
    Foam::functionObject
    Foam::functionObjects::fieldsToSmartRedisFunctionObject

SourceFiles
    smartRedisFunctionObject.C
    smartRedisFunctionObjectTemplates.C

\*---------------------------------------------------------------------------*/

#ifndef smartRedisFunctionObject_H
#define smartRedisFunctionObject_H

#include "fvMeshFunctionObject.H"
#include "smartRedisAdapter.H"
#include "fvMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                     Class smartRedisFunctionObject Declaration
\*---------------------------------------------------------------------------*/

class smartRedisFunctionObject
:
    public functionObjects::fvMeshFunctionObject
{
protected:

    // Protected Data

        //- Client name (for debug output)
        word clientName_;

        //- The RedisAI database client
        tmp<smartRedisAdapter> redisDB_;

        //- Naming schemes
        HashTable<word> namingConvention_;

    // Protected Member Functions
        
        //- Post metadata dataset to DB
        //  Currently, this has to be called in dervied constructors if 
        //  they introduce new Jinja2 naming
        void postMetadata();

        //- Get metadata Dataset for this function object
        DataSet getMetadata();

        //- No copy construct
        smartRedisFunctionObject(const smartRedisFunctionObject&) = delete;

        //- No copy assignment
        void operator=(const smartRedisFunctionObject&) = delete;


public:

    //- Runtime type information
    TypeName("smartRedis");


    // Constructors

        //- Construct from Time and dictionary
        smartRedisFunctionObject
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~smartRedisFunctionObject() = default;


    // Member Functions

        //- Return ref to mesh
        const fvMesh& mesh() const {
            return this->mesh_;
        }

        //- Return naming convention
        const HashTable<word>& namingConvention() const {
            return namingConvention_;
        }

        //- Return naming convention
        SmartRedis::Client& client() {
            return redisDB_->client();
        }

        //- Read and set the function object if its data have changed
        virtual bool read(const dictionary&);

        //- Set dataset name on the DB
        // @todo: Can we do better on naming conventions on the DB?
        // @body: Current API is generic enough because the naming scheme itself
        //        is posted to the DB, and can be fetched by python clients.
        //        but these method signatures need to match Jinja2 rendering args!
        virtual word datasetName
        (
            word timeIndex,
            word mpiRank
        ) const = 0;

        //- Set field name on the DB
        virtual word fieldName
        (
            word field,
            word patch = "internal"
        ) const = 0;

        //- Send fields to SmartRedis Dataset
        template<class T>
        void packFields
        (
            DataSet& ds,
            const wordList& fieldNames
        );

        //- Get fields from SmartRedis DB
        template<class T>
        void recvFields(const wordList& fieldNames);

        //- Send a list of objects to SmartRedis DB
        template<class T>
        void sendList(const List<T>& lst, const word& listName);

        //- Get a list of objects from SmartRedis DB
        template<class T>
        void recvList(List<T>& lst, const word& listName);

        //- Execute (at time-step)
        virtual bool execute() = 0;

        //- Write (at write interval)
        virtual bool write() = 0;

        //- Executed at the final time-loop
        virtual bool end() = 0;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "smartRedisFunctionObjectTemplates.C"
#endif

#endif

// ************************************************************************* //
