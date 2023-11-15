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

\*---------------------------------------------------------------------------*/

#include "smartRedisFunctionObject.H"
#include "volFields.H"
#include "dictionary.H"
#include "Time.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(smartRedisFunctionObject, 0);
} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::smartRedisFunctionObject::smartRedisFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    functionObjects::fvMeshFunctionObject(name, runTime, dict),
    clientName_(dict.getOrDefault<word>("clientName", "default")),
    redisDB_(
        runTime.foundObject<smartRedisAdapter>(clientName_)
        ? &runTime.lookupObjectRef<smartRedisAdapter>(clientName_)
        : new smartRedisAdapter
            (
                IOobject
                (
                    clientName_,
                    runTime.timeName(),
                    runTime,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                dict
            )
    ),
    namingConvention_()
{
    read(dict);
    postMetadata();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::functionObjects::smartRedisFunctionObject::postMetadata()
{
    if (Pstream::master() && this->namingConvention_.toc().size() > 0) {
        word dsName = name()+"_metadata";
        if (redisDB_->client().dataset_exists(dsName))
        {
            DataSet ds = redisDB_->client().get_dataset(dsName);
            forAllConstIter(HashTable<word>, namingConvention_, i) {
                ds.add_meta_string(i.key(), *i);
            }
            redisDB_->client().put_dataset(ds);
        } else {
            DataSet ds(dsName);
            forAllConstIter(HashTable<word>, namingConvention_, i) {
                ds.add_meta_string(i.key(), *i);
            }
            redisDB_->client().put_dataset(ds);
        }
    }
}

DataSet
Foam::functionObjects::smartRedisFunctionObject::getMetadata()
{
    word dsName = name()+"_metadata";
    if (!redisDB_->client().dataset_exists(dsName))
    {
        FatalErrorInFunction
            << "Metadata dataset: " << dsName
            << " does not exist in SmartRedis database."
            << nl << abort(FatalError);
    }
    return redisDB_->client().get_dataset(dsName);
}

bool
Foam::functionObjects::smartRedisFunctionObject::read(const dictionary& dict)
{
    return fvMeshFunctionObject::read(dict);
}

// ************************************************************************* //
