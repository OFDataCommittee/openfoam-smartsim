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

#include "fieldsToSmartRedisFunctionObject.H"
#include "Time.H"
#include "fvMesh.H"
#include "addToRunTimeSelectionTable.H"
#include "smartRedisClient.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(fieldsToSmartRedisFunctionObject, 0);
    addToRunTimeSelectionTable
    (
        functionObject,
        fieldsToSmartRedisFunctionObject,
        dictionary
    );
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::fieldsToSmartRedisFunctionObject::fieldsToSmartRedisFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    smartRedisClient(name, runTime, dict),
    fields_(dict.lookup("fields")),
    patches_(dict.lookupOrDefault("patches", wordList{"internal"}))
{}

bool
Foam::functionObjects::fieldsToSmartRedisFunctionObject::execute()
{
    Info<< "Writing fields to SmartRedis database\n" << endl;
    updateNamingConventionState();
    sendGeometricFields(fields_, patches_);
    return true;
}


bool
Foam::functionObjects::fieldsToSmartRedisFunctionObject::write()
{
    return true;
}

bool
Foam::functionObjects::fieldsToSmartRedisFunctionObject::end()
{
    DataSet ds = getMetadata();
    ds.add_meta_string("EndTimeIndex", Foam::name(mesh().time().timeIndex()));
    client().put_dataset(ds);
    return true;
}

// ************************************************************************* //
