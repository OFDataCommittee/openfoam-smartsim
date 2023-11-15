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
#include "volFields.H"
#include "dictionary.H"
#include "Time.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"

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
} // End namespace functionObjects
} // End namespace Foam


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::fieldsToSmartRedisFunctionObject::fieldsToSmartRedisFunctionObject
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    smartRedisFunctionObject(name, runTime, dict),
    fields_(dict.lookupOrDefault("fields", wordList()))
{
    read(dict);
    this->namingConvention_.set
    (
        "DataSetNaming",
        datasetName("{{ timestep }}", "{{ processor }}")
    );
    this->namingConvention_.set
    (
        "FieldNaming",
        fieldName("{{ field }}", "{{ patch }}")
    );
    Info
        << "Naming conventions on the Database:"
        << namingConvention_ << endl;
    postMetadata();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool
Foam::functionObjects::fieldsToSmartRedisFunctionObject::read(const dictionary& dict)
{
    return fvMeshFunctionObject::read(dict);
}

bool
Foam::functionObjects::fieldsToSmartRedisFunctionObject::execute()
{
    // !! before attempting DB communication !!
    // First check if the mesh has all requested fields
    checkAllFields<
        volScalarField,
        volVectorField,
        volTensorField,
        volSphericalTensorField,
        volSymmTensorField,
        surfaceScalarField,
        surfaceVectorField,
        surfaceTensorField,
        surfaceSphericalTensorField,
        surfaceSymmTensorField
    >(fields_, mesh());
    // If the dataset exists, fetch it, otherwise create a new one
    // @todo: Not clear what to do if the dataset exists but is not complete
    // @body: Typically, we want to put all fields (from all types) into The
    //        dataset and then send it
    autoPtr<DataSet> dsPtr;
    word dsName = datasetName(Foam::name(mesh().time().timeIndex()), Foam::name(Pstream::myProcNo()));
    if (redisDB_->client().dataset_exists(dsName)) {
        // Potentionally buggy behavior
        DataSet ds = redisDB_->client().get_dataset(dsName);
        dsPtr.reset(&ds);
    } else {
        dsPtr.reset(new DataSet(dsName));
    }
    sendAllFields<
        volScalarField,
        volVectorField,
        volTensorField,
        volSphericalTensorField,
        volSymmTensorField,
        surfaceScalarField,
        surfaceVectorField,
        surfaceTensorField,
        surfaceSphericalTensorField,
        surfaceSymmTensorField
    >(*dsPtr, fields_);
    // post the dataset
    redisDB_->client().put_dataset(*dsPtr);
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
    redisDB_->client().put_dataset(ds);
    return true;
}

// ************************************************************************* //
