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

#include "surfaceFieldValueToSmartRedis.H"
#include "Time.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(surfaceFieldValueToSmartRedis, 0);
    addToRunTimeSelectionTable
    (
        functionObject,
        surfaceFieldValueToSmartRedis,
        dictionary
    );
}

}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::fieldValues::surfaceFieldValue::surfaceFieldValueToSmartRedis::surfaceFieldValueToSmartRedis
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
{}

bool Foam::functionObjects::fieldValues::surfaceFieldValue::surfaceFieldValueToSmartRedis::write()
{
    bool result = Foam::functionObjects::fieldValues::surfaceFieldValue::write();

    if (Pstream::master())
    {
        Foam::word datasetListName = "surfaceFieldValue" + "_" + "regionName" + "_" + regionName_;
        Foam::word datasetName = datasetListName + "_" + "time_index" + "_" +
                                 std::to_string(Foam::name(mesh().time().timeIndex()));
        Dataset ds = Dataset(datasetName)

        Foam::word prefix, suffix;
        {
            if (postOperation_ != postOpNone)
            {
                // Adjust result name to include post-operation
                prefix += postOperationTypeNames_[postOperation_];
                prefix += '(';
                suffix += ')';
            }

            prefix += operationTypeNames_[operation_];
            prefix += '(';
            suffix += ')';
        }

        for (const word& fieldName : fields_)
        {
            Foam::word resultName = prefix + regionName_ + ',' + fieldName + suffix;
            auto result =
            ds.add_meta_scalar(fieldName, this->getResult(resultName))
        }
        redisDB_->client().put_dataset(ds);
        redisDB_->client().append_to_list(datasetListName, datasetName)
    }

}

// ************************************************************************* //
