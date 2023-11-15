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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "fieldsToSmartRedisFunctionObject.H"

namespace Foam
{
namespace functionObjects
{

template<class... Types>
bool fieldsToSmartRedisFunctionObject::checkAllFields
(
    const wordList& fieldNames,
    const objectRegistry& obr
)
{
    static_assert(sizeof...(Types) > 0, "At least one template argument is required");
    forAll(fieldNames, fi) {
        if (!(obr.foundObject<Types>(fieldNames[fi]) || ...)) {
            FatalErrorInFunction
                << "Field " << fieldNames[fi] << " not found in objectRegistry"
                << " as any of the supported types." << nl
                << exit(FatalError);
        }
    }
    return true;
}


template<class... Types>
void fieldsToSmartRedisFunctionObject::sendAllFields
(
    DataSet& ds,
    const wordList& fieldNames
)
{
    static_assert(sizeof...(Types) > 0, "At least one template argument is required");
    (packFields<Types>(ds, fieldNames), ...);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// ************************************************************************* //
