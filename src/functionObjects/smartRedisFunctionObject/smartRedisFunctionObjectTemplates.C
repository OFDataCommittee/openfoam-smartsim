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

#include "smartRedisFunctionObject.H"

namespace Foam
{
namespace functionObjects
{

// Specialization of components counting for double
template<>
struct smartRedisFunctionObject::NComponents<double>
{
    static constexpr label value = 1;
};


template<class T>
void smartRedisFunctionObject::packFields
(
    DataSet& ds,
    const wordList& fieldNames
)
{
    for (auto& fName : fieldNames)
    {
        if (!mesh().foundObject<T>(fName)) continue;
        const T& sField = mesh().lookupObject<T>(fName);
        std::vector<size_t> dims = {
            size_t(sField.size()),
            size_t(NComponents<typename T::cmptType>::value)
        };
        word fNameDB = fieldName(fName, "internal");
        // @todo Again, SmartRedis API does not seem to prefer const-correctness
        // @body Can add_tensor take a const pointer instead? I'd like to
        //       fetch fields in const-correct manner, hence the following
        //       horrible casting
        ds.add_tensor
        (
            fNameDB,
            const_cast<void*>(static_cast<const void*>(sField.internalField().cdata())),
            dims,
            SRTensorTypeDouble, SRMemLayoutContiguous
        );
    }
}

template<class T>
void smartRedisFunctionObject::recvFields
(
    const wordList& fieldNames
)
{
    word dsName = datasetName(Foam::name(mesh().time().timeIndex()), Foam::name(Pstream::myProcNo()));
    if (redisDB_->client().dataset_exists(dsName))
    {
        auto ds = redisDB_->client().get_dataset(dsName);
        for (auto& fName : fieldNames)
        {
            const T& sField = mesh().lookupObject<T>(fName);
            std::vector<size_t> dims = {
                size_t(sField.size()),
                size_t(NComponents<typename T::cmptType>::value)
            };
            word fNameDB = fieldName(fName, "internal");
            ds.get_tensor
            (
                fNameDB,
                sField.internalField().data(),
                dims,
                SRTensorTypeDouble, SRMemLayoutContiguous
            );
        }
    } else {
        FatalErrorInFunction
            << "SmartRedis Dataset " << dsName << " does not exist"
            << abort(FatalError);
    }
}


template<class T>
void smartRedisFunctionObject::sendList
(
    const List<T>& lst,
    const word& listName
)
{
    // @todo Implement send for generic lists
    // @body This is supposed to be a way to send stuff outside any dataset scope 
    NotImplemented;
}

template<class T>
void smartRedisFunctionObject::recvList
(
    List<T>& lst,
    const word& listName
)
{
    if (!client().key_exists(listName))
    {
        FatalErrorInFunction
            << "SmartRedis tensor " << listName << " does not exist"
            << abort(FatalError);
    }
    std::vector<size_t> dims = {
        size_t(lst.size()),
        size_t(NComponents<typename T::cmptType>::value)
    };
    client().get_tensor
    (
        listName,
        lst.data(),
        dims,
        SRTensorTypeDouble, SRMemLayoutContiguous
    );
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// ************************************************************************* //
