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

#include "smartRedisClient.H"

namespace Foam
{

template<class T>
void smartRedisClient::packFields
(
    DataSet& ds,
    const wordList& fieldNames,
    const wordList& patchNames
)
{
    checkPatchNames(patchNames);
    for (auto& fName : fieldNames)
    {
        if (!mesh().foundObject<T>(fName)) continue;
        const T& sField = mesh().lookupObject<T>(fName);
        for (auto& pName : patchNames)
        {
            auto patch = mesh().boundaryMesh().findPatchID(pName);
            std::vector<size_t> dims = {
                size_t(pName == "internal" ? sField.size() : mesh().boundaryMesh()[patch].size()),
                size_t(pTraits<typename T::value_type>::nComponents)
            };
            dictionary schemeValues = namingConventionState_;
            schemeValues.subDict("field").set<string>("name", fName);
            schemeValues.subDict("field").set<string>("patch", pName);
            word fNameDB = extractName("field", schemeValues);
            const void* data = 
                pName == "internal"
                ? sField.internalField().cdata()
                : sField.boundaryField()[patch].cdata();
            if (data != nullptr)
            {
                ds.add_tensor
                (
                    fNameDB,
                    //const_cast<void*>(static_cast<const void*>(sField.internalField().cdata())),
                    data,
                    dims,
                    SRTensorTypeDouble, SRMemLayoutContiguous
                );
            } else {
                if (debug)
                {
                    WarningInFunction
                        << "Field " << fName << " does not have patch " << pName
                        << " allocated. Skipping sending to DB. This is probably fine." << endl;
                }
            }
        }
    }
}

template<class T>
void smartRedisClient::getFields
(
    DataSet& ds,
    const wordList& fieldNames,
    const wordList& patchNames
)
{
    checkPatchNames(patchNames);
    for (auto& fName : fieldNames)
    {
        if (!mesh().foundObject<T>(fName)) continue;
        T& sField = mesh().lookupObjectRef<T>(fName);
        for (auto& pName : patchNames)
        {
            auto patch = mesh().boundaryMesh().findPatchID(pName);
            size_t patch_size = pName == "internal" ? sField.size() : mesh().boundaryMesh()[patch].size();
            std::vector<size_t> dims = {
                patch_size*size_t(pTraits<typename T::value_type>::nComponents)
            };
            dictionary schemeValues = namingConventionState_;
            schemeValues.subDict("field").set<string>("name", fName);
            schemeValues.subDict("field").set<string>("patch", pName);
            word fNameDB = extractName("field", schemeValues);
            void* data = pName == "internal" ? sField.data() : sField.boundaryFieldRef()[patch].data();
            if (data != nullptr)
            {
                ds.unpack_tensor
                (
                    fNameDB,
                    data,
                    dims,
                    SRTensorTypeDouble, SRMemLayoutContiguous
                );
            } else {
                if (debug)
                {
                    WarningInFunction
                        << "Field " << fName << " does not have patch " << pName
                        << " allocated. Skipping reading from DB. This is probably fine." << endl;
                }
            }
        }
    }
}


template<class T>
void smartRedisClient::sendList
(
    const List<T>& lst,
    const word& listName
)
{
    std::vector<size_t> dims = {
        size_t(lst.size()),
        size_t(pTraits<T>::nComponents)
    };
    client().put_tensor
    (
        listName+"_mpirank_"+Foam::name(Pstream::myProcNo()),
        const_cast<void*>(static_cast<const void*>(lst.cdata())),
        dims,
        SRTensorTypeDouble, SRMemLayoutContiguous
    );
}

template<class T>
void smartRedisClient::getList
(
    List<T>& lst,
    const word& listName
)
{
    word key = listName+"_mpirank_"+Foam::name(Pstream::myProcNo());
    if (!client().key_exists(key))
    {
        FatalErrorInFunction
            << "SmartRedis tensor " << key << " does not exist"
            << abort(FatalError);
    }
    std::vector<size_t> dims = {
        size_t(lst.size())*size_t(pTraits<T>::nComponents)
    };
    client().unpack_tensor
    (
        key,
        lst.data(),
        dims,
        SRTensorTypeDouble, SRMemLayoutContiguous
    );
}

template<class... Types>
bool smartRedisClient::checkAllFields
(
    const wordList& fieldNames,
    const objectRegistry& obr
)
{
    static_assert
    (
        sizeof...(Types) > 0,
        "At least one template argument is required for checkAllFields"
    );
    forAll(fieldNames, fi) {
        if (!(obr.foundObject<Types>(fieldNames[fi]) || ...)) {
            word supportedTypes = word("(") + nl;
            ((supportedTypes += tab + Types::typeName + nl), ...);
            supportedTypes += ")";
            FatalErrorInFunction
                << "Field " << fieldNames[fi] << " not found in objectRegistry"
                << " as any of the supported types:" << nl
                << supportedTypes
                << exit(FatalError);
        }
    }
    return true;
}


template<class... Types>
void smartRedisClient::sendAllFields
(
    DataSet& ds,
    const wordList& fieldNames,
    const wordList& patchNames
)
{
    static_assert
    (
        sizeof...(Types) > 0,
        "At least one template argument is required for sendAllFields"
    );
    checkAllFields<Types...>(fieldNames, mesh());
    (packFields<Types>(ds, fieldNames, patchNames), ...);
}

template<class... Types>
void smartRedisClient::getAllFields
(
    DataSet& ds,
    const wordList& fieldNames,
    const wordList& patchNames
)
{
    static_assert(sizeof...(Types) > 0, "At least one template argument is required");
    checkAllFields<Types...>(fieldNames, mesh());
    (getFields<Types>(ds, fieldNames, patchNames), ...);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// ************************************************************************* //
