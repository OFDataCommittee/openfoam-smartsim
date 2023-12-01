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

#include "Ostream.H"
#include "smartRedisDatabase.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "dictionary.H"
#include "Time.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(smartRedisDatabase, 0);
} // End namespace Foam

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::smartRedisDatabase::smartRedisDatabase
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    name_(name),
    region_(dict.getOrDefault<word>("region", polyMesh::defaultRegion)),
    mesh_(runTime.lookupObject<fvMesh>(region_)),
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
    namingConvention_(),
    namingConventionState_()
{
    namingConvention_.set("dataset", wordList{"time_index", "mpi_rank"});
    namingConvention_.set("field", wordList{"name", "patch"});
    updateNamingConventionState();
    dictionary meta = postMetadata();
    if (debug)
    {
        Info
            << "The following scheme is adopted as a naming convention:"
            << meta << endl;
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void
Foam::smartRedisDatabase::updateNamingConventionState()
{
    dictionary values;
    values.set<string>("time_index", Foam::name(mesh().time().timeIndex()));
    values.set<string>("mpi_rank", Foam::name(Pstream::myProcNo()));
    values.set<string>("name", "+");
    values.set<string>("patch", "internal");
    namingConventionState_ = createSchemeValues(values);
}

void
Foam::smartRedisDatabase::checkPatchNames
(
    const wordList& patchNames
) const
{
    for(auto& patch: patchNames)
    {
        if (patch != "internal" && mesh().boundaryMesh().findPatchID(patch) == -1)
        {
            FatalErrorInFunction
                << "Boundary patch: " << patch
                << " does not exist in mesh: " << region_
                << nl << abort(FatalError);
        }
    }
}

Foam::dictionary
Foam::smartRedisDatabase::postMetadata()
{
    dictionary meta;
    if (Pstream::master() && this->namingConvention_.toc().size() > 0) {
        word dsName = name()+"_metadata";
        // This will override the dataset if it exists
        DataSet ds(dsName);
        forAllConstIter(HashTable<wordList>, namingConvention_, i) {
            const wordList& args = *i;
            word tmplt = (i.key() == "dataset" )?  name() : i.key();
            forAll(args, e) {
                tmplt += "_" + args[e] + "_{{ " + args[e] + " }}";
            }
            ds.add_meta_string(i.key(), tmplt);
            meta.set<string>(i.key(), tmplt);
        }
        redisDB_->client().put_dataset(ds);
    }
    // @todo Relying on Jinja2 for templating
    // @body Maybe this is the best course of action for c/fortran clients
    //       wanting to interact with this class
    Info<< "The following Jinja2 templates define the naming convention:"
        << meta << endl;
    return meta;
}

DataSet
Foam::smartRedisDatabase::getMetadata()
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

Foam::dictionary
Foam::smartRedisDatabase::createSchemeValues
(
    const dictionary& values,
    bool allowIncomplete
) const
{
    dictionary schemeVals;
    forAllConstIter(HashTable<wordList>, namingConvention(), i)
    {
        dictionary subVals;
        word key = i.key();
        const wordList& args = *i;
        for(auto& arg: args) {
            if (allowIncomplete) {
                if (values.found(arg))
                {
                    subVals.set<string>(arg, word(values.lookup(arg)));
                } else {
                    subVals.set<string>(arg, "");
                }
            } else {
                if (!values.found(arg))
                {
                    FatalErrorInFunction
                        << "Keyword '" << arg << "' not found in flat scheme values dictionary"
                        << nl << abort(FatalError);
                }
                subVals.set<string>(arg, word(values.lookup(arg)));
            }
        }
        schemeVals.set(key, subVals);
    }
    return schemeVals;
}

Foam::word
Foam::smartRedisDatabase::extractName
(
    const word key,
    const dictionary& schemeValue
) const
{
    word nameDB = (key == "dataset") ? name() : key;
    const dictionary args = schemeValue.subDict(key);
    for(const auto& arg: namingConvention()[key])
    {
       nameDB += "_" + arg + "_" + word(args.lookup(arg));
    }
    return nameDB;
}

Foam::word
Foam::smartRedisDatabase::getDBFieldName
(
    const word fName
) const
{
    dictionary fieldConvention = namingConventionState_;
    fieldConvention.subDict("field").set<string>("patch", "internal");
    fieldConvention.subDict("field").set<string>("name", fName);
    auto dt = extractName("dataset", fieldConvention);
    auto ft = extractName("field", fieldConvention);
    return word("{") + dt + "}." + ft;
}

void
Foam::smartRedisDatabase::addToMetadata
(
    const word& key,
    const word& value
)
{
    if (Pstream::master())
    {
        DataSet ds = getMetadata();
        ds.add_meta_string(key, value);
        client().put_dataset(ds);
    }
}

void
Foam::smartRedisDatabase::sendGeometricFields
(
    const wordList& fieldNames,
    const wordList& patchNames
)
{
    updateNamingConventionState();
    word dsName = extractName("dataset", namingConventionState_);
    DataSet ds(dsName);
    sendAllFields<supportedFieldTypes>(ds, fieldNames, patchNames);
    client().put_dataset(ds);
}

void
Foam::smartRedisDatabase::getGeometricFields
(
    const wordList& fieldNames,
    const wordList& patchNames
)
{
    updateNamingConventionState();
    word dsName = extractName("dataset", namingConventionState_);
    client().poll_dataset(dsName, 10, 1000);
    DataSet ds = client().get_dataset(dsName);
    getAllFields<supportedFieldTypes>(ds, fieldNames, patchNames);
}

// ************************************************************************* //
