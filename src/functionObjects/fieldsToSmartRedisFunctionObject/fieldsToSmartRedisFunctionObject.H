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
    Foam::functionObjects::fieldsToSmartRedisFunctionObject

Group
    grpUtilitiesFunctionObjects

Description
    Provides a general interface to interact with SmartRedis databases
    from OpenFOAM code.

Usage
    Example of function object specification:
    \verbatim
    difference
    {
        type smartRedis;
        libs (fieldsToSmartRedisFunctionObjects);

        // Unique name for this function object instance
        name pU;
    }
    \endverbatim

See also
    Foam::functionObject

SourceFiles
    fieldsToSmartRedisFunctionObject.C
    fieldsToSmartRedisFunctionObjectTemplates.C

\*---------------------------------------------------------------------------*/

#ifndef fieldsToSmartRedisFunctionObject_H
#define fieldsToSmartRedisFunctionObject_H

#include "objectRegistry.H"
#include "smartRedisFunctionObject.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                     Class fieldsToSmartRedisFunctionObject Declaration
\*---------------------------------------------------------------------------*/

class fieldsToSmartRedisFunctionObject
:
    public functionObjects::smartRedisFunctionObject
{
protected:

    // Protected Data

        //- list of fields
        wordList fields_;

    // Protected Member Functions

        //- No copy construct
        fieldsToSmartRedisFunctionObject(const fieldsToSmartRedisFunctionObject&) = delete;

        //- No copy assignment
        void operator=(const fieldsToSmartRedisFunctionObject&) = delete;


public:

    //- Runtime type information
    TypeName("fieldsToSmartRedis");


    // Constructors

        //- Construct from Time and dictionary
        fieldsToSmartRedisFunctionObject
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~fieldsToSmartRedisFunctionObject() = default;


    // Member Functions

        //- Read and set the function object if its data have changed
        virtual bool read(const dictionary&);

        //- Set dataset name on the DB
        virtual word datasetName(word timeIndex, word mpiRank) const {
            return name() + "_timeindex_" + timeIndex + "_mpirank_" + mpiRank;
        }

        //- Set field name on the DB
        virtual word fieldName
        (
            word field,
            word patch = "internal"
        ) const {
            return field + "_" + patch;
        }

        //- Execute (at time-step)
        virtual bool execute();

        //- Write (at write interval)
        virtual bool write();

        //- Executed at the final time-loop
        virtual bool end();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "fieldsToSmartRedisFunctionObjectTemplates.C"
#endif

#endif

// ************************************************************************* //
