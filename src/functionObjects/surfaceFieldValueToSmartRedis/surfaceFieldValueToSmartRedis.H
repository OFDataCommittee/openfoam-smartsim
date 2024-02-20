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
    Foam::functionObjects::surfaceFieldValueToSmartRedis

Description
    Provides a general interface to interact with SmartRedis databases
    from OpenFOAM code.

Usage
    Example of function object specification:
    \verbatim
    difference
    {
        type surfaceFieldValueToSmartRedis;
        libs ("libsmartredisFunctionObjects.so");

        // Unique name for this function object instance
        name pU;

        // SmartRedis client mode
        clusterMode off;

        // OpenFOAM fields to send to the database
        fields (p U phi);

        // OpenFOAM patches to send
        // "internal" is a special patch refering to internal field
        patches (internal inlet);
    }
    \endverbatim

SourceFiles
    surfaceFieldValueToSmartRedis.C

\*---------------------------------------------------------------------------*/

#ifndef functionObjects_surfaceFieldValueToSmartRedis_H
#define functionObjects_surfaceFieldValueToSmartRedis_H

#include "surfaceFieldValueToSmartRedis.H"
#include "smartRedisClient.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                              Class surfaceFieldValueToSmartRedis Declaration
\*---------------------------------------------------------------------------*/

class surfaceFieldValueToSmartRedis
:
    public smartRedisClient, public surfaceFieldValue
{
protected:

    // Protected Member Data

    // Protected Member Functions

        //- No copy construct
        surfaceFieldValueToSmartRedis(const surfaceFieldValueToSmartRedis&) = delete;

        //- No copy assignment
        void operator=(const surfaceFieldValueToSmartRedis&) = delete;


public:

    //- Runtime type information
    TypeName("surfaceFieldValueToSmartRedis");


    // Constructors

        //- Construct from Time and dictionary
        surfaceFieldValueToSmartRedis
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~surfaceFieldValueToSmartRedis() = default;

    // Member Functions

        //- Called at each ++ or += of the time-loop.
        virtual bool write();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
