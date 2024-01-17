/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 
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

Application
    foamSmartSimSvd

Description

    Uses SmartSim/SmartRedis for computing Singular Value Decomposition 
    of OpenFOAM fields.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "wordList.H"
#include "timeSelector.H"
#include "smartRedisClient.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // Selecting the time step folders for the approximated fields.
    timeSelector::addOptions();

    // Add the option to the application for the name of the approximated field.
    argList::addOption
    (
        "fieldName", 
        "fieldName",
        "Name of the approximated field, e.g. p." 
    ); 

    // OpenFOAM boilerplate: set root folder and options
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // Read the name of the approximated field from the command line option
    // foamSmartSimDmd -case cavity -fieldName p
    const word fieldName = args.get<word>("fieldName");
    Info << "Approximating field " << fieldName << endl;

    // Create a database connection
    dictionary dbDict;
    dbDict.set("region", polyMesh::defaultRegion);
    dbDict.set("clusterMode", false);
    dbDict.set("clientName", "default");
    smartRedisClient db
    (
        "foamSmartSimSvdDBAPI",
        runTime,
        dbDict
    );

    // Create a list of all time step folders from the case folder.
    instantList inputTimeDirs = timeSelector::select0(runTime, args);
    
    // Posting number of processed time steps
    db.addToMetadata("NTimes", Foam::name(inputTimeDirs.size()));

    forAll(inputTimeDirs, timeI)
    {
        const auto currentTime = inputTimeDirs[timeI];
        #include "createFields.H"
        wordList fields{fieldName};
        db.sendGeometricFields(fields, wordList{"internal"});

        if (timeI+1 != inputTimeDirs.size())
        {
            runTime.setDeltaT(inputTimeDirs[timeI+1].value() - inputTimeDirs[timeI].value());
        }
        ++runTime;
    }


    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    Info<< nl;
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
