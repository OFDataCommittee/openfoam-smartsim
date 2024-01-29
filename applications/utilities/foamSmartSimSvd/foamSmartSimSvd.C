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

#include "error.H"
#include "fvCFD.H"
#include "wordList.H"
#include "timeSelector.H"
#include "client.h"
#include <iomanip>
#include <vector>

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

    SmartRedis::Client smartRedisClient(false);

    // Create a list of all time step folders from the case folder.
    instantList inputTimeDirs = timeSelector::select0(runTime, args);
    const auto& currentTime = inputTimeDirs[0];

    // Name the tensor in SmartRedis 
    const auto mpiIndexStr = std::to_string(Pstream::myProcNo());
    const auto tensorName = "fieldName_" + fieldName + 
                            "-MPIrank_" + mpiIndexStr;  

    // Read fields from the first time step.
    #include "createFields.H"

    // Reserve storage for OpenFOAM fields sent to SmartRedis for SVD 
    std::vector<Foam::scalar> scalarXrank;
    std::vector<Foam::vector> vectorXrank; 

    if (!inputVolScalarFieldTmp->empty())
    {
        scalarXrank.reserve(mesh.nCells() * inputTimeDirs.size());
    }

    // Each MPI rank sends Xrank data tensor for torch.svd in SmartRedis
    forAll(inputTimeDirs, timeI)
    {
        const auto& currentTime = inputTimeDirs[timeI];

        #include "createFields.H"

        // If 'field' is a volScalarField 
        if (!inputVolScalarFieldTmp->empty())
        {
            const auto& field = inputVolScalarFieldTmp();
            scalarXrank.insert(scalarXrank.begin() + timeI * field.size(), 
                               field.begin(), field.end());
        }
    }

    Pout << "Writing Xrank tensor " << tensorName << endl;
    if (scalarXrank.size())
    {
        smartRedisClient.put_tensor(tensorName,
                                    scalarXrank.data(), 
                                    std::vector<size_t>{size_t(mesh.nCells()), inputTimeDirs.size()},
                                    SRTensorTypeDouble, SRMemLayoutContiguous);
    }

    Pout << "Data size = " << scalarXrank.size() 
        << " , nCells = " << mesh.nCells() 
        << " , nTimeSteps = " << inputTimeDirs.size() 
        << " , nCells * nTimeSteps = " << mesh.nCells() * inputTimeDirs.size() << endl; 

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    Info<< nl;
    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
