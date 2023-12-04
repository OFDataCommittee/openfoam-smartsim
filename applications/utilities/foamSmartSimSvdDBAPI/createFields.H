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

#include "regIOobject.H"

// Initialize empty shared pointers (Foam::tmp)
tmp<volScalarField> inputVolScalarFieldTmp(nullptr); 
tmp<volVectorField> inputVolVectorFieldTmp(nullptr); 

// Create the field header 
IOobject fieldHeader
(
    fieldName, 
    currentTime.name(), 
    mesh,  
    IOobject::MUST_READ
);
    
// If the cell centered scalar field is available for reading 
if (fieldHeader.typeHeaderOk<volScalarField>(true, true, false))
{
    Info <<  "Reading input field " <<  fieldHeader.objectPath() << endl;
    inputVolScalarFieldTmp = new volScalarField(fieldHeader, mesh);
}
// If the cell centered vector field is available for reading 
else if (fieldHeader.typeHeaderOk<volVectorField>(true, true, false)) 
{
    Info <<  "Reading input field " <<  fieldHeader.objectPath() << endl;
    inputVolVectorFieldTmp = new volVectorField(fieldHeader, mesh);
}

// ************************************************************************* //
