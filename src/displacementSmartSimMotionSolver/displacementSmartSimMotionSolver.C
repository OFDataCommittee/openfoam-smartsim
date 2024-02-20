/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 Tomislav Maric, TU Darmstadt 
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

#include "Pstream.H"
#include "displacementSmartSimMotionSolver.H"
#include "addToRunTimeSelectionTable.H"
#include "OFstream.H"
#include "meshTools.H"
#include "mapPolyMesh.H"
#include "fvPatch.H"
#include "fixedValuePointPatchFields.H"
//#include "motionInterpolation.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(displacementSmartSimMotionSolver, 0);

    addToRunTimeSelectionTable
    (
        motionSolver,
        displacementSmartSimMotionSolver,
        dictionary
    );

    addToRunTimeSelectionTable
    (
        displacementMotionSolver,
        displacementSmartSimMotionSolver,
        displacement
    );
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::displacementSmartSimMotionSolver::displacementSmartSimMotionSolver
(
    const polyMesh& mesh,
    const IOdictionary& dict
)
:
    displacementMotionSolver(mesh, dict, typeName),
    fvMotionSolver(mesh),
    //cellDisplacement_
    //(
        //IOobject
        //(
            //"cellDisplacement",
            //mesh.time().timeName(),
            //mesh,
            //IOobject::READ_IF_PRESENT,
            //IOobject::AUTO_WRITE
        //),
        //fvMesh_,
        //dimensionedVector(pointDisplacement_.dimensions(), Zero),
        //cellMotionBoundaryTypes<vector>(pointDisplacement_.boundaryField())
    //),
    //interpolationPtr_
    //(
        //coeffDict().found("interpolation")
      //? motionInterpolation::New(fvMesh_, coeffDict().lookup("interpolation"))
      //: motionInterpolation::New(fvMesh_)
    //),
    clusterMode_(this->coeffDict().get<bool>("clusterMode")), 
    client_(clusterMode_)
{}

Foam::displacementSmartSimMotionSolver::
displacementSmartSimMotionSolver
(
    const polyMesh& mesh,
    const IOdictionary& dict,
    const pointVectorField& pointDisplacement,
    const pointIOField& points0
)
:
    displacementMotionSolver(mesh, dict, pointDisplacement, points0, typeName),
    fvMotionSolver(mesh),
    //cellDisplacement_
    //(
        //IOobject
        //(
            //"cellDisplacement",
            //mesh.time().timeName(),
            //mesh,
            //IOobject::READ_IF_PRESENT,
            //IOobject::AUTO_WRITE
        //),
        //fvMesh_,
        //dimensionedVector(pointDisplacement_.dimensions(), Zero),
        //cellMotionBoundaryTypes<vector>(pointDisplacement_.boundaryField())
    //),
    //interpolationPtr_
    //(
        //coeffDict().found("interpolation")
      //? motionInterpolation::New(fvMesh_, coeffDict().lookup("interpolation"))
      //: motionInterpolation::New(fvMesh_)
    //),
    clusterMode_(dict.getOrDefault<bool>("clusterMode", true)),
    client_(clusterMode_)
{}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::displacementSmartSimMotionSolver::
~displacementSmartSimMotionSolver()
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::pointField> Foam::displacementSmartSimMotionSolver::curPoints() const
{
    //interpolationPtr_->interpolate
    //(
        //cellDisplacement_,
        //pointDisplacement_
    //);

    tmp<pointField> tcurPoints
    (
        points0() + pointDisplacement_.primitiveField()
    );
    pointField& curPoints = tcurPoints.ref();
    twoDCorrectPoints(curPoints);

    return tcurPoints;
}

void Foam::displacementSmartSimMotionSolver::solve() 
{
    // The points have moved so before interpolation update
    pointDisplacement_.boundaryFieldRef().evaluate();

    // Send mesh boundary points and displacements to smartRedis 
    const auto& boundaryDisplacements = pointDisplacement().boundaryField();
    const auto& meshBoundary = motionSolver::mesh().boundaryMesh(); 

    // MPI rank is used for identifying tensors 
    auto mpiIndexStr = std::to_string(Pstream::myProcNo());

    // Create the boundary points DataSet
    auto pointsDsetName = "points_dataset_MPI_" + mpiIndexStr; 
    SmartRedis::DataSet pointsDataset(pointsDsetName);
    // Create the boundary displacements DataSet
    auto displDsetName = "displacement_dataset_MPI_" + mpiIndexStr;
    SmartRedis::DataSet displacementsDataset(displDsetName);

    forAll(boundaryDisplacements, patchI)
    {
        // Skip processor and empty Finite Volume patches
        if ((meshBoundary[patchI].type() == "empty") || 
            (meshBoundary[patchI].type() == "processor"))
        {
            Pout << "Skipping " << meshBoundary[patchI].name() << ", "
                << meshBoundary[patchI].type() << endl;
            continue;
        }
        
        const polyPatch& patch = meshBoundary[patchI];

        const pointField& patchPoints = patch.localPoints();
        const pointPatchVectorField& patchDisplacements = 
		boundaryDisplacements[patchI];
        vectorField patchDisplacementData = 
		patchDisplacements.patchInternalField(); 

        // Point patch addressing is global - the boundary loop on each MPI rank
        // sees all patches, and those not available in this MPI rank will have 
        // size 0. Size 0 data cannot be written into the SmartRedis database.
        if (patch.size() == 0)
        {
            Pout << "Skipping " << patch.name() << " with points size "
                << patchPoints.size() << " and displacements size " 
                << patchDisplacementData.size() << endl;
            continue;
        }
    
        Pout << "Sending " << patch.name() 
             << "points size " << patchPoints.size() << endl
             << " displacements size " << patchDisplacementData.size() << endl
             << " to SmartRedis." << endl;
        
        // Add the patch points to the boundary points dataset 
        auto pointsName = "points_" + patch.name() + "_MPI_" + mpiIndexStr; 
        pointsDataset.add_tensor(pointsName,
                                 (void*)patchPoints.cdata(), 
                                 std::vector<size_t>{size_t(patchPoints.size()), 3},
                                 SRTensorTypeDouble, SRMemLayoutContiguous);

        // Add the patch displacements to the boundary displacements dataset 
        auto displacementsName = "displacements_" + patch.name() + "_MPI_" + mpiIndexStr;  
        displacementsDataset.add_tensor(displacementsName,
                                        (void*)patchDisplacementData.cdata(), 
                                        std::vector<size_t>{size_t(patchPoints.size()), 3},
                                        SRTensorTypeDouble, SRMemLayoutContiguous);
        
    }

    client_.put_dataset(pointsDataset);
    client_.put_dataset(displacementsDataset);
    client_.append_to_list("pointsDatasetList", pointsDataset);
    client_.append_to_list("displacementsDatasetList", displacementsDataset);

    bool model_updated = client_.poll_key("model_updated", 10, 10000);
    if (! model_updated)
    {
        FatalErrorInFunction
            << "Displacement model not found in SmartRedis database."
            << exit(Foam::FatalError);
    }
    else
    {
        // Send only internal mesh points to SmartRedis for forward inference.
        // Since mesh.points() is a pointField we can use pointDisplacement_
        // internal/boundary field addressing. OpenFOAM stores geometric fields
        // as internal field X = [x0, x1, ..., x_{N_i -1}, x_{N_i}, ...
        // x_{|X|}] basically storing internal values up to x_{N_i -1}, with
        // N_i = number of internal values, then appending boundary field as
        // patches from N_i up to |X|. 
        
        auto inputName = "meshPoints_" + mpiIndexStr;
        const auto& meshPoints = fvMesh_.points();
        const size_t nInternalPoints = pointDisplacement_.size(); 

        // Extract 2D internal points from 3D OpenFOAM points into a flattened
        // contiguous std::vector  
        // [[p00 p01 p02], [p10 p11 p12], ...] ->
        // [q00 q01 q10 q11 q20 q21 ...]
        std::vector<double> inputPoints2D(nInternalPoints*2, 0);
        for(std::size_t pI = 0; pI < nInternalPoints; ++pI)
        {
            inputPoints2D[pI*2] = meshPoints[pI][0];
            inputPoints2D[pI*2+1] = meshPoints[pI][1];
        }
        // Send the 2D flattened points to SmartRedis, but as 
        // [q00 q01 q10 q11 q20 q21 ...] ->
        // [[q00 q01], [q10 q11], [q20 q21], ...]
        // This shape is needed for the MLP! 
        client_.put_tensor(inputName,
                           (void*)inputPoints2D.data(), 
                           {nInternalPoints,2},
                           SRTensorTypeDouble, 
                           SRMemLayoutContiguous);

        // Perform the forward inference in SmartRedis
        auto outputName = "outputDisplacements_" + mpiIndexStr;
        client_.run_model("MLP", {inputName}, {outputName});

        std::vector<double> outputDisplacements2D(nInternalPoints*2, 0);
        client_.unpack_tensor(outputName, outputDisplacements2D.data(), 
                              {nInternalPoints*2},
                              SRTensorTypeDouble, 
                              SRMemLayoutContiguous);


        // Evaluate boundary and internal displacements froom the ML model.
        pointVectorField newDisplacements ("newDisplacements", 
                                            pointDisplacement_);
        for(std::size_t i = 0; i < outputDisplacements2D.size(); ++i)
        {
            newDisplacements[i / 2][i % 2] = outputDisplacements2D[i];
        }
        newDisplacements.boundaryFieldRef().evaluate(); 
        pointDisplacement_.internalFieldRef() = newDisplacements.internalField(); 
        pointDisplacement_.boundaryFieldRef().evaluate(); 
        // TODO: debugging 
        newDisplacements.write();
        // - 
    }

    // At the end of the simulation, have MPI rank 0 notify the python 
    // client via SmartRedis that the simulation has completed by writing
    // an end_time_index tensor to SmartRedis. 
    const auto& runTime = fvMesh_.time();
    if ((Pstream::myProcNo() == 0) && (runTime.timeIndex() == 20))
    {
        std::vector<double> end_time_vec {double(runTime.timeIndex())};
        Info << "Seting end time flag : " << end_time_vec[0] << endl;
        client_.put_tensor("end_time_index", end_time_vec.data(), {1}, 
                            SRTensorTypeDouble, SRMemLayoutContiguous);
    }

    // Emulate MPI_Barrier() - wait for all MPI ranks to perform forward
    // inference of displacements and move the mesh with ML displacements.
    label totalRank = Pstream::myProcNo();
    reduce(totalRank, sumOp<label>(), totalRank);

    // Delete the model flag. 
    if (Pstream::myProcNo() == 0)
    {
        client_.delete_tensor("model_updated");
    }

}

// ************************************************************************* //
