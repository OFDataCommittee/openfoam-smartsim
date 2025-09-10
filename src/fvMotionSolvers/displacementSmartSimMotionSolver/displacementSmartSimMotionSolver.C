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

Foam::labelList Foam::displacementSmartSimMotionSolver::filterValidCmpts(const Vector<label>& dims) 
{
    labelList valid; 

    forAll(dims, dI) 
        if (dims[dI] == 1)  // Active solution dimension in OpenFOAM 
            valid.push_back(dI); // Valid dimension 0 (x), 1 (y), or 2 (z)

    return valid;
}

// * * * * * * * * * * * * * Private Member Functions * * * * * * * * * * * * * * //

void Foam::displacementSmartSimMotionSolver::writeSolutionDimToDatabase() 
{
    client_.put_tensor("solution_dim",
                        &solutionDim_, 
                        {1},
                        SRTensorTypeInt32, 
                        SRMemLayoutContiguous);
}

void Foam::displacementSmartSimMotionSolver::writeMeshPointsToDatabase() 
{
    const auto& meshPoints = fvMesh_.points();

    if (solutionDim_ == 3) // 3D case
    {
        // Send existing 3D mesh points for forward inference: nPoints,
        // dim=3. Saves time and memory in avoiding to create a 2D point buffer. 
        client_.put_tensor(rankMeshPointsName_,
                           meshPoints.cdata(), 
                           {size_t(meshPoints.size()), 3},
                           SRTensorTypeDouble, 
                           SRMemLayoutContiguous);
    }
    else if (solutionDim_ == 2) // OpenFOAM pseudo 2D case
    {
        // Initialize local buffer for 2D points.
        std::vector<double> points2D;
        // For adaptive meshing, time-saving push_back below.
        points2D.resize(meshPoints.size() * solutionDim_);
        // Fill the points2D buffer with 2D mesh points
        forAll(meshPoints, pointI)
        {
            // Assign components in points2D using valid solution directions and
            // 3D meshPoint data.
            points2D[2*pointI] = meshPoints[pointI][validCmpts_[0]];
            points2D[2*pointI + 1] = meshPoints[pointI][validCmpts_[1]];
        } 

        // Send points2D to SmartRedis
        client_.put_tensor(
            rankMeshPointsName_,
            points2D.data(), 
            {size_t(meshPoints.size()), size_t(solutionDim_)},
            SRTensorTypeDouble, 
            SRMemLayoutContiguous
        );
    }
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
    clusterMode_(this->coeffDict().get<bool>("clusterMode")), 
    client_(clusterMode_), 
    solutionDim_(
        std::accumulate( 
            fvMesh_.solutionD().cbegin(),   
            fvMesh_.solutionD().cend(),     
            0                               
        )
    ),
    validCmpts_(filterValidCmpts(fvMesh_.solutionD())),
    rankMeshPointsName_("points_MPI_" + std::to_string(Pstream::myProcNo())),
    rankMeshDisplacementsName_("displacements_MPI_" + std::to_string(Pstream::myProcNo())),
    boundaryPoints_(), 
    boundaryDisplacements_()
{
    writeSolutionDimToDatabase();
    writeMeshPointsToDatabase();
}

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
    clusterMode_(dict.getOrDefault<bool>("clusterMode", true)),
    client_(clusterMode_),
    solutionDim_(
        std::accumulate( 
            fvMesh_.solutionD().cbegin(),   
            fvMesh_.solutionD().cend(),     
            0                               
        )
    ),
    validCmpts_(filterValidCmpts(fvMesh_.solutionD())),
    rankMeshPointsName_("points_MPI_" + std::to_string(Pstream::myProcNo())),
    rankMeshDisplacementsName_("displacements_MPI_" + std::to_string(Pstream::myProcNo())),
    boundaryPoints_(), 
    boundaryDisplacements_()
{
    writeSolutionDimToDatabase();
    writeMeshPointsToDatabase();
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::displacementSmartSimMotionSolver::
~displacementSmartSimMotionSolver() {}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::pointField> Foam::displacementSmartSimMotionSolver::curPoints() const
{
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

    // Assemble and send boundary points and their displacements to SmartRedis 

    // - Agglomerate boundary points and displacements for the MPI rank
    const auto& boundaryDisplacements = pointDisplacement().boundaryField();
    const auto& meshBoundary = motionSolver::mesh().boundaryMesh(); 
    List<point> mpiRankPoints;
    List<vector> mpiRankDisplacements;
    forAll(boundaryDisplacements, patchI)
    {
        if (meshBoundary[patchI].type() == "empty"
            || meshBoundary[patchI].type() == "processor")
        {
            continue;
        }

        const polyPatch& patch   = meshBoundary[patchI];
        const pointField& pts    = patch.localPoints();
        tmp<vectorField> dispPtr = boundaryDisplacements[patchI].patchInternalField();
        const vectorField& disp  = dispPtr();

        forAll(pts, i)
        {
            mpiRankPoints.append(pts[i]);
            mpiRankDisplacements.append(disp[i]);
        }
    }

    // - Prepare lobal displacement and point lists for gather
    List<List<point>>   globalPointListList(Pstream::nProcs());
    List<List<vector>>  globalDisplacementListList(Pstream::nProcs());

    // - Assign data in the lobal lists list from this MPI rank
    globalPointListList[Pstream::myProcNo()] = mpiRankPoints;
    globalDisplacementListList[Pstream::myProcNo()] = mpiRankDisplacements;

    // - Gather all data from all ranks at the main rank (0)
    Pstream::gatherList(globalPointListList);
    Pstream::gatherList(globalDisplacementListList);

    // - Send data to SmartRedis for ML model training from the main rank (0)
    if (Pstream::myProcNo() == 0)
    {
        // - Compute the global number of boundary points and displacements. 
        label nGlobalBoundaryPoints = 0; 
        forAll(globalPointListList, rankI)
        {
            nGlobalBoundaryPoints += globalPointListList[rankI].size();
        }
        
        // - Resize agglomerated point and displacement data to equal size. 
        boundaryPoints_.resize(nGlobalBoundaryPoints * solutionDim_);
        boundaryDisplacements_.resize(nGlobalBoundaryPoints * solutionDim_);

        // - Agglomerate the gathered boundary List<List<vector>> points and
        // displacements into boundaryPoints_ and boundaryDisplacements_ attributes. 
        label globalCmptI = 0;
        forAll(globalPointListList, rankI)
        {
            // Get the list of points from each rank 
            const List<point>& rankPoints = globalPointListList[rankI];
            // Get the list of displacements from each rank 
            const List<point>& rankDisplacements = globalDisplacementListList[rankI];

            // Assign rank points and rank displacements to boundaryPoints_ and
            // boundaryDisplacements_. 
            // meshPoints [1,2,3],[4,5,6]
            // validCmpts [0,2] - xz axis is the solution plane.
            // globalPoints_ = [1,3,4,6] - viewed as [1,3], [4,6].   

            // Iteration step is therefore point * solution dimension for
            // globalPoints_  and globalDisplacements_
            forAll(rankPoints, pointI)
            {
                forAll(validCmpts_, dimI)
                {
                    boundaryPoints_[globalCmptI] = rankPoints[pointI][validCmpts_[dimI]];
                    boundaryDisplacements_[globalCmptI] = rankDisplacements[pointI][validCmpts_[dimI]];
                    ++globalCmptI;
                }
            }
        }

        // Send points to SmartRedis for ML model training.
        client_.put_tensor(
            "points",
            boundaryPoints_.data(), 
            {size_t(nGlobalBoundaryPoints), size_t(solutionDim_)},
            SRTensorTypeDouble, 
            SRMemLayoutContiguous
        );

        client_.put_tensor(
            "displacements",
            boundaryDisplacements_.data(), 
            {size_t(nGlobalBoundaryPoints), size_t(solutionDim_)},
            SRTensorTypeDouble, 
            SRMemLayoutContiguous
        );

        client_.put_tensor(
            "data_ready", 
            &solutionDim_, 
            {1},
            SRTensorTypeInt32, 
            SRMemLayoutContiguous
        );
    }

    // Refresh points_MPI_<rank> with current mesh points.
    writeMeshPointsToDatabase();  // TODO(TM): can we remove this using points0 displacements?

    bool model_ready = client_.poll_key("model_ready", 1, 10000);
    if (! model_ready)
    {
        FatalErrorInFunction
            << "Displacement model not available in the SmartRedis database."
            << exit(Foam::FatalError);
    }
    else // Perform forward inference in the database and assign rank-displacements 
    {
        // Perform the forward inference in SmartRedis
        client_.run_model(
            "model", 
            {rankMeshPointsName_}, 
            {rankMeshDisplacementsName_}
        );

        // Allocate the displacements buffer.
        const auto& meshPoints = fvMesh_.points();
        std::vector<double> rankMeshDisplacements(
            meshPoints.size() * solutionDim_,
            0
        );

        // Unpack into the allocated displacements
        client_.unpack_tensor(
            rankMeshDisplacementsName_,
            rankMeshDisplacements.data(),
            {rankMeshDisplacements.size()},
            SRTensorTypeDouble,
            SRMemLayoutContiguous
        );   

        label globalId = 0;
        pointVectorField newDisplacement("newDisplacement", pointDisplacement_);
        forAll(pointDisplacement_, pointI)
        {
            forAll(validCmpts_, cmptI)
            {
                newDisplacement[pointI][validCmpts_[cmptI]] = rankMeshDisplacements[globalId];
                ++globalId;
            }
        }
        //newDisplacement.boundaryFieldRef().evaluate(); 
        pointDisplacement_.internalFieldRef() = newDisplacement.internalField(); 
        pointDisplacement_.boundaryFieldRef().evaluate(); 
    }

    // At the end of the simulation, have MPI rank 0 notify the python 
    // client via SmartRedis that the simulation has completed by writing
    // an end_time_index tensor to SmartRedis. 
    const auto& runTime = fvMesh_.time();
    if ((Pstream::myProcNo() == 0) &&  
        (runTime.timeOutputValue() >= runTime.endTime().value()))
    {
        std::vector<double> end_time_vec {double(runTime.timeIndex())};
        Info << "Seting end time flag : " << end_time_vec[0] << endl;
        client_.put_tensor(
            "final_iteration", 
            end_time_vec.data(), 
            {1}, 
            SRTensorTypeDouble, SRMemLayoutContiguous
        );
    }

    // Emulate MPI_Barrier() - wait for all MPI ranks to perform forward
    // inference of displacements and move the mesh with ML displacements.
    label totalRank = Pstream::myProcNo();
    reduce(totalRank, sumOp<label>(), totalRank);

    if (Pstream::myProcNo() == 0)
        client_.delete_tensor("model_ready");
}

// ************************************************************************* //
