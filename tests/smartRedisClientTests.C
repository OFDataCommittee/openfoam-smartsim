#include "IOobject.H"
#include "PstreamReduceOps.H"
#include "catch2/catch_all.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fvCFD.H"
#include "fvMesh.H"

#include "smartRedisClient.H"

using namespace Foam;
extern Time* timePtr;
extern argList* argsPtr;

TEST_CASE("standard naming convention", "[cavity][serial][parallel]")
{
    Time& runTime = *timePtr;
    fvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    dictionary dict0;
    dict0.set("region", polyMesh::defaultRegion);
    dict0.set("clusterMode", false);
    dict0.set("clientName", "default");
    smartRedisClient db("db0", runTime, dict0);

    // Dataset name should be always ready through the naming convention state
    dictionary schemeVals = db.namingConventionState();
    REQUIRE(db.extractName("dataset", schemeVals) == word("db0_time_index_0_mpi_rank_")+Foam::name(Pstream::myProcNo()));
    // For a specific field, the convention state needs to be updated
    schemeVals.subDict("field").set<string>("name", "p");
    schemeVals.subDict("field").set<string>("patch", "internal");
    REQUIRE(db.extractName("field", schemeVals) == "field_name_p_patch_internal");
}

TEST_CASE("packing of scalar fields with patches to a dataset", "[cavity][serial][parallel]")
{
    Time& runTime = *timePtr;
    fvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    volScalarField p
    (
        IOobject
        (
            "p",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true
        ),
        mesh,
        dimensionedScalar("p", dimPressure, 0.0)
    );
    dictionary dict0;
    dict0.set("region", polyMesh::defaultRegion);
    dict0.set("clusterMode", false);
    dict0.set("clientName", "default");
    smartRedisClient db("db0", runTime, dict0);

    DataSet ds(db.name() + "_test");
    wordList fields{"p"};
    wordList patches{"internal", "movingWall"};
    db.packFields<volScalarField>(ds, fields, patches);
    bool fieldsExistOnDB = true;
    bool fieldsDimsMatch = true;
    auto ts = ds.get_tensor_names();
    auto ndims = [&mesh] (const word& pName)
    {
        label patch = mesh.boundaryMesh().findPatchID(pName);
        return pName == "internal" ? mesh.nCells() : mesh.boundaryMesh()[patch].size();
    };
    forAll(fields, i){
        forAll(patches, j){
            if (patches[j] != "internal" && mesh.boundaryMesh()[patches[j]].size() == 0) continue;
            dictionary schemeValues = db.namingConventionState();
            schemeValues.subDict("field").set<string>("name", fields[i]);
            schemeValues.subDict("field").set<string>("patch", patches[j]);
            auto fName = db.extractName("field", schemeValues);
            fieldsExistOnDB = fieldsExistOnDB && (std::find(ts.begin(), ts.end(), fName) != ts.end());
            if (fieldsExistOnDB) {
                const auto& dims = ds.get_tensor_dims(fName);
                fieldsDimsMatch = fieldsDimsMatch && (dims[0]*dims[1] == ndims(patches[j])*pTraits<scalar>::nComponents);
            }
        }
    }
    CHECK(fieldsExistOnDB);
    REQUIRE(fieldsDimsMatch);
}

TEST_CASE("getting scalar fields from a dataset with patches", "[cavity][serial][parallel]")
{
    Time& runTime = *timePtr;
    fvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    volScalarField p
    (
        IOobject
        (
            "p",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true
        ),
        mesh,
        dimensionedScalar("p", dimPressure, 0.0)
    );
    dictionary dict0;
    dict0.set("region", polyMesh::defaultRegion);
    dict0.set("clusterMode", false);
    dict0.set("clientName", "default");
    smartRedisClient db("db0", runTime, dict0);

    DataSet ds(db.name() + "_test");
    wordList fields{"p"};
    wordList patches{"internal", "movingWall"};
    auto patchID = mesh.boundaryMesh().findPatchID(patches[1]);
    db.packFields<volScalarField>(ds, fields, patches);
    forAll(p, ci) {
        p[ci] = 1.0;
    }
    forAll(p.boundaryField()[patchID], fi){
        p.boundaryFieldRef()[patchID][fi] = 1.0;
    }
    db.getFields<volScalarField>(ds, fields, patches);
    bool hasExpectedValues = true;
    forAll(p, ci){
        hasExpectedValues = hasExpectedValues && (p[ci] == 0.0);
    }
    forAll(p.boundaryField()[patchID], fi){
        hasExpectedValues = hasExpectedValues && (p.boundaryField()[patchID][fi] == 0.0);
    }
    REQUIRE(hasExpectedValues);
}

TEST_CASE("send a list of fields of different types to the DB", "[cavity][serial][parallel]")
{
    Time& runTime = *timePtr;
    fvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    volScalarField p
    (
        IOobject
        (
            "p",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true
        ),
        mesh,
        dimensionedScalar("p0", dimPressure, 0.0)
    );
    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true
        ),
        mesh,
        dimensionedVector("U0", dimVelocity, vector::zero)
    );
    surfaceScalarField phi
    (
        IOobject
        (
            "phi",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true
        ),
        linearInterpolate(U) & mesh.Sf()
    );
    dictionary dict0;
    dict0.set("region", polyMesh::defaultRegion);
    dict0.set("clusterMode", false);
    dict0.set("clientName", "default");
    smartRedisClient db("db0", runTime, dict0);

    wordList fields{"p", "U", "phi"};
    labelList ncomponents{1, 3, 1};
    db.sendGeometricFields(fields);
    bool fieldsExistOnDB = true;
    bool fieldsDimsMatch = true;
    for (int i = 0; i < fields.size(); i++) {
        word fName = db.getDBFieldName(fields[i]);
        fieldsExistOnDB = fieldsExistOnDB && (db.client().tensor_exists(fName));
    }
    REQUIRE(fieldsExistOnDB);
}

TEST_CASE("get a list of fields of different types from the DB", "[cavity][serial][parallel]")
{
    Time& runTime = *timePtr;
    fvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    volScalarField p
    (
        IOobject
        (
            "p",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true
        ),
        mesh,
        dimensionedScalar("p0", dimPressure, 0.0)
    );
    volVectorField U
    (
        IOobject
        (
            "U",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true
        ),
        mesh,
        dimensionedVector("U0", dimVelocity, vector::zero)
    );
    dictionary dict0;
    dict0.set("region", polyMesh::defaultRegion);
    dict0.set("clusterMode", false);
    dict0.set("clientName", "default");
    smartRedisClient db("db0", runTime, dict0);

    wordList fields{"p", "U"};
    labelList ncomponents{1, 3};
    db.sendGeometricFields(fields);
    forAll(p, ci)
    {
        p[ci] = 1;
    }
    forAll(U, ci)
    {
        U[ci] = Foam::vector{1, 1, 1};
    }
    db.getGeometricFields(fields);
    bool allValuesMatch = true;
    forAll(p, ci)
    {
        allValuesMatch = allValuesMatch && (p[ci] == 0);
    }
    forAll(U, ci)
    {
        allValuesMatch = allValuesMatch && (U[ci] == vector::zero);
    }
    REQUIRE(allValuesMatch);
}

TEMPLATE_TEST_CASE
(
    "Generic send and recieve of list of values to SmartRedis",
    "[cavity][serial][parallel]",
    scalar, vector, tensor
)
{
    Time& runTime = *timePtr;
    fvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    dictionary dict0;
    dict0.set("region", polyMesh::defaultRegion);
    dict0.set("clusterMode", false);
    dict0.set("clientName", "default");
    smartRedisClient db("db0", runTime, dict0);

    SECTION("Send a list of values to SmartRedis") {
        List<TestType> lst(12);
        db.sendList(lst, "lst");
        REQUIRE(db.client().tensor_exists("lst_mpirank_"+Foam::name(Pstream::myProcNo())));
    }

    SECTION("Get a list of values from SmartRedis") {
        List<TestType> lst(12);
        for(auto& e : lst) {
            constexpr int nComponents = pTraits<TestType>::nComponents;
            if constexpr (nComponents > 1)
            {
                for (int i = 0; i < nComponents; i++) {
                    e[i] = 0;
                }
            } else {
                e = 0;
            }
        }
        db.sendList(lst, "lst");
        List<TestType> rcv(12);
        for(auto& e : rcv) {
            constexpr int nComponents = pTraits<TestType>::nComponents;
            if constexpr (nComponents > 1)
            {
                for (int i = 0; i < nComponents; i++) {
                    e[i] = 1;
                }
            } else {
                e = 1;
            }
        }
        db.getList(rcv, "lst");
        REQUIRE(rcv == lst);
    }
}
