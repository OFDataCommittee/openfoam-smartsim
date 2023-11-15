#include "IOobject.H"
#include "PstreamReduceOps.H"
#include "catch2/catch_all.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fvCFD.H"
#include "fvMesh.H"

#include "fieldsToSmartRedisFunctionObject.H"
#include "functionObjectList.H"

using namespace Foam;
extern Time* timePtr;
extern argList* argsPtr;

TEST_CASE("call checkAllFields on fields", "[cavity][serial][parallel]")
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
    dict0.set("type", "fieldsToSmartRedisFunctionObject");
    dict0.set("fields", wordList());
    dict0.set("clusterMode", false);
    dict0.set("clientName", "default");
    functionObjects::fieldsToSmartRedisFunctionObject o0("smartSim0", runTime, dict0);

    SECTION("call checkAllFields on non-registred fields", "[!throws]")
    {
        wordList fields;
        fields.append("p");
        FatalError.throwExceptions();
        REQUIRE_THROWS(o0.checkAllFields<
            volScalarField,
            volVectorField,
            volTensorField,
            volSphericalTensorField,
            volSymmTensorField,
            surfaceScalarField,
            surfaceVectorField,
            surfaceTensorField,
            surfaceSphericalTensorField,
            surfaceSymmTensorField
        >(fields, mesh));
    }

    SECTION("call checkAllFields on registred fields")
    {
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
        wordList fields;
        fields.append("p");
        REQUIRE(o0.checkAllFields<
            volScalarField,
            volVectorField,
            volTensorField,
            volSphericalTensorField,
            volSymmTensorField,
            surfaceScalarField,
            surfaceVectorField,
            surfaceTensorField,
            surfaceSphericalTensorField,
            surfaceSymmTensorField
        >(fields, mesh));
    }
}

TEST_CASE("call sendAllFields on fields", "[cavity][serial][parallel]")
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
    dict0.set("type", "fieldsToSmartRedisFunctionObject");
    dict0.set("fields", wordList());
    dict0.set("clusterMode", false);
    dict0.set("clientName", "default");
    functionObjects::fieldsToSmartRedisFunctionObject o0("smartSim0", runTime, dict0);

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
        dimensionedVector("U", dimVelocity, vector::zero)
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
    wordList fields;
    fields.append("p");
    fields.append("U");
    fields.append("phi");
    auto dsName = o0.datasetName(Foam::name(runTime.timeIndex()), Foam::name(Pstream::myProcNo()));
    DataSet ds(dsName);

    o0.sendAllFields<
        volScalarField,
        volVectorField,
        volTensorField,
        volSphericalTensorField,
        volSymmTensorField,
        surfaceScalarField,
        surfaceVectorField,
        surfaceTensorField,
        surfaceSphericalTensorField,
        surfaceSymmTensorField
    >(ds, fields);

    // Is this supposed to be a blocking operation?
    // @todo is client.put_dataset a blocking operation?
    // @body do we have to poll for the DB 
    o0.client().put_dataset(ds);

    bool found = o0.client().poll_dataset(dsName, 10, 1000);
    CHECK(found);
    DataSet fetchedDS = o0.client().get_dataset(dsName);
    auto ts = fetchedDS.get_tensor_names();
    bool fieldsExistsOnDB = true;
    forAll(fields, i){
        auto fName = o0.fieldName(fields[i], "internal");
        fieldsExistsOnDB = fieldsExistsOnDB && (std::find(ts.begin(), ts.end(), fName) != ts.end());
    }
    REQUIRE(fieldsExistsOnDB);
}
