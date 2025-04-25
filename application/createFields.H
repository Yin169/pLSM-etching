Info<< "Reading field phi (level set)\n" << endl; // Clarify which phi

volScalarField phi // This is the level set field
(
    IOobject
    (
        "phi",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;

volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Calculating velocity flux phiFlux\n" << endl; // New flux field

surfaceScalarField phiFlux // Renamed from phi to phiFlux
(
    IOobject
    (
        "phiFlux", // Use a distinct name
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT, // Don't require reading, calculate it
        IOobject::AUTO_WRITE
    ),
    fvc::interpolate(U) & mesh.Sf() // Calculate flux U dot surfaceNormalArea
);

// Read transport properties
IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

// Read re-initialization interval
int reInitInterval = readInt(transportProperties.lookup("reInitInterval"));
