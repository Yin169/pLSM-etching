Info<< "Reading field phi (level set function)\n" << endl;

volScalarField phi
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

Info<< "Reading field etchRate\n" << endl;

volScalarField etchRate
(
    IOobject
    (
        "etchRate",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("etchRate", dimLength/dimTime, 0.0)
);

// Read angular flux parameters
IOdictionary etchProperties
(
    IOobject
    (
        "etchProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ,
        IOobject::NO_WRITE
    )
);

// Maximum etch rate
dimensionedScalar maxEtchRate
(
    "maxEtchRate",
    dimLength/dimTime,
    etchProperties.lookup("maxEtchRate")
);

// Angular dependence parameters
scalar n(readScalar(etchProperties.lookup("n")));
scalar m(readScalar(etchProperties.lookup("m")));
scalar theta0(readScalar(etchProperties.lookup("theta0")));

// Incident flux direction
vector fluxDirection
(
    etchProperties.lookup("fluxDirection")
);
fluxDirection = fluxDirection/mag(fluxDirection);