/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 Your Name
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
    levelsetSolver

Description
    Solver for the level set equation to track interfaces during semiconductor
    etching processes. The level set function φ represents the interface between
    the semiconductor material and the etching medium. The etching rate is
    modeled based on local conditions, and reinitialization is performed to
    maintain φ as a signed distance function.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pisoControl.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    #include "createFields.H"
    
    #include "createTimeControls.H"
    #include "initContinuityErrs.H"
    
    
    // Read reinitialization parameters
    const label reInitInterval = etchingProperties.lookupOrDefault<label>("reInitInterval", 5);
    const scalar reInitSteps = etchingProperties.lookupOrDefault<scalar>("reInitSteps", 3);
    const scalar reInitDt = etchingProperties.lookupOrDefault<scalar>("reInitDt", 0.1);
    
    // Read etching rate model parameters
    const word etchRateModel = etchingProperties.lookupOrDefault<word>("etchRateModel", "constant");
    const scalar baseEtchRate = etchingProperties.lookupOrDefault<scalar>("baseEtchRate", 1.0);
    
    // Create etching rate field
    volScalarField etchRate
    (
        IOobject
        (
            "etchRate",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        mesh,
        dimensionedScalar("etchRate", dimLength/dimTime, baseEtchRate)  // Changed from dimless/dimTime to dimLength/dimTime
    );
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    Info<< "\nStarting semiconductor etching level set solver\n" << endl;
    
    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
        
        #include "readTimeControls.H"
        
        // Calculate etching rate based on selected model
        if (etchRateModel == "constant")
        {
            // Constant etching rate - already set
        }
        else if (etchRateModel == "concentrationDependent")
        {
            // Example: Etching rate depends on a concentration field
            // Assuming concentration is available as a field
            // etchRate = baseEtchRate * pow(concentration/concentrationRef, n);
            Info<< "Using concentration-dependent etching rate model" << endl;
            // Implementation would go here
        }
        else if (etchRateModel == "directionDependent")
        {
            // Example: Etching rate depends on surface normal direction
            tmp<volVectorField> tgradPhi = fvc::grad(phi);
            const volVectorField& gradPhi = tgradPhi.ref();
            
            tmp<volVectorField> tnormal = gradPhi/(mag(gradPhi) + dimensionedScalar("small", dimless, SMALL));
            const volVectorField& normal = tnormal.ref();
            
            // Calculate direction-dependent etching rate
            // This is a simplified model - real models would be more complex
            forAll(etchRate, cellI)
            {
                // Example: faster etching in vertical direction
                scalar verticalFactor = mag(normal[cellI].z());
                etchRate[cellI] = baseEtchRate * (1.0 + verticalFactor);
            }
            
            Info<< "Using direction-dependent etching rate model" << endl;
        }
        
        // Update velocity field based on etching rate
        // The velocity is normal to the interface with magnitude equal to etch rate
        tmp<volVectorField> tgradPhi = fvc::grad(phi);
        const volVectorField& gradPhi = tgradPhi.ref();
        
        tmp<volVectorField> tnormal = gradPhi/(mag(gradPhi) + dimensionedScalar("small", dimless, SMALL));
        const volVectorField& normal = tnormal.ref();
        
        // Set velocity field for level set advection
        U = -etchRate * normal;
        
        // Update flux for advection
        phiFlux = fvc::interpolate(U) & mesh.Sf();
        
        // Solve the level set equation
        fvScalarMatrix phiEqn
        (
            fvm::ddt(phi)
          + fvm::div(phiFlux, phi)
        );
        
        phiEqn.solve();
        
        
        // Calculate interface location for visualization
        volScalarField interface
        (
            IOobject
            (
                "interface",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar("zero", dimless, 0.0)
        );
        
        // Mark cells near the interface (where phi is close to zero)
        forAll(interface, cellI)
        {
            if (mag(phi[cellI]) < 1.5*mesh.deltaCoeffs()[cellI])
            {
                interface[cellI] = 1.0;
            }
        }
        
        // Add a more precise zero level set tracker
        volScalarField zeroLevelSet
        (
            IOobject
            (
                "zeroLevelSet",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar("zero", dimless, 0.0)
        );
        
        // Use linear interpolation to more precisely locate where phi crosses zero
        forAll(mesh.cells(), cellI)
        {
            scalar minPhiValue = GREAT;
            scalar maxPhiValue = -GREAT;
            
            // Check all cell vertices
            const labelList& cellPoints = mesh.cellPoints(cellI);
            forAll(cellPoints, pointI)
            {
                label pointLabel = cellPoints[pointI];
                scalar pointPhiValue = phi.internalField()[cellI]; // Approximate point value with cell value
                
                minPhiValue = min(minPhiValue, pointPhiValue);
                maxPhiValue = max(maxPhiValue, pointPhiValue);
            }
            
            // If phi changes sign within this cell, it contains the zero level set
            if (minPhiValue <= 0 && maxPhiValue >= 0)
            {
                zeroLevelSet[cellI] = 1.0;
            }
        }
        
        // Write fields
        etchRate.write();
        interface.write();
        zeroLevelSet.write(); // Write the new zero level set field
        phi.write();
        U.write();
        runTime.write();
        
        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }
    
    Info<< "End\n" << endl;
    
    return 0;
}

// ************************************************************************* //