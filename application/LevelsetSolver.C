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
            
            // Read angular distribution parameters
            const scalar sigma = etchingProperties.lookupOrDefault<scalar>("angularSigma", 0.2);
            const vector referenceDirection = etchingProperties.lookupOrDefault<vector>("referenceDirection", vector(0, 0, 1));
            const vector normalizedRefDir = referenceDirection/mag(referenceDirection);
            
            // Calculate direction-dependent etching rate using the flux equation
            // dF = (r̂·n̂)e^(-θ/2σ²)dΩ
            forAll(etchRate, cellI)
            {
                // Only calculate for cells near the interface
                if (mag(phi[cellI]) < 2.0*mesh.deltaCoeffs()[cellI])
                {
                    // Get the surface normal at this cell
                    vector n = normal[cellI];
                    
                    // Calculate the cosine of the angle between normal and reference direction
                    scalar cosTheta = (n & normalizedRefDir);
                    
                    // Calculate the angle (θ) in radians
                    scalar theta = Foam::acos(min(1.0, max(-1.0, cosTheta)));
                    
                    // Apply the angular distribution function
                    // The dot product (r̂·n̂) is already included in cosTheta
                    // We use max(0, cosTheta) to ensure flux is only in the direction of the normal
                    scalar angularFactor = max(0.0, cosTheta) * Foam::exp(-theta*theta/(2.0*sigma*sigma));
                    
                    // Set the etch rate based on the angular distribution
                    etchRate[cellI] = baseEtchRate * angularFactor;
                }
            }
            
            Info<< "Using direction-dependent etching rate model with angular distribution" << endl;
        }
        else if (etchRateModel == "angularFlux")
        {
            // Implement the full angular flux model from the equation
            // dF = (r̂·n̂)e^(-θ/2σ²)dΩ
            
            // Read model parameters
            const scalar sigma = etchingProperties.lookupOrDefault<scalar>("angularSigma", 0.2);
            const label nDirections = etchingProperties.lookupOrDefault<label>("nDirections", 10);
            const scalar fluxIntensity = etchingProperties.lookupOrDefault<scalar>("fluxIntensity", 1.0);
            
            // Calculate surface normals
            tmp<volVectorField> tgradPhi = fvc::grad(phi);
            const volVectorField& gradPhi = tgradPhi.ref();
            
            tmp<volVectorField> tnormal = gradPhi/(mag(gradPhi) + dimensionedScalar("small", dimless, SMALL));
            const volVectorField& normal = tnormal.ref();
            
            // Reset etch rate field
            etchRate = dimensionedScalar("zero", etchRate.dimensions(), 0.0);
            
            // Create a hemisphere of directions for integration
            List<vector> directions;
            List<scalar> weights;
            
            // Simple hemisphere discretization
            // In a real implementation, you would use a more sophisticated method
            scalar dTheta = M_PI / (2.0 * nDirections);
            scalar dPhi = 2.0 * M_PI / nDirections;
            
            for (label i = 0; i < nDirections; i++)
            {
                scalar theta = i * dTheta; // Angle from z-axis (0 to π/2)
                
                for (label j = 0; j < nDirections; j++)
                {
                    scalar phi = j * dPhi; // Azimuthal angle (0 to 2π)
                    
                    // Convert spherical to Cartesian coordinates
                    vector dir(
                        Foam::sin(theta) * Foam::cos(phi),
                        Foam::sin(theta) * Foam::sin(phi),
                        Foam::cos(theta)
                    );
                    
                    // Weight is proportional to solid angle
                    scalar weight = Foam::sin(theta) * dTheta * dPhi;
                    
                    directions.append(dir);
                    weights.append(weight);
                }
            }
            
            // Calculate etch rate by integrating over all directions
            forAll(etchRate, cellI)
            {
                // Only calculate for cells near the interface
                if (mag(phi[cellI]) < 2.0*mesh.deltaCoeffs()[cellI])
                {
                    // Get the surface normal at this cell
                    vector n = normal[cellI];
                    
                    scalar totalFlux = 0.0;
                    
                    // Integrate over all directions
                    forAll(directions, dirI)
                    {
                        vector r = directions[dirI];
                        scalar weight = weights[dirI];
                        
                        // Calculate r̂·n̂ (dot product)
                        scalar dotProduct = r & n;
                        
                        // Only consider directions pointing toward the surface
                        if (dotProduct > 0.0)
                        {
                            // Calculate angle between direction and reference (z-axis)
                            vector refDir(0, 0, 1);
                            scalar cosTheta = (r & refDir);
                            scalar theta = Foam::acos(min(1.0, max(-1.0, cosTheta)));
                            
                            // Calculate flux contribution using the equation
                            // dF = (r̂·n̂)e^(-θ/2σ²)dΩ
                            scalar fluxContribution = dotProduct * 
                                                     Foam::exp(-theta*theta/(2.0*sigma*sigma)) * 
                                                     weight;
                            
                            totalFlux += fluxContribution;
                        }
                    }
                    
                    // Set etch rate proportional to total flux
                    etchRate[cellI] = baseEtchRate * fluxIntensity * totalFlux;
                }
            }
            
            Info<< "Using angular flux integration etching rate model" << endl;
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
        
        // Create a field to precisely track the zero level set (phi == 0)
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
        
        // Calculate the exact zero level set using sign change detection
        // This is more precise than the threshold-based interface field
        forAll(mesh.cells(), cellI)
        {
            // Get neighboring cells
            const labelList& neighborCells = mesh.cellCells()[cellI];
            
            // Check if phi changes sign across this cell and its neighbors
            bool signChange = false;
            
            if (neighborCells.size() > 0)
            {
                scalar cellPhi = phi[cellI];
                
                forAll(neighborCells, nI)
                {
                    scalar neighborPhi = phi[neighborCells[nI]];
                    
                    // If phi changes sign between this cell and neighbor, 
                    // the zero level set passes through
                    if (cellPhi * neighborPhi <= 0.0)
                    {
                        signChange = true;
                        break;
                    }
                }
            }
            
            if (signChange)
            {
                zeroLevelSet[cellI] = 1.0;
            }
        }
 
        
        // Write fields
        etchRate.write();
        interface.write();
        zeroLevelSet.write();
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