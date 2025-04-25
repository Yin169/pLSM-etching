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
    Solver for the level set equation to track interfaces between fluids.
    The level set function φ is advected by the velocity field.

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
    
    #include "createFields.H" // This now creates phi (volScalarField) and phiFlux (surfaceScalarField)
    #include "createTimeControls.H"
    #include "initContinuityErrs.H"
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    
    Info<< "\nStarting level set solver\n" << endl;
    
    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
        
        #include "readTimeControls.H"
        // #include "CourantNo.H" // Remove this - Courant number handled by time controls
        
        // --- Recalculate flux if velocity changes (if U is solved for) ---
        // If U is static (just read), this might not be needed inside the loop
        // If U changes, uncomment:
        // phiFlux = fvc::interpolate(U) & mesh.Sf();
        
        // Solve the level set equation
        fvScalarMatrix phiEqn
        (
            fvm::ddt(phi)
          + fvm::div(phiFlux, phi) // Use the flux field phiFlux here
        );
        
        phiEqn.solve();
        
        // Re-initialization step to maintain phi as a signed distance function
        // if (reInitInterval > 0 && runTime.timeIndex() % reInitInterval == 0) // Check if reInitInterval > 0
        // {
        //     #include "reInitialization.H"
        // }
        
        runTime.write();
        
        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }
    
    Info<< "End\n" << endl;
    
    return 0;
}

// ************************************************************************* //