#include "LevelSetMethodCacheOblivious.hpp"
#include <stdexcept>    // For std::out_of_range
#include <iostream>     // For std::cerr (error reporting)

/**
 * Override of the evolve method to use the cache-oblivious algorithm
 * This implementation uses a recursive space-time decomposition approach
 * to improve cache efficiency during level set evolution.
 */
bool LevelSetMethodCacheOblivious::evolve() {
    try {
        // Initialize the signed distance field and narrow band
        phi = initializeSignedDistanceField();
        updateNarrowBand();
        
        // Pre-compute material properties and boundary status for better cache efficiency
        precomputePointData();
        
        // Allocate space for all time steps
        std::vector<Eigen::VectorXd> phi_steps(STEPS + 1);
        phi_steps[0] = phi;
        
        // Initialize all time steps with zero vectors
        for (int i = 1; i <= STEPS; ++i) {
            phi_steps[i] = Eigen::VectorXd::Zero(phi.size());
        }
        
        // Define the trapezoid for the entire computation
        TrapezoidParams mainTrapezoid;
        mainTrapezoid.t0 = 0;
        mainTrapezoid.t1 = STEPS;
        mainTrapezoid.startIdx = 0;
        mainTrapezoid.endIdx = narrowBand.size();
        mainTrapezoid.dx0 = 0;  // Not used in this implementation
        mainTrapezoid.dx1 = 0;  // Not used in this implementation
        
        // Process the main trapezoid with parallel execution
        #pragma omp parallel
        {
            #pragma omp single
            processParallelTrapezoid(mainTrapezoid, phi_steps);
        }
        
        // Final phi is the last time step
        phi = phi_steps[STEPS];
        
        // Perform reinitialization to maintain signed distance property
        reinitialize();
        
        std::cout << "Cache-oblivious evolution completed successfully." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during cache-oblivious evolution: " << e.what() << std::endl;
        return false;
    }
}