#include "LevelSetMethodCacheOblivious.hpp"
#include <stdexcept>    // For std::runtime_error
#include <iostream>     // For std::cerr, std::cout
#include <vector>
#include <eigen3/Eigen/Dense>

/**
 * Override of the evolve method to use the cache-oblivious algorithm.
 * This implementation uses a recursive space-time decomposition approach
 * to improve cache efficiency during level set evolution.
 * It performs a simple Forward Euler time integration within the cache-oblivious structure.
 */
bool LevelSetMethodCacheOblivious::evolve() {
    try {
        // Initialize phi (SDF) and narrowBandIndices from base class
        this->phi = initializeSignedDistanceField(); // `this->` is optional but clear for base members
        this->updateNarrowBand();
        
        if (narrowBandIndices.empty() && STEPS > 0) {
            std::cerr << "Warning: Initial narrow band is empty. Evolution might not proceed as expected." << std::endl;
            // Depending on the problem, this might be an error or an expected state.
        }

        // Pre-compute data for points in the narrow band (etching rates, boundary status)
        precomputeNarrowBandPointData(); 
        
        // Allocate space for phi at all time steps. phi_all_steps[t] is phi at the START of step t.
        // So, phi_all_steps[0] is initial phi. phi_all_steps[STEPS] will be the final phi.
        std::vector<Eigen::VectorXd> phi_all_steps(STEPS + 1);
        phi_all_steps[0] = this->phi; // Initial SDF
        
        // Initialize subsequent phi states. They will be computed.
        // If processTrapezoidBase directly computes phi_all_steps[t+1] from phi_all_steps[t],
        // then explicit zeroing might not be needed if all narrow band points are updated.
        // However, points outside narrow band (if any were included in phi_all_steps[t].size())
        // would need to be copied from phi_all_steps[t] to phi_all_steps[t+1].
        // For simplicity, assume full vectors are used and non-narrowband points are copied.
        for (int t = 0; t < STEPS; ++t) {
            phi_all_steps[t+1] = phi_all_steps[t]; // Initialize next step's phi with current as baseline
                                                 // The CO algorithm will then update narrow band points.
        }
        
        // Define the main trapezoid covering the entire computation domain (all time, all narrow band space)
        TrapezoidParams mainTrapezoid;
        mainTrapezoid.timeStart = 0;            // Start at time step 0
        mainTrapezoid.timeEnd = STEPS;          // Go up to (but not including) STEPS time steps (total STEPS evolutions)
                                                // So, if STEPS = 80, it's t=0 to t=79. phi_all_steps has size STEPS+1 (indices 0 to 80).
                                                // The loop in processTrapezoidBase will be for t = 0 to STEPS-1.
                                                // phi_all_steps[params.timeStart+1] will be written.
        mainTrapezoid.narrowBandStartIndex = 0;
        mainTrapezoid.narrowBandEndIndex = static_cast<int>(narrowBandIndices.size());
        
        std::cout << "Starting cache-oblivious evolution for " << STEPS << " steps..." << std::endl;
        const int progressInterval = std::max(1, STEPS / 20);


        // The recursive processing happens here.
        // If parallel tasks are used inside, this needs to be in a parallel region.
        // The #pragma omp single ensures only one thread initiates the recursive calls.
        // Note: The current `processTrapezoidRecursiveParallel` and `processTrapezoidBase`
        // implement a Forward Euler step. If a different time scheme (e.g., RK3) from the base class
        // is desired with the cache-oblivious structure, the `processTrapezoidBase` and the
        // overall `phi_all_steps` management would need significant adaptation.
        // This CO version currently implements its own Forward Euler.

        // If STEPS is 0, skip evolution.
        if (STEPS > 0) {
            #pragma omp parallel // Create the team of threads
            {
                #pragma omp single // Only one thread starts the recursive decomposition
                {
                    processTrapezoidRecursiveParallel(mainTrapezoid, phi_all_steps);
                }
            } // Threads synchronize here
        }
        
        // Final phi is the state after all STEPS evolutions
        this->phi = phi_all_steps[STEPS];
        
        // Perform reinitialization and narrow band update as per base class logic (or override if needed)
        // These are called after the main CO evolution loop.
        // The reinitialization interval logic is not naturally part of this CO structure,
        // which processes all time steps together.
        // For a true CO approach with periodic reinit/NB update, the CO decomposition
        // would need to be applied to segments of time steps between these operations.
        // Current implementation: reinit/NB update effectively happens *after* all CO steps.
        // This might deviate from the intended behavior if reinit/NB are crucial *during* evolution.

        // If reinitialization/narrow band updates are needed periodically *within* the CO evolution:
        // The evolve() loop would need to be structured differently, e.g.,
        // for (segment = 0 to num_segments):
        //      CO_evolve_for_segment_duration()
        //      reinitialize()
        //      updateNarrowBand()
        //      precomputeNarrowBandPointData()
        // This current structure does one large CO evolution then one reinit.

        std::cout << "Cache-oblivious evolution processing completed." << std::endl;
        if (REINIT_INTERVAL > 0 && STEPS > 0) { // Reinitialize at the end if configured
             std::cout << "Performing final reinitialization..." << std::endl;
             reinitialize(); // Uses the updated this->phi
        }
        // Final narrow band update might be useful if the surface is extracted next.
        if (NARROW_BAND_UPDATE_INTERVAL > 0 && STEPS > 0) { // NARROW_BAND_UPDATE_INTERVAL used as a flag
            std::cout << "Performing final narrow band update..." << std::endl;
            updateNarrowBand(); 
        }
        
        std::cout << "Cache-oblivious evolution finished successfully." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during cache-oblivious evolution: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown error during cache-oblivious evolution." << std::endl;
        return false;
    }
}

