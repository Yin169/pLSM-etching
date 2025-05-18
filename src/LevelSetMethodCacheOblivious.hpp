#ifndef LEVEL_SET_METHOD_CACHE_OBLIVIOUS_HPP
#define LEVEL_SET_METHOD_CACHE_OBLIVIOUS_HPP

#include "LevelSetMethod.hpp"
#include <vector>
#include <eigen3/Eigen/Dense>
#include <iostream> // For std::cout, std::cerr

/**
 * Cache-Oblivious Level Set Method implementation.
 * This class extends the standard LevelSetMethod with a cache-oblivious algorithm
 * that aims to improve cache efficiency through recursive space-time decomposition.
 * It inherits privately to reuse implementation details but provides its own `evolve`.
 */
class LevelSetMethodCacheOblivious : private LevelSetMethod {
private:
    // Cached data for narrow band points for efficient computation during evolution
    struct CachedPointData {
        Eigen::Vector3d etchingRates; // Precomputed etching velocity vector V_material
        bool isGridBoundary;          // True if the point is on the physical boundary of the grid
    };
    std::vector<CachedPointData> narrowBandCachedData;
    
    // Parameters for the trapezoidal decomposition in the cache-oblivious algorithm
    struct TrapezoidParams {
        int timeStart, timeEnd;             // Time step range [t0, t1)
        int narrowBandStartIndex, narrowBandEndIndex; // Index range in narrowBandIndices [start, end)
        // dx0, dx1 are not used in this particular level set context but kept for potential future use or consistency with general CO algorithms.
    };
    
    // Precompute material properties and boundary status for all points in the current narrowBandIndices
    void precomputeNarrowBandPointData() {
        if (narrowBandIndices.empty()) {
            narrowBandCachedData.clear();
            return;
        }
        narrowBandCachedData.resize(narrowBandIndices.size());
        
        #pragma omp parallel for schedule(dynamic, 1024) // Consider guided or static
        for (size_t k = 0; k < narrowBandIndices.size(); ++k) {
            const int gridIdx = narrowBandIndices[k];
            std::string material = getMaterialAtPoint(gridIdx); // Inherited from LevelSetMethod
            
            Eigen::Vector3d rates;
            const auto it = materialProperties.find(material); // materialProperties is inherited
            if (it != materialProperties.end()) { 
                const auto& props = it->second;  
                const double lateral_etch = props.lateralRatio * props.etchRatio; 
                rates << -lateral_etch, -lateral_etch, -props.etchRatio; // Etching is typically negative
            } else {
                rates.setZero(); // Default to zero etch rate if material not found
            }
            narrowBandCachedData[k].etchingRates = rates;
            narrowBandCachedData[k].isGridBoundary = isOnBoundary(gridIdx); // isOnBoundary is inherited
        }
    }
    
    // Helper function to compute d(phi)/dt for a single point in the narrow band
    // k is the index within narrowBandIndices and narrowBandCachedData
    // gridIdx is the actual index in the global phi grid
    double computePhiTimeDerivativeAtPoint(int gridIdx, size_t k_narrowBand, const Eigen::VectorXd& currentPhi) {
        // Skip points on the physical boundary of the grid
        if (narrowBandCachedData[k_narrowBand].isGridBoundary) {
            return 0.0; // No change at boundary
        }
        
        const Eigen::Vector3d& V_material = narrowBandCachedData[k_narrowBand].etchingRates;
        
        DerivativeOperator Dop;
        spatialScheme->SpatialSch(gridIdx, currentPhi, GRID_SPACING, Dop); // spatialScheme and GRID_SPACING inherited
        
        // Advection term: -V_material . grad(phi)
        double advection_term = 0.0;
        advection_term += (V_material.x() > 0 ? V_material.x() * Dop.dxN : V_material.x() * Dop.dxP);
        advection_term += (V_material.y() > 0 ? V_material.y() * Dop.dyN : V_material.y() * Dop.dyP);
        advection_term += (V_material.z() > 0 ? V_material.z() * Dop.dzN : V_material.z() * Dop.dzP);
            
        // Curvature term: -alpha * K * |grad(phi)|
        double curvature_flow_term = 0.0;
        if (CURVATURE_WEIGHT > 0.0) { // CURVATURE_WEIGHT inherited
            double mean_k = computeMeanCurvature(gridIdx, currentPhi); // computeMeanCurvature inherited
            
            double gx_c = 0.5 * (Dop.dxP + Dop.dxN); 
            double gy_c = 0.5 * (Dop.dyP + Dop.dyN);
            double gz_c = 0.5 * (Dop.dzP + Dop.dzN);
            constexpr double epsilon_grad_mag = 1e-10;
            double grad_mag = std::sqrt(gx_c*gx_c + gy_c*gy_c + gz_c*gz_c + epsilon_grad_mag);
            
            curvature_flow_term = CURVATURE_WEIGHT * mean_k * grad_mag;
        }
        
        return -advection_term - curvature_flow_term;
    }
    
    // Base case for the trapezoid recursion: process a single time step for a range of narrow band points
    // This is essentially one step of Forward Euler for the specified points.
    void processTrapezoidBase(int currentTimeStepIdx, int narrowBandStart, int narrowBandEnd, 
                             const Eigen::VectorXd& phi_at_t, Eigen::VectorXd& phi_at_t_plus_dt) {
        // phi_at_t_plus_dt should be initialized (e.g. to phi_at_t or zeros if accumulating changes)
        // Assuming phi_at_t_plus_dt is initially a copy of phi_at_t for this Forward Euler step.
        // Or, if phi_steps are distinct, phi_at_t_plus_dt[idx] = phi_at_t[idx] + dt * L_phi_idx;

        #pragma omp parallel for schedule(dynamic, 1024) // Chunk size might need tuning
        for (int k = narrowBandStart; k < narrowBandEnd; ++k) {
            const int gridIdx = narrowBandIndices[k]; // narrowBandIndices is inherited
            
            // If phi_at_t_plus_dt is not pre-filled with phi_at_t, do it here for non-updated points.
            // However, typically only narrow band points are evolved.
            // This function computes L(phi) and the caller (processTrapezoid or evolve) does time integration.
            // Let's assume this computes L(phi) and stores it in phi_at_t_plus_dt for now.
            // The evolve method's structure suggests phi_steps[t+1] is built from phi_steps[t].
            
            double L_phi_idx = computePhiTimeDerivativeAtPoint(gridIdx, k, phi_at_t);
            phi_at_t_plus_dt[gridIdx] = phi_at_t[gridIdx] + dt * L_phi_idx; // dt is inherited
        }
    }
    
    // Recursive trapezoid decomposition for cache-oblivious processing
    // phi_steps is a vector of phi states, phi_steps[t] is the SDF at time step t.
    void processTrapezoidRecursive(const TrapezoidParams& params, std::vector<Eigen::VectorXd>& phi_all_steps) {
        const int timeRange = params.timeEnd - params.timeStart;
        const int spaceRange = params.narrowBandEndIndex - params.narrowBandStartIndex;

        if (spaceRange <= 0 || timeRange <= 0) return; // Nothing to process

        // Base case: if the time range is small enough (e.g., 1 time step), process directly.
        // Or if space range is small enough. Thresholds might need tuning.
        // For this algorithm, the base case is typically a single time step.
        if (timeRange == 1) {
            // Ensure phi_all_steps[params.timeEnd] is correctly initialized or handled.
            // If phi_all_steps[t] stores phi at start of step t, then result goes to phi_all_steps[t+1]
            // The call is processTrapezoidBase(t0, start, end, phi_steps[t0], phi_steps[t0+1])
            processTrapezoidBase(params.timeStart, params.narrowBandStartIndex, params.narrowBandEndIndex, 
                                phi_all_steps[params.timeStart], phi_all_steps[params.timeStart + 1]); // timeEnd = timeStart + 1
            return;
        }
        
        // Heuristic: if width (space) is much larger than height (time), cut in space. Otherwise, cut in time.
        // A common heuristic is if space_range >= C * time_range (C often 1 or 2).
        if (spaceRange >= 2 * timeRange) { // Cut in space
            int midNarrowBandIndex = params.narrowBandStartIndex + spaceRange / 2;
            
            TrapezoidParams leftParams = params;
            leftParams.narrowBandEndIndex = midNarrowBandIndex;
            processTrapezoidRecursive(leftParams, phi_all_steps);
            
            TrapezoidParams rightParams = params;
            rightParams.narrowBandStartIndex = midNarrowBandIndex;
            processTrapezoidRecursive(rightParams, phi_all_steps);
        } else { // Cut in time
            int midTimeStep = params.timeStart + timeRange / 2;
            
            TrapezoidParams bottomParams = params;
            bottomParams.timeEnd = midTimeStep;
            processTrapezoidRecursive(bottomParams, phi_all_steps); // Process [t0, t_mid)
            
            // After bottom part is done up to t_mid, process top part from t_mid
            TrapezoidParams topParams = params;
            topParams.timeStart = midTimeStep; // Process [t_mid, t1)
            processTrapezoidRecursive(topParams, phi_all_steps);
        }
    }

    // Parallel version of the recursive trapezoid decomposition using OpenMP tasks
    void processTrapezoidRecursiveParallel(const TrapezoidParams& params, std::vector<Eigen::VectorXd>& phi_all_steps) {
        const int timeRange = params.timeEnd - params.timeStart;
        const int spaceRange = params.narrowBandEndIndex - params.narrowBandStartIndex;

        if (spaceRange <= 0 || timeRange <= 0) return;

        // Base case for recursion (e.g., single time step or small enough problem)
        // Thresholds can be tuned. For simplicity, base case is one time step.
        // A larger base case (e.g. few time steps or small spatial range) might reduce tasking overhead.
        const int MIN_TIME_RANGE_FOR_TASKS = 4; // Example threshold
        const int MIN_SPACE_RANGE_FOR_TASKS = 1024; // Example threshold

        if (timeRange == 1) {
            processTrapezoidBase(params.timeStart, params.narrowBandStartIndex, params.narrowBandEndIndex,
                                phi_all_steps[params.timeStart], phi_all_steps[params.timeStart + 1]);
            return;
        }
        
        // Heuristic for cutting
        if (spaceRange >= 2 * timeRange && spaceRange > MIN_SPACE_RANGE_FOR_TASKS) { // Cut in space (parallel tasks)
            int midNarrowBandIndex = params.narrowBandStartIndex + spaceRange / 2;
            
            TrapezoidParams leftParams = params;
            leftParams.narrowBandEndIndex = midNarrowBandIndex;
            
            TrapezoidParams rightParams = params;
            rightParams.narrowBandStartIndex = midNarrowBandIndex;
            
            #pragma omp task // if (spaceRange > SOME_THRESHOLD_FOR_TASKING)
            processTrapezoidRecursiveParallel(leftParams, phi_all_steps);
            
            #pragma omp task // if (spaceRange > SOME_THRESHOLD_FOR_TASKING)
            processTrapezoidRecursiveParallel(rightParams, phi_all_steps);
            
            // No taskwait here needed if tasks are fully independent for space cuts.
            // However, if the base case modifies shared state or if there are dependencies not captured,
            // taskwait would be essential. For this level set, spatial points are usually independent for one time step.
            // If processTrapezoidBase has side effects beyond its phi_at_t_plus_dt, be cautious.
            // Assuming independence for spatial decomposition at a given time slice.

        } else { // Cut in time (sequential dependency) or if spaceRange is too small for effective tasking
            int midTimeStep = params.timeStart + timeRange / 2;
            
            TrapezoidParams bottomParams = params;
            bottomParams.timeEnd = midTimeStep;
            processTrapezoidRecursiveParallel(bottomParams, phi_all_steps); // Must complete first
            
            // #pragma omp taskwait // Ensure bottom is done before starting top if tasks were used for bottom.
                                 // Not needed if bottom call is synchronous.

            TrapezoidParams topParams = params;
            topParams.timeStart = midTimeStep;
            processTrapezoidRecursiveParallel(topParams, phi_all_steps);
        }
        // If tasks were created in this block, a taskwait might be needed before this function returns,
        // depending on how the caller manages tasks. The #pragma omp single implies one master thread
        // creates initial tasks, and taskwaits manage dependencies.
    }


public:
    // Constructor inherits from LevelSetMethod and initializes its members
    LevelSetMethodCacheOblivious(
                const std::string& meshFile,
                const std::string& orgFile, // Original mesh for material mapping
                const std::string& materialCsvFile,
                int gridSize = 400, 
                double timeStep = 0.01, 
                int maxSteps = 80, 
                int reinitInterval = 5,
                int narrowBandInterval = 100,
                double narrowBandWidth = 10.0, // In terms of grid cells
                int numThreads = -1,
                double curvatureWeight = 0.0,
                SpatialSchemeType spatialSchemeType = SpatialSchemeType::UPWIND,
                TimeSchemeType timeSchemeType = TimeSchemeType::FORWARD_EULER)
        : LevelSetMethod(meshFile, orgFile, materialCsvFile, gridSize, timeStep, maxSteps,
                        reinitInterval, narrowBandInterval, narrowBandWidth, numThreads,
                        curvatureWeight, spatialSchemeType, timeSchemeType) {
        // Base class constructor handles most initialization.
        // Specific initializations for CacheOblivious version can go here.
        std::cout << "Cache-Oblivious Level Set Method initialized." << std::endl;
    }
    
    // Override the evolve method to use the cache-oblivious algorithm
    bool evolve() override;

    // Provide access to base class's surface extraction if needed, or implement CO specific one
	void extractSurfaceMeshCGAL(const std::string& outputFile) {
        // Call the base class's implementation
		LevelSetMethod::extractSurfaceMeshCGAL(outputFile);
	}
};

#endif // LEVEL_SET_METHOD_CACHE_OBLIVIOUS_HPP
