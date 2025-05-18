// Cache-Oblivious Level Set Method implementation
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

class LevelSetMethodCacheOblivious {
private:
    // Original data members
    Eigen::VectorXd phi;
    std::vector<int> narrowBand;
    std::unordered_map<std::string, MaterialProperties> materialProperties;
    double GRID_SPACING;
    int STEPS;
    int REINIT_INTERVAL;
    int NARROW_BAND_UPDATE_INTERVAL;
    double CURVATURE_WEIGHT;
    SpatialScheme* spatialScheme;
    TimeScheme* timeScheme;
    
    // Cached data for efficient computation
    struct CachedPointData {
        Eigen::Vector3d etching_rates;
        bool isBoundary;
    };
    std::vector<CachedPointData> cachedData;
    
    // Trapezoidal decomposition parameters
    struct TrapezoidParams {
        int t0, t1;            // Time range
        int startIdx, endIdx;  // Index range in narrowBand
        int dx0, dx1;          // Slope parameters (not used in this context but kept for consistency)
    };
    
    // Precompute material properties and boundary status
    void precomputePointData() {
        cachedData.resize(narrowBand.size());
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t k = 0; k < narrowBand.size(); ++k) {
            const int idx = narrowBand[k];
            std::string material = getMaterialAtPoint(idx);
            
            Eigen::Vector3d etching_rates;
            const auto it = materialProperties.find(material);
            if (it != materialProperties.end()) { 
                const auto& props = it->second;  
                const double lateral_etch = props.lateralRatio * props.etchRatio; 
                etching_rates << -lateral_etch, -lateral_etch, -props.etchRatio; 
            } else {
                etching_rates.setZero();  
            }
            cachedData[k].etching_rates = etching_rates;
            cachedData[k].isBoundary = isOnBoundary(narrowBand[k]);
        }
    }
    
    // Helper function to compute derivatives and level set updates for a specific point
    void computePointUpdate(int idx, int k, const Eigen::VectorXd& phi_current, Eigen::VectorXd& result) {
        // Skip boundary points to avoid instability
        if (cachedData[k].isBoundary) {
            return;
        }
        
        // Use pre-computed material properties
        const Eigen::Vector3d& modifiedU = cachedData[k].etching_rates;
        
        // Calculate spatial derivatives
        DerivativeOperator Dop;
        spatialScheme->SpatialSch(idx, phi_current, GRID_SPACING, Dop);
        
        // Calculate normal vector - only if needed for curvature
        Eigen::Vector3d normal;
        const bool use_curvature = CURVATURE_WEIGHT > 0.0;
        if (use_curvature) {
            normal = Eigen::Vector3d(
                (Dop.dxP + Dop.dxN) * 0.5,
                (Dop.dyP + Dop.dyN) * 0.5,
                (Dop.dzP + Dop.dzN) * 0.5
            );
        }
        
        // Compute advection terms more efficiently
        const double modU_x = modifiedU.x();
        const double modU_y = modifiedU.y();
        const double modU_z = modifiedU.z();
        
        // Compute advection terms directly
        const double advectionN = std::max(modU_x, 0.0) * Dop.dxN + 
                               std::max(modU_y, 0.0) * Dop.dyN + 
                               std::max(modU_z, 0.0) * Dop.dzN;
        const double advectionP = std::min(modU_x, 0.0) * Dop.dxP + 
                               std::min(modU_y, 0.0) * Dop.dyP + 
                               std::min(modU_z, 0.0) * Dop.dzP;
        
        // Compute gradient magnitudes
        const double epsilon = 1e-10;
        const double dxN_sq = Dop.dxN * Dop.dxN;
        const double dyN_sq = Dop.dyN * Dop.dyN;
        const double dzN_sq = Dop.dzN * Dop.dzN;
        const double dxP_sq = Dop.dxP * Dop.dxP;
        const double dyP_sq = Dop.dyP * Dop.dyP;
        const double dzP_sq = Dop.dzP * Dop.dzP;
        
        const double NP = std::sqrt(dxN_sq + dyN_sq + dzN_sq + epsilon);
        const double PP = std::sqrt(dxP_sq + dyP_sq + dzP_sq + epsilon);
       
        // Compute curvature term only if needed
        double curvatureterm = 0.0;
        if (use_curvature) {
            const double curvature = CURVATURE_WEIGHT * computeMeanCurvature(idx, phi_current);
            curvatureterm = std::max(curvature, 0.0) * NP + std::min(curvature, 0.0) * PP; 
        }
        
        // Compute final result
        result[idx] = -(advectionN + advectionP) + curvatureterm;
    }
    
    // Base case for the trapezoid recursion
    // Process a single time step for a range of narrow band points
    void processTrapezoidBase(int t, int startIdx, int endIdx, 
                             const Eigen::VectorXd& phi_current, 
                             Eigen::VectorXd& phi_next) {
        #pragma omp parallel for schedule(dynamic, 1024)
        for (int k = startIdx; k < endIdx; ++k) {
            const int idx = narrowBand[k];
            // Skip boundary points
            if (cachedData[k].isBoundary) {
                continue;
            }
            
            // Calculate derivatives and level set updates for this point
            Eigen::VectorXd result = Eigen::VectorXd::Zero(phi_current.size());
            computePointUpdate(idx, k, phi_current, result);
            
            // Apply time integration
            phi_next[idx] = phi_current[idx] + result[idx];
        }
    }
    
    // Recursive trapezoid decomposition
    void processTrapezoid(const TrapezoidParams& params, std::vector<Eigen::VectorXd>& phi_steps) {
        const int time_range = params.t1 - params.t0;
        const int space_range = params.endIdx - params.startIdx;
        
        // Base case: single time step
        if (time_range == 1) {
            processTrapezoidBase(params.t0, params.startIdx, params.endIdx, 
                                phi_steps[params.t0], phi_steps[params.t1]);
            return;
        }
        
        // Decide whether to cut in space or time
        // If width ≥ 2*height, do a space cut; otherwise, do a time cut
        if (space_range >= 2 * time_range) {
            // Space cut: divide the spatial domain
            int midIdx = params.startIdx + space_range / 2;
            
            // Process left trapezoid
            TrapezoidParams leftParams = params;
            leftParams.endIdx = midIdx;
            processTrapezoid(leftParams, phi_steps);
            
            // Process right trapezoid
            TrapezoidParams rightParams = params;
            rightParams.startIdx = midIdx;
            processTrapezoid(rightParams, phi_steps);
        } else {
            // Time cut: divide the temporal domain
            int midTime = params.t0 + time_range / 2;
            
            // Process bottom trapezoid
            TrapezoidParams bottomParams = params;
            bottomParams.t1 = midTime;
            processTrapezoid(bottomParams, phi_steps);
            
            // Process top trapezoid
            TrapezoidParams topParams = params;
            topParams.t0 = midTime;
            processTrapezoid(topParams, phi_steps);
        }
    }
    
    // For more efficient parallel execution
    void processParallelTrapezoid(const TrapezoidParams& params, std::vector<Eigen::VectorXd>& phi_steps) {
        const int time_range = params.t1 - params.t0;
        const int space_range = params.endIdx - params.startIdx;
        
        // Base case: single time step
        if (time_range == 1) {
            processTrapezoidBase(params.t0, params.startIdx, params.endIdx, 
                                phi_steps[params.t0], phi_steps[params.t1]);
            return;
        }
        
        // Decide whether to cut in space or time
        if (space_range >= 2 * time_range) {
            // Space cut: divide the spatial domain
            int midIdx = params.startIdx + space_range / 2;
            
            // Process left and right trapezoids in parallel
            TrapezoidParams leftParams = params;
            leftParams.endIdx = midIdx;
            
            TrapezoidParams rightParams = params;
            rightParams.startIdx = midIdx;
            
            #pragma omp task
            processTrapezoid(leftParams, phi_steps);
            
            #pragma omp task
            processTrapezoid(rightParams, phi_steps);
            
            #pragma omp taskwait
        } else {
            // Time cut: divide the temporal domain
            int midTime = params.t0 + time_range / 2;
            
            // Bottom trapezoid must be processed first
            TrapezoidParams bottomParams = params;
            bottomParams.t1 = midTime;
            processTrapezoid(bottomParams, phi_steps);
            
            // Then process top trapezoid
            TrapezoidParams topParams = params;
            topParams.t0 = midTime;
            processTrapezoid(topParams, phi_steps);
        }
    }
    
public:
    // Constructor and other necessary methods
    // ...
    
    // Main evolution method using cache-oblivious algorithm
    bool evolve() {
        try {
            phi = initializeSignedDistanceField();
            updateNarrowBand();
            
            // Pre-compute material properties and boundary status
            precomputePointData();
            
            // Allocate space for all time steps
            std::vector<Eigen::VectorXd> phi_steps(STEPS + 1);
            phi_steps[0] = phi;
            
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
            
            // Process the main trapezoid
            #pragma omp parallel
            {
                #pragma omp single
                processParallelTrapezoid(mainTrapezoid, phi_steps);
            }
            
            // Final phi is the last time step
            phi = phi_steps[STEPS];
            
            // Perform reinitialization
            reinitialize();
            
            std::cout << "Cache-oblivious evolution completed successfully." << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error during cache-oblivious evolution: " << e.what() << std::endl;
            return false;
        }
    }
    
};