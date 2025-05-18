#include "LevelSetMethod.hpp"
#include <stdexcept>    // For std::out_of_range
#include <iostream>     // For std::cerr (error reporting)
#include <execution>    // For parallel algorithms
#include <algorithm>    // For std::sort with execution policy

CGAL::Bbox_3 LevelSetMethod::calculateBoundingBox() const {
    if (mesh.is_empty()) {
        throw std::runtime_error("Mesh is empty - cannot calculate bounding box");
    }
    return CGAL::Polygon_mesh_processing::bbox(mesh);
}

void LevelSetMethod::loadMesh(const std::string& filename) {
    if (!PMP::IO::read_polygon_mesh(filename, mesh) || is_empty(mesh) || !CGAL::is_closed(mesh) || !is_triangle_mesh(mesh)) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
        
    tree = std::make_unique<AABB_tree>(faces(mesh).first, faces(mesh).second, mesh);
    tree->accelerate_distance_queries(); 
}

double LevelSetMethod::computeMeanCurvature(int idx, const Eigen::VectorXd& phi) {
    // Fast boundary check
    if (isOnBoundary(idx)) {
        return 0.0; // Return zero curvature at boundaries for stability
    }
    
    // Extract coordinates once
    static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
    const int x = idx % GRID_SIZE;
    const int y = (idx / GRID_SIZE) % GRID_SIZE;
    const int z = idx / GRID_SIZE_SQ;
    
    // Pre-compute all indices at once for better cache locality
    // Basic neighbor indices
    const int idx_x_plus = idx + 1; // Optimized from getIndex(x+1, y, z)
    const int idx_x_minus = idx - 1; // Optimized from getIndex(x-1, y, z)
    const int idx_y_plus = idx + GRID_SIZE; // Optimized from getIndex(x, y+1, z)
    const int idx_y_minus = idx - GRID_SIZE; // Optimized from getIndex(x, y-1, z)
    const int idx_z_plus = idx + GRID_SIZE_SQ; // Optimized from getIndex(x, y, z+1)
    const int idx_z_minus = idx - GRID_SIZE_SQ; // Optimized from getIndex(x, y, z-1)
    
    // Mixed derivatives indices - optimized direct calculation
    const int idx_xy_plus = idx + 1 + GRID_SIZE; // Optimized from getIndex(x+1, y+1, z)
    const int idx_xy_minus = idx - 1 - GRID_SIZE; // Optimized from getIndex(x-1, y-1, z)
    const int idx_xz_plus = idx + 1 + GRID_SIZE_SQ; // Optimized from getIndex(x+1, y, z+1)
    const int idx_xz_minus = idx - 1 - GRID_SIZE_SQ; // Optimized from getIndex(x-1, y, z-1)
    const int idx_yz_plus = idx + GRID_SIZE + GRID_SIZE_SQ; // Optimized from getIndex(x, y+1, z+1)
    const int idx_yz_minus = idx - GRID_SIZE - GRID_SIZE_SQ; // Optimized from getIndex(x, y-1, z-1)
    
    // Pre-compute phi values for better cache locality
    const double phi_center = phi[idx];
    const double phi_x_plus = phi[idx_x_plus];
    const double phi_x_minus = phi[idx_x_minus];
    const double phi_y_plus = phi[idx_y_plus];
    const double phi_y_minus = phi[idx_y_minus];
    const double phi_z_plus = phi[idx_z_plus];
    const double phi_z_minus = phi[idx_z_minus];
    const double phi_xy_plus = phi[idx_xy_plus];
    const double phi_xy_minus = phi[idx_xy_minus];
    const double phi_xz_plus = phi[idx_xz_plus];
    const double phi_xz_minus = phi[idx_xz_minus];
    const double phi_yz_plus = phi[idx_yz_plus];
    const double phi_yz_minus = phi[idx_yz_minus];
    
    // Pre-compute common constants
    static const double inv_spacing = 1.0 / (2.0 * GRID_SPACING);
    static const double inv_spacing_squared = 1.0 / (GRID_SPACING * GRID_SPACING);
    static const double quarter_inv_spacing_squared = 0.25 * inv_spacing_squared;
    static const double epsilon = 1e-10;
    static const double min_gradient = 1e-6;
    static const double max_curvature = 1.0 / GRID_SPACING;
    
    // First derivatives (central differences) - more efficient calculation
    const double phi_x = (phi_x_plus - phi_x_minus) * inv_spacing;
    const double phi_y = (phi_y_plus - phi_y_minus) * inv_spacing;
    const double phi_z = (phi_z_plus - phi_z_minus) * inv_spacing;
    
    // Calculate gradient magnitude with small epsilon to avoid division by zero
    const double phi_x_squared = phi_x * phi_x;
    const double phi_y_squared = phi_y * phi_y;
    const double phi_z_squared = phi_z * phi_z;
    const double grad_phi_squared = phi_x_squared + phi_y_squared + phi_z_squared + epsilon;
    const double grad_phi_magnitude = std::sqrt(grad_phi_squared);
    
    // Early return if gradient is too small (curvature not well-defined)
    if (grad_phi_magnitude < min_gradient) {
        return 0.0;
    }
    
    // Second derivatives (central differences) - more efficient calculation
    const double phi_xx = (phi_x_plus - 2.0 * phi_center + phi_x_minus) * inv_spacing_squared;
    const double phi_yy = (phi_y_plus - 2.0 * phi_center + phi_y_minus) * inv_spacing_squared;
    const double phi_zz = (phi_z_plus - 2.0 * phi_center + phi_z_minus) * inv_spacing_squared;
    
    // Mixed derivatives (central differences) with more stable and efficient calculation
    const double phi_xy = (phi_xy_plus - phi_x_plus - phi_y_plus + phi_center +
                          phi_center - phi_x_minus - phi_y_minus + phi_xy_minus) * 
                          quarter_inv_spacing_squared;
    
    const double phi_xz = (phi_xz_plus - phi_x_plus - phi_z_plus + phi_center +
                          phi_center - phi_x_minus - phi_z_minus + phi_xz_minus) * 
                          quarter_inv_spacing_squared;
    
    const double phi_yz = (phi_yz_plus - phi_y_plus - phi_z_plus + phi_center +
                          phi_center - phi_y_minus - phi_z_minus + phi_yz_minus) * 
                          quarter_inv_spacing_squared;
    
    // Pre-compute common terms for mean curvature formula
    const double phi_y_z_squared_sum = phi_y_squared + phi_z_squared;
    const double phi_x_z_squared_sum = phi_x_squared + phi_z_squared;
    const double phi_x_y_squared_sum = phi_x_squared + phi_y_squared;
    
    // Compute mean curvature using the formula with pre-computed terms
    const double numerator = phi_xx * phi_y_z_squared_sum +
                            phi_yy * phi_x_z_squared_sum +
                            phi_zz * phi_x_y_squared_sum -
                            2.0 * (phi_xy * phi_x * phi_y +
                                  phi_xz * phi_x * phi_z +
                                  phi_yz * phi_y * phi_z);
    
    // Use grad_phi_magnitude^3 with epsilon to avoid division by very small numbers
    const double grad_phi_cubed = grad_phi_magnitude * grad_phi_squared;
    
    // Calculate curvature and apply limiter to prevent numerical instability
    const double curvature = numerator / grad_phi_cubed;
    return std::max(std::min(curvature, max_curvature), -max_curvature);
}

bool LevelSetMethod::evolve() {
    try {
        phi = initializeSignedDistanceField();
        updateNarrowBand();
        
        Eigen::VectorXd newPhi = phi;
        
        // Progress tracking
        const int progressInterval = 10;
        const double inv_grid_spacing = 1.0 / GRID_SPACING;
        
        // Pre-compute material properties for all grid points in narrow band
        // This avoids repeated string lookups during evolution
        std::vector<Eigen::Vector3d> materialEtchRates(narrowBand.size());
        
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
            materialEtchRates[k] = etching_rates;
        }
        
        // Pre-compute boundary status for all narrow band points
        std::vector<bool> isBoundary(narrowBand.size());
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t k = 0; k < narrowBand.size(); ++k) {
            isBoundary[k] = isOnBoundary(narrowBand[k]);
        }
        
        // Cache frequently used constants
        const double epsilon = 1e-10;
        const bool use_curvature = CURVATURE_WEIGHT > 0.0;
        
        // Main evolution loop
        for (int step = 0; step < STEPS; ++step) {
            // Report progress periodically
            if (step % progressInterval == 0) {
                std::cout << "Evolution step " << step << "/" << STEPS << std::endl;
            }
            
            // Define level set operator with optimized memory access
            auto levelSetOperator = [this, &materialEtchRates, &isBoundary, epsilon, use_curvature](const Eigen::VectorXd& phi_current) -> Eigen::VectorXd {
                // Pre-allocate result vector with zeros
                Eigen::VectorXd result = Eigen::VectorXd::Zero(phi_current.size());
                
                // Process narrow band points in parallel with larger chunks for better cache efficiency
                #pragma omp parallel for schedule(dynamic, 1024)
                for (size_t k = 0; k < narrowBand.size(); ++k) {
                    const int idx = narrowBand[k];
                    
                    // Skip boundary points to avoid instability - use pre-computed boundary status
                    if (isBoundary[k]) {
                        continue;
                    }
                    
                    // Use pre-computed material properties
                    const Eigen::Vector3d& modifiedU = materialEtchRates[k];
                    
                    // Calculate spatial derivatives
                    DerivativeOperator Dop;
                    spatialScheme->SpatialSch(idx, phi_current, GRID_SPACING, Dop);
                    
                    // Calculate normal vector - only if needed for curvature
                    Eigen::Vector3d normal;
                    if (use_curvature) {
                        normal = Eigen::Vector3d(
                            (Dop.dxP + Dop.dxN) * 0.5, // Multiply by 0.5 instead of divide by 2.0
                            (Dop.dyP + Dop.dyN) * 0.5,
                            (Dop.dzP + Dop.dzN) * 0.5
                        );
                    }
                    
                    // Compute advection terms more efficiently
                    // Use branchless programming with std::max/min
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
                return result;
            };
            
            // Apply the time integration scheme
            Eigen::VectorXd phi_updated = timeScheme->advance(phi, levelSetOperator);
            
            // Update only the narrow band points with larger chunks for better cache efficiency
            #pragma omp parallel for schedule(dynamic, 1024)
            for (size_t k = 0; k < narrowBand.size(); ++k) {
                const int idx = narrowBand[k];
                newPhi[idx] = phi_updated[idx];
            }
            
            // Use swap for efficient memory management
            phi.swap(newPhi);
            
            // Perform reinitialization and narrow band update periodically
            if (step % REINIT_INTERVAL == 0 && step > 0) {
                reinitialize();
            }

            if (step % NARROW_BAND_UPDATE_INTERVAL == 0 && step > 0) {
                updateNarrowBand();
                
                // Resize and recompute cached data after narrow band update
                materialEtchRates.resize(narrowBand.size());
                isBoundary.resize(narrowBand.size());
                
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
                    materialEtchRates[k] = etching_rates;
                    isBoundary[k] = isOnBoundary(narrowBand[k]);
                }
            }
        }
        
        std::cout << "Evolution completed successfully." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during evolution: " << e.what() << std::endl;
        return false;
    }
}

void LevelSetMethod::reinitialize() {
    // Create a temporary copy of phi
    Eigen::VectorXd tempPhi = phi;
    
    // Number of iterations for reinitialization
    const int REINIT_STEPS = 10; // Increased for better convergence
    const double dtau = 0.5 * GRID_SPACING; // CFL condition for stability
    const double epsilon = 1e-6; // Small value to avoid division by zero
    const double inv_grid_spacing = 1.0 / GRID_SPACING;
    const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
    
    // Pre-compute indices for common neighbor patterns to improve cache locality
    std::vector<std::array<int, 6>> neighbor_indices(narrowBand.size());
    
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t k = 0; k < narrowBand.size(); ++k) {
        const int idx = narrowBand[k];
        const int x = idx % GRID_SIZE;
        const int y = (idx / GRID_SIZE) % GRID_SIZE;
        const int z = idx / GRID_SIZE_SQ;
        
        // Store neighbor indices for faster access
        neighbor_indices[k][0] = getIndex(x-1, y, z); // x-
        neighbor_indices[k][1] = getIndex(x+1, y, z); // x+
        neighbor_indices[k][2] = getIndex(x, y-1, z); // y-
        neighbor_indices[k][3] = getIndex(x, y+1, z); // y+
        neighbor_indices[k][4] = getIndex(x, y, z-1); // z-
        neighbor_indices[k][5] = getIndex(x, y, z+1); // z+
    }
    
    // Perform reinitialization iterations
    for (int step = 0; step < REINIT_STEPS; ++step) {
        // Only reinitialize points in the narrow band - parallelize this loop
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t k = 0; k < narrowBand.size(); ++k) {
            const int idx = narrowBand[k];
            
            // Get pre-computed neighbor indices
            const int idx_x_minus = neighbor_indices[k][0];
            const int idx_x_plus = neighbor_indices[k][1];
            const int idx_y_minus = neighbor_indices[k][2];
            const int idx_y_plus = neighbor_indices[k][3];
            const int idx_z_minus = neighbor_indices[k][4];
            const int idx_z_plus = neighbor_indices[k][5];
            
            // Compute sign function once at the beginning
            // Use a smooth sign function for better numerical stability
            const double phi0 = phi[idx]; // Original phi value
            const double sign = phi0 / std::sqrt(phi0*phi0 + GRID_SPACING*GRID_SPACING);
            
            // Pre-compute differences for all directions
            const double phi_center = tempPhi[idx];
            const double phi_x_minus = tempPhi[idx_x_minus];
            const double phi_x_plus = tempPhi[idx_x_plus];
            const double phi_y_minus = tempPhi[idx_y_minus];
            const double phi_y_plus = tempPhi[idx_y_plus];
            const double phi_z_minus = tempPhi[idx_z_minus];
            const double phi_z_plus = tempPhi[idx_z_plus];
            
            // Compute all derivatives at once
            const double dx_minus = (phi_center - phi_x_minus) * inv_grid_spacing;
            const double dx_plus = (phi_x_plus - phi_center) * inv_grid_spacing;
            const double dy_minus = (phi_center - phi_y_minus) * inv_grid_spacing;
            const double dy_plus = (phi_y_plus - phi_center) * inv_grid_spacing;
            const double dz_minus = (phi_center - phi_z_minus) * inv_grid_spacing;
            const double dz_plus = (phi_z_plus - phi_center) * inv_grid_spacing;
            
            // Use upwind scheme for gradient calculation based on sign
            double dx, dy, dz;
            
            // X direction upwind - branchless version using conditional math
            const double dx_minus_term = std::max(0.0, dx_minus) * std::max(0.0, dx_minus);
            const double dx_plus_term = std::min(0.0, dx_plus) * std::min(0.0, dx_plus);
            const double dx_minus_term_neg = std::min(0.0, dx_minus) * std::min(0.0, dx_minus);
            const double dx_plus_term_neg = std::max(0.0, dx_plus) * std::max(0.0, dx_plus);
            
            // Select terms based on sign
            dx = (sign > 0.0) ? (dx_minus_term + dx_plus_term) : (dx_minus_term_neg + dx_plus_term_neg);
            
            // Y direction upwind - same branchless approach
            const double dy_minus_term = std::max(0.0, dy_minus) * std::max(0.0, dy_minus);
            const double dy_plus_term = std::min(0.0, dy_plus) * std::min(0.0, dy_plus);
            const double dy_minus_term_neg = std::min(0.0, dy_minus) * std::min(0.0, dy_minus);
            const double dy_plus_term_neg = std::max(0.0, dy_plus) * std::max(0.0, dy_plus);
            
            dy = (sign > 0.0) ? (dy_minus_term + dy_plus_term) : (dy_minus_term_neg + dy_plus_term_neg);
            
            // Z direction upwind - same branchless approach
            const double dz_minus_term = std::max(0.0, dz_minus) * std::max(0.0, dz_minus);
            const double dz_plus_term = std::min(0.0, dz_plus) * std::min(0.0, dz_plus);
            const double dz_minus_term_neg = std::min(0.0, dz_minus) * std::min(0.0, dz_minus);
            const double dz_plus_term_neg = std::max(0.0, dz_plus) * std::max(0.0, dz_plus);
            
            dz = (sign > 0.0) ? (dz_minus_term + dz_plus_term) : (dz_minus_term_neg + dz_plus_term_neg);
            
            // Calculate gradient magnitude with proper upwinding
            const double gradMag = std::sqrt(dx + dy + dz + epsilon);
            
            // Update equation for reinitialization with TVD Runge-Kutta
            tempPhi[idx] = phi_center - dtau * sign * (gradMag - 1.0);
        }
    }
    
    // Update phi with reinitialized values using swap for efficiency
    phi.swap(tempPhi);
}


void LevelSetMethod::updateNarrowBand() {
    // Reserve memory to avoid reallocations
    narrowBand.clear();
    const size_t estimated_size = grid.size();
    narrowBand.reserve(estimated_size / 4); // More realistic estimate
    
    // Calculate narrow band width in grid units once
    const double narrow_band_grid_units = NARROW_BAND_WIDTH * GRID_SPACING;
    
    // Use thread-local storage with block processing for better cache locality
    const size_t block_size = 8192; // Larger blocks for better cache efficiency
    
    // Create thread-local vectors first, then merge at the end
    const int num_threads = omp_get_max_threads();
    std::vector<std::vector<int>> thread_local_bands(num_threads);
    
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        auto& localBand = thread_local_bands[thread_id];
        localBand.reserve(estimated_size / num_threads / 4); // More realistic estimate
        
        // Process grid in blocks for better cache efficiency
        #pragma omp for schedule(dynamic, block_size) nowait
        for (size_t i = 0; i < grid.size(); ++i) {
            // Use branchless programming where possible
            const bool is_in_band = !isOnBoundary(i) && std::abs(phi[i]) <= narrow_band_grid_units;
            if (is_in_band) {
                localBand.push_back(i);
            }
        }
    }
    
    // Merge thread-local vectors without locking
    // First calculate total size needed
    size_t total_size = 0;
    for (const auto& local_band : thread_local_bands) {
        total_size += local_band.size();
    }
    
    // Pre-allocate memory
    narrowBand.reserve(total_size);
    
    // Merge all thread-local vectors
    for (auto& local_band : thread_local_bands) {
        narrowBand.insert(narrowBand.end(), local_band.begin(), local_band.end());
        // Clear the thread-local vector to free memory
        std::vector<int>().swap(local_band);
    }
    

    std::sort(narrowBand.begin(), narrowBand.end());

    
    std::cout << "Narrow band updated. Size: " << narrowBand.size() 
              << " (" << (narrowBand.size() * 100.0 / grid.size()) << "% of grid)" << std::endl;
}

void LevelSetMethod::generateGrid() {
    if (mesh.is_empty()) {
        throw std::runtime_error("Mesh not loaded - cannot generate grid");
    }
    
    CGAL::Bbox_3 bbox = calculateBoundingBox();
    // Add 10% padding around the mesh
    double padding = 0.1 * std::max({bbox.xmax()-bbox.xmin(), 
                                   bbox.ymax()-bbox.ymin(), 
                                   bbox.zmax()-bbox.zmin()});
    
    double xmin = bbox.xmin() - padding;
    double xmax = bbox.xmax() + padding;
    double ymin = bbox.ymin() - padding;
    double ymax = bbox.ymax() + padding;
    double zmin = bbox.zmin() - padding;
    double zmax = bbox.zmax() + padding;
    
    // Calculate grid spacing based on largest dimension
    double max_dim = std::max({xmax-xmin, ymax-ymin, zmax-zmin});
    BOX_SIZE = max_dim;
    GRID_SPACING = max_dim / (GRID_SIZE - 1);
    
    grid.clear();
    grid.reserve(GRID_SIZE * GRID_SIZE * GRID_SIZE);
    
    for (int z = 0; z < GRID_SIZE; ++z) {
        double pz = zmin + z * GRID_SPACING;
        for (int y = 0; y < GRID_SIZE; ++y) {
            double py = ymin + y * GRID_SPACING;
            for (int x = 0; x < GRID_SIZE; ++x) {
                double px = xmin + x * GRID_SPACING;
                grid.emplace_back(px, py, pz);
            }
        }
    }
}


Eigen::VectorXd LevelSetMethod::initializeSignedDistanceField() {
    if (!tree) {
        throw std::runtime_error("AABB tree not initialized. Load a mesh first.");
    }
    
    std::cout << "Initializing signed distance field..." << std::endl;
    
    // Pre-allocate memory for the signed distance field
    const size_t grid_size = grid.size();
    Eigen::VectorXd sdf(grid_size);
    
    // Create inside/outside classifier once (thread-safe in CGAL)
    CGAL::Side_of_triangle_mesh<Mesh, Kernel> inside(mesh);
    
    // Process grid points in blocks for better cache locality
    // Larger blocks for better cache efficiency and fewer thread synchronizations
    const size_t block_size = 8192;
    
    // Get number of threads for better load balancing
    const int num_threads = omp_get_max_threads();
    
    // Track progress with atomic counter
    std::atomic<size_t> progress_counter(0);
    const size_t progress_interval = grid_size / 20;
    
    // Use thread-local storage for AABB tree queries to reduce contention
    #pragma omp parallel
    {
        // Thread-local variables for better performance
        const int thread_id = omp_get_thread_num();
        size_t local_counter = 0;
        
        // Process grid in blocks for better cache efficiency
        #pragma omp for schedule(dynamic, block_size) nowait
        for (size_t i = 0; i < grid_size; ++i) {
            // Cache grid point to reduce memory access
            const Point_3& point = grid[i];
            
            // Compute squared distance to the mesh using AABB tree
            // Use direct access to avoid function call overhead
            auto closest = tree->closest_point_and_primitive(point);
            double sq_dist = CGAL::sqrt(CGAL::squared_distance(point, closest.first));
            
            // Determine if point is inside or outside the mesh
            // Use branchless programming for better performance
            CGAL::Bounded_side res = inside(point);
            
            // Set signed distance using branchless programming
            // Avoid branching with conditional operator
            double sign = (res == CGAL::ON_BOUNDED_SIDE) ? -1.0 : 
                         (res == CGAL::ON_BOUNDARY) ? 0.0 : 1.0;
            
            sdf[i] = sign * sq_dist;
            
            // Update local progress counter
            local_counter++;
            
            // Periodically update global progress counter to reduce atomic operations
            if (local_counter % (block_size / 4) == 0) {
                size_t global_progress = progress_counter.fetch_add(local_counter, std::memory_order_relaxed);
                if (thread_id == 0 && (global_progress / progress_interval) < ((global_progress + local_counter) / progress_interval)) {
                    std::cout << "SDF initialization progress: " << (global_progress + local_counter) * 100 / grid_size << "%\r" << std::flush;
                }
                local_counter = 0;
            }
        }
        
        // Add remaining local counter to global counter
        if (local_counter > 0) {
            progress_counter.fetch_add(local_counter, std::memory_order_relaxed);
        }
    }
    
    std::cout << "\nSigned distance field initialization complete." << std::endl;
    return sdf;
}

inline bool LevelSetMethod::isOnBoundary(int idx) const {
    // Fast boundary check using grid coordinates
    // Use static constant for better compiler optimization
    static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
    static const int BOUNDARY_THICKNESS = 3; // Thickness of boundary layer
    static const int BOUNDARY_INNER = BOUNDARY_THICKNESS;
    static const int BOUNDARY_OUTER = GRID_SIZE - BOUNDARY_THICKNESS;
    
    // Extract coordinates with optimized modulo operations
    // For powers of 2 grid sizes, these could be optimized further with bit operations
    const int x = idx % GRID_SIZE;
    const int y = (idx / GRID_SIZE) % GRID_SIZE;
    const int z = idx / GRID_SIZE_SQ;
    
    // Use branchless programming with bitwise OR for boundary check
    // A point is on boundary if any coordinate is within boundary thickness
    return ((x < BOUNDARY_INNER) | (x >= BOUNDARY_OUTER) | 
            (y < BOUNDARY_INNER) | (y >= BOUNDARY_OUTER) | 
            (z < BOUNDARY_INNER) | (z >= BOUNDARY_OUTER));
}

inline int LevelSetMethod::getIndex(int x, int y, int z) const {
    // Use static constant for better compiler optimization
    static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
    
    // Bounds checking in debug mode only
#ifdef DEBUG
    if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE || z < 0 || z >= GRID_SIZE) {
        return 0; // Return safe index for out-of-bounds access
    }
#endif
    
    // Fast index calculation with multiplication
    return x + y * GRID_SIZE + z * GRID_SIZE_SQ;
}

bool LevelSetMethod::extractSurfaceMeshCGAL(const std::string& filename) {
    try {
        if (phi.size() != grid.size()) {
            throw std::runtime_error("Level set function not initialized.");
        }
 
        typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
        typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
        typedef Tr::Geom_traits GT;
        typedef GT::Sphere_3 Sphere_3;
        typedef GT::FT FT;
        typedef std::function<FT(typename GT::Point_3)> Function;
        typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;
    
        // Define the implicit function for the zero level set
        class LevelSetImplicitFunction {
        private:
            const std::vector<Point_3>& grid;
            const Eigen::VectorXd& phi;
            const int GRID_SIZE;
            const double GRID_SPACING;
            const double gridOriginX, gridOriginY, gridOriginZ;
            
        public:
            LevelSetImplicitFunction(const std::vector<Point_3>& grid, const Eigen::VectorXd& phi, 
                                    int gridSize, double gridSpacing)
                : grid(grid), phi(phi), GRID_SIZE(gridSize), GRID_SPACING(gridSpacing),
                  gridOriginX(grid[0].x()), gridOriginY(grid[0].y()), gridOriginZ(grid[0].z()) {
            }
                
            FT operator()(const Point_3& p) const {
                // Fast grid-based lookup instead of linear search
                // Calculate grid indices based on point position
                int x = std::round((p.x() - gridOriginX) / GRID_SPACING);
                int y = std::round((p.y() - gridOriginY) / GRID_SPACING);
                int z = std::round((p.z() - gridOriginZ) / GRID_SPACING);
                
                // Clamp to grid boundaries
                x = std::max(0, std::min(x, GRID_SIZE - 1));
                y = std::max(0, std::min(y, GRID_SIZE - 1));
                z = std::max(0, std::min(z, GRID_SIZE - 1));
                
                // Calculate grid index
                int idx = x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                
                // Bounds check
                if (idx >= 0 && idx < static_cast<int>(phi.size())) {
                    return phi[idx];
                }
                
                // Fallback to trilinear interpolation for points outside the grid
                // This provides smoother results than nearest neighbor
                // Find the cell containing the point
                x = std::floor((p.x() - gridOriginX) / GRID_SPACING);
                y = std::floor((p.y() - gridOriginY) / GRID_SPACING);
                z = std::floor((p.z() - gridOriginZ) / GRID_SPACING);
                
                // Clamp to valid range for interpolation
                x = std::max(0, std::min(x, GRID_SIZE - 2));
                y = std::max(0, std::min(y, GRID_SIZE - 2));
                z = std::max(0, std::min(z, GRID_SIZE - 2));
                
                // Calculate fractional position within cell
                double fx = (p.x() - (gridOriginX + x * GRID_SPACING)) / GRID_SPACING;
                double fy = (p.y() - (gridOriginY + y * GRID_SPACING)) / GRID_SPACING;
                double fz = (p.z() - (gridOriginZ + z * GRID_SPACING)) / GRID_SPACING;
                
                // Get the eight corners of the cell
                int idx000 = x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                int idx001 = x + y * GRID_SIZE + (z+1) * GRID_SIZE * GRID_SIZE;
                int idx010 = x + (y+1) * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                int idx011 = x + (y+1) * GRID_SIZE + (z+1) * GRID_SIZE * GRID_SIZE;
                int idx100 = (x+1) + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                int idx101 = (x+1) + y * GRID_SIZE + (z+1) * GRID_SIZE * GRID_SIZE;
                int idx110 = (x+1) + (y+1) * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                int idx111 = (x+1) + (y+1) * GRID_SIZE + (z+1) * GRID_SIZE * GRID_SIZE;
                
                // Perform trilinear interpolation
                double v000 = phi[idx000];
                double v001 = phi[idx001];
                double v010 = phi[idx010];
                double v011 = phi[idx011];
                double v100 = phi[idx100];
                double v101 = phi[idx101];
                double v110 = phi[idx110];
                double v111 = phi[idx111];
                
                // Interpolate along x
                double v00 = v000 * (1 - fx) + v100 * fx;
                double v01 = v001 * (1 - fx) + v101 * fx;
                double v10 = v010 * (1 - fx) + v110 * fx;
                double v11 = v011 * (1 - fx) + v111 * fx;
                
                // Interpolate along y
                double v0 = v00 * (1 - fy) + v10 * fy;
                double v1 = v01 * (1 - fy) + v11 * fy;
                
                // Interpolate along z
                return v0 * (1 - fz) + v1 * fz;
            }
        };

        // Create the implicit function with grid parameters
        LevelSetImplicitFunction implicitFunction(grid, phi, GRID_SIZE, GRID_SPACING);

        // Wrap the implicit function with the corrected type
        Function function = [&implicitFunction](const GT::Point_3& p) {
            return implicitFunction(Point_3(p.x(), p.y(), p.z()));
        };
        
        Tr tr;
        C2t3 c2t3(tr);
        
        // Adjust bounding sphere to better match your data
        double boundingSphereRadius = BOX_SIZE;
        Surface_3 surface(function, Sphere_3(CGAL::ORIGIN, boundingSphereRadius*boundingSphereRadius), 1e-5);
        
        // Adjust mesh criteria for better performance/quality tradeoff
        typedef CGAL::Surface_mesh_default_criteria_3<Tr> Criteria;
        Criteria criteria(30.0, GRID_SPACING * 2.0, GRID_SPACING * 2.0);
        
        // Define the mesh data structure
        typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
        Surface_mesh surface_mesh;
        
        std::cout << "Starting surface mesh generation..." << std::endl;
        // Generate the surface mesh
        CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());
        std::cout << "Surface mesh generation completed." << std::endl;
        
        // Convert the complex to a surface mesh
        CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, surface_mesh);
        
        
        // Save the surface mesh to a file
        if (!CGAL::IO::write_polygon_mesh(filename, surface_mesh, CGAL::parameters::stream_precision(17))) {
            throw std::runtime_error("Failed to write surface mesh to file.");
        }
        
        std::cout << "Surface mesh extracted and saved to " << filename << std::endl;
        std::cout << "Surface mesh has " << surface_mesh.number_of_vertices() << " vertices and " 
                  << surface_mesh.number_of_faces() << " faces." << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error extracting surface mesh: " << e.what() << std::endl;
        return false;
    }
}

std::string getMaterialForVertex(int face_idx, std::unordered_map<int, std::string> faceMaterials) {
    auto it = faceMaterials.find(face_idx);
    if (it != faceMaterials.end()) {
        return it->second;
    }
    return "unknown";
}

void LevelSetMethod::loadMaterialInfo(const std::string& csvFilename, const std::string& meshFilename) {
    std::cout << "Loading material information from CSV file: " << csvFilename << std::endl;
    
    // Use faster unordered_map for material lookups
    std::unordered_map<int, std::string> faceMaterials;
    
    // Read CSV file more efficiently
    std::ifstream csvFile(csvFilename);
    if (!csvFile.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + csvFilename);
    }
    
    // Skip header lines more efficiently
    std::string line;
    while (csvFile.peek() == '#') {
        std::getline(csvFile, line);
    }
    
    // Parse CSV content with less string manipulation
    while (std::getline(csvFile, line)) {
        size_t commaPos = line.find(',');
        if (commaPos != std::string::npos) {
            int faceIdx = std::stoi(line.substr(0, commaPos));
            std::string material = line.substr(commaPos + 1);
            faceMaterials[faceIdx] = material;
        }
    }
    
    std::cout << "Loaded " << faceMaterials.size() << " materials from CSV file." << std::endl;
    std::cout << "Loading mesh from file: " << meshFilename << std::endl;
    
    Mesh meshOrg;
    if (!PMP::IO::read_polygon_mesh(meshFilename, meshOrg) || is_empty(meshOrg) || !is_triangle_mesh(meshOrg)) {
        throw std::runtime_error("Failed to read mesh in LoadMaterialInfo");
    }
    
    // Create AABB tree for efficient spatial queries
    std::unique_ptr<AABB_tree> Ptree = std::make_unique<AABB_tree>(faces(meshOrg).first, faces(meshOrg).second, meshOrg);
    Ptree->accelerate_distance_queries();
    
    // Resize result vector once instead of pushing back
    const size_t gridSize = grid.size();
    gridMaterials.resize(gridSize);
    
    // Thread-local cache can improve performance
    const std::string defaultMaterial = "default";
    
    // Parallelize with better chunking for load balancing
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < gridSize; ++i) {
            const Point_3& point = grid[i];
            
            // Default material if no tree or no close face found
            std::string material = defaultMaterial;
            
            auto closest = Ptree->closest_point_and_primitive(point);
            int faceIdx = closest.second.id();
                
            auto it = faceMaterials.find(faceIdx);
            if (it != faceMaterials.end()) {
                material = it->second;
            } else {
                std::cerr << "Warning: Material not found for face index: " << faceIdx << std::endl;
            }
            static std::mutex mutex;
            {
                std::lock_guard<std::mutex> lock(mutex);
                gridMaterials[i] = material;
            }
        }
    }
}

std::string LevelSetMethod::getMaterialAtPoint(int idx) const {
    if (idx >= 0 && idx < static_cast<int>(gridMaterials.size())) {
        return gridMaterials[idx];
    }
    return "default";
}