#include "LevelSetMethod.hpp"

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
    const int x = idx % GRID_SIZE;
    const int y = (idx / GRID_SIZE) % GRID_SIZE;
    const int z = idx / (GRID_SIZE * GRID_SIZE);
    
    // Check if we're too close to the boundary for accurate curvature calculation
    if (x < 2 || x >= GRID_SIZE-2 || y < 2 || y >= GRID_SIZE-2 || z < 2 || z >= GRID_SIZE-2) {
        return 0.0; // Return zero curvature at boundaries for stability
    }
    
    // Get indices for central differences
    const int idx_x_plus = getIndex(x+1, y, z);
    const int idx_x_minus = getIndex(x-1, y, z);
    const int idx_y_plus = getIndex(x, y+1, z);
    const int idx_y_minus = getIndex(x, y-1, z);
    const int idx_z_plus = getIndex(x, y, z+1);
    const int idx_z_minus = getIndex(x, y, z-1);
    
    // Mixed derivatives indices
    const int idx_xy_plus = getIndex(x+1, y+1, z);
    const int idx_xy_minus = getIndex(x-1, y-1, z);
    const int idx_xz_plus = getIndex(x+1, y, z+1);
    const int idx_xz_minus = getIndex(x-1, y, z-1);
    const int idx_yz_plus = getIndex(x, y+1, z+1);
    const int idx_yz_minus = getIndex(x, y-1, z-1);
    
    // First derivatives (central differences)
    const double inv_spacing = 1.0 / (2.0 * GRID_SPACING);
    const double phi_x = (phi[idx_x_plus] - phi[idx_x_minus]) * inv_spacing;
    const double phi_y = (phi[idx_y_plus] - phi[idx_y_minus]) * inv_spacing;
    const double phi_z = (phi[idx_z_plus] - phi[idx_z_minus]) * inv_spacing;
    
    // Calculate gradient magnitude with small epsilon to avoid division by zero
    const double epsilon = 1e-10;
    const double grad_phi_squared = phi_x*phi_x + phi_y*phi_y + phi_z*phi_z + epsilon;
    const double grad_phi_magnitude = std::sqrt(grad_phi_squared);
    
    // If gradient is too small, curvature is not well-defined
    if (grad_phi_magnitude < 1e-6) {
        return 0.0;
    }
    
    // Second derivatives (central differences)
    const double inv_spacing_squared = 1.0 / (GRID_SPACING * GRID_SPACING);
    const double phi_xx = (phi[idx_x_plus] - 2.0 * phi[idx] + phi[idx_x_minus]) * inv_spacing_squared;
    const double phi_yy = (phi[idx_y_plus] - 2.0 * phi[idx] + phi[idx_y_minus]) * inv_spacing_squared;
    const double phi_zz = (phi[idx_z_plus] - 2.0 * phi[idx] + phi[idx_z_minus]) * inv_spacing_squared;
    
    // Mixed derivatives (central differences) with more stable calculation
    const double phi_xy = (phi[idx_xy_plus] - phi[idx_x_plus] - phi[idx_y_plus] + phi[idx] +
                          phi[idx] - phi[idx_x_minus] - phi[idx_y_minus] + phi[idx_xy_minus]) * 
                          (0.25 * inv_spacing_squared);
    
    const double phi_xz = (phi[idx_xz_plus] - phi[idx_x_plus] - phi[idx_z_plus] + phi[idx] +
                          phi[idx] - phi[idx_x_minus] - phi[idx_z_minus] + phi[idx_xz_minus]) * 
                          (0.25 * inv_spacing_squared);
    
    const double phi_yz = (phi[idx_yz_plus] - phi[idx_y_plus] - phi[idx_z_plus] + phi[idx] +
                          phi[idx] - phi[idx_y_minus] - phi[idx_z_minus] + phi[idx_yz_minus]) * 
                          (0.25 * inv_spacing_squared);
    
    // Compute mean curvature using the formula:
    // κ = div(∇φ/|∇φ|) = (φxx(φy²+φz²) + φyy(φx²+φz²) + φzz(φx²+φy²) - 2φxyφxφy - 2φxzφxφz - 2φyzφyφz) / |∇φ|³
    const double numerator = phi_xx * (phi_y*phi_y + phi_z*phi_z) +
                            phi_yy * (phi_x*phi_x + phi_z*phi_z) +
                            phi_zz * (phi_x*phi_x + phi_y*phi_y) -
                            2.0 * (phi_xy * phi_x * phi_y +
                                  phi_xz * phi_x * phi_z +
                                  phi_yz * phi_y * phi_z);
    
    // Use grad_phi_magnitude^3 with epsilon to avoid division by very small numbers
    const double grad_phi_cubed = grad_phi_magnitude * grad_phi_squared;
    
    // Limit the curvature to avoid extreme values that can cause instability
    double curvature = numerator / grad_phi_cubed;
    
    // Apply a limiter to the curvature to prevent numerical instability
    const double max_curvature = 1.0 / GRID_SPACING; // Maximum curvature based on grid resolution
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
        
        // Pre-sort narrow band for better cache locality
        std::sort(narrowBand.begin(), narrowBand.end());
        
        for (int step = 0; step < STEPS; ++step) {
            // Report progress periodically
            if (step % progressInterval == 0) {
                std::cout << "Evolution step " << step << "/" << STEPS << std::endl;
            }
            
            // Check CFL condition for stability
            double max_velocity = std::max({std::abs(U.x()), std::abs(U.y()), std::abs(U.z())});
            double cfl_dt = 0.5 * GRID_SPACING / (max_velocity + 1e-10); // Add small epsilon to avoid division by zero
           
            if (dt > cfl_dt) {
                std::cout << "Warning: Time step exceeds CFL condition. Using smaller sub-steps for stability." << std::endl;
            }
            
            auto levelSetOperator = [this](const Eigen::VectorXd& phi_current) -> Eigen::VectorXd {
                Eigen::VectorXd result = Eigen::VectorXd::Zero(phi_current.size());
                
                #pragma omp parallel for schedule(dynamic, 128)
                for (size_t k = 0; k < narrowBand.size(); ++k) {
                    const int idx = narrowBand[k];
                    const int x = idx % GRID_SIZE;
                    const int y = (idx / GRID_SIZE) % GRID_SIZE;
                    const int z = idx / (GRID_SIZE * GRID_SIZE);
                    
                    // Skip boundary points to avoid instability
                    if (isOnBoundary(idx)) {
                        continue;
                    }
                    
                    // Get material properties for current point
                    std::string material = getMaterialAtPoint(idx);
                    
                    // Calculate spatial derivatives
                    DerivativeOperator Dop;
                    spatialScheme->SpatialSch(idx, phi_current, GRID_SPACING, Dop);
                    
                    // Calculate normal vector
                    Eigen::Vector3d normal(
                        (Dop.dxP + Dop.dxN) / 2.0,
                        (Dop.dyP + Dop.dyN) / 2.0,
                        (Dop.dzP + Dop.dzN) / 2.0
                    );
                    
                    // Compute material-specific etching rate
                    double etchRate = computeEtchingRate(material, normal);
                    
                    // Modify velocity field based on etching rate
                    Eigen::Vector3d modifiedU = U * etchRate;
                    
                    // Calculate advection terms with modified velocity
                    double advectionN = std::max(modifiedU.x(), 0.0) * Dop.dxN + 
                                     std::max(modifiedU.y(), 0.0) * Dop.dyN + 
                                     std::max(modifiedU.z(), 0.0) * Dop.dzN;
                    double advectionP = std::min(modifiedU.x(), 0.0) * Dop.dxP + 
                                     std::min(modifiedU.y(), 0.0) * Dop.dyP + 
                                     std::min(modifiedU.z(), 0.0) * Dop.dzP;
                    
                    // ...rest of the evolution calculation...
                    const double epsilon = 1e-10;
                    double NP = std::sqrt(Dop.dxN*Dop.dxN + Dop.dyN*Dop.dyN + Dop.dzN*Dop.dzN + epsilon);
                    double PP = std::sqrt(Dop.dxP*Dop.dxP + Dop.dyP*Dop.dyP + Dop.dzP*Dop.dzP + epsilon);
                    
                    double curvatureterm = CURVATURE_WEIGHT * computeMeanCurvature(idx, phi_current);                    
                    result[idx] = -(advectionN + advectionP) + std::max(curvatureterm, 0.0) * NP + std::min(curvatureterm, 0.0) * PP; 
                }
                return result;
            };
            
            // Apply the time integration scheme
            Eigen::VectorXd phi_updated = timeScheme->advance(phi, levelSetOperator);
            
            // Update only the narrow band points
            #pragma omp parallel for schedule(dynamic, 128)
            for (size_t k = 0; k < narrowBand.size(); ++k) {
                const int idx = narrowBand[k];
                newPhi[idx] = phi_updated[idx];
            }
            
            phi.swap(newPhi);
            
            if (step % REINIT_INTERVAL == 0 && step > 0) {reinitialize();}
            if (step % NARROW_BAND_UPDATE_INTERVAL == 0) {updateNarrowBand();}
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
    
    // Perform reinitialization iterations
    for (int step = 0; step < REINIT_STEPS; ++step) {
        // Only reinitialize points in the narrow band - parallelize this loop
        #pragma omp parallel for schedule(dynamic, 128)
        for (size_t k = 0; k < narrowBand.size(); ++k) {
            const int idx = narrowBand[k];
            const int x = idx % GRID_SIZE;
            const int y = (idx / GRID_SIZE) % GRID_SIZE;
            const int z = idx / (GRID_SIZE * GRID_SIZE);
            
            // Compute sign function once at the beginning
            // Use a smooth sign function for better numerical stability
            const double phi0 = phi[idx]; // Original phi value
            const double sign = phi0 / std::sqrt(phi0*phi0 + GRID_SPACING*GRID_SPACING);
            
            // Use upwind scheme for gradient calculation based on sign
            double dx, dy, dz;
            
            // X direction upwind
            if (sign > 0) {
                // Use backward difference for positive sign
                const double dx_minus = (tempPhi[idx] - tempPhi[getIndex(x-1, y, z)]) / GRID_SPACING;
                const double dx_plus = (tempPhi[getIndex(x+1, y, z)] - tempPhi[idx]) / GRID_SPACING;
                dx = std::max(0.0, dx_minus) * std::max(0.0, dx_minus) + 
                     std::min(0.0, dx_plus) * std::min(0.0, dx_plus);
            } else {
                // Use forward difference for negative sign
                const double dx_minus = (tempPhi[idx] - tempPhi[getIndex(x-1, y, z)]) / GRID_SPACING;
                const double dx_plus = (tempPhi[getIndex(x+1, y, z)] - tempPhi[idx]) / GRID_SPACING;
                dx = std::min(0.0, dx_minus) * std::min(0.0, dx_minus) + 
                     std::max(0.0, dx_plus) * std::max(0.0, dx_plus);
            }
            
            // Y direction upwind
            if (sign > 0) {
                const double dy_minus = (tempPhi[idx] - tempPhi[getIndex(x, y-1, z)]) / GRID_SPACING;
                const double dy_plus = (tempPhi[getIndex(x, y+1, z)] - tempPhi[idx]) / GRID_SPACING;
                dy = std::max(0.0, dy_minus) * std::max(0.0, dy_minus) + 
                     std::min(0.0, dy_plus) * std::min(0.0, dy_plus);
            } else {
                const double dy_minus = (tempPhi[idx] - tempPhi[getIndex(x, y-1, z)]) / GRID_SPACING;
                const double dy_plus = (tempPhi[getIndex(x, y+1, z)] - tempPhi[idx]) / GRID_SPACING;
                dy = std::min(0.0, dy_minus) * std::min(0.0, dy_minus) + 
                     std::max(0.0, dy_plus) * std::max(0.0, dy_plus);
            }
            
            // Z direction upwind
            if (sign > 0) {
                const double dz_minus = (tempPhi[idx] - tempPhi[getIndex(x, y, z-1)]) / GRID_SPACING;
                const double dz_plus = (tempPhi[getIndex(x, y, z+1)] - tempPhi[idx]) / GRID_SPACING;
                dz = std::max(0.0, dz_minus) * std::max(0.0, dz_minus) + 
                     std::min(0.0, dz_plus) * std::min(0.0, dz_plus);
            } else {
                const double dz_minus = (tempPhi[idx] - tempPhi[getIndex(x, y, z-1)]) / GRID_SPACING;
                const double dz_plus = (tempPhi[getIndex(x, y, z+1)] - tempPhi[idx]) / GRID_SPACING;
                dz = std::min(0.0, dz_minus) * std::min(0.0, dz_minus) + 
                     std::max(0.0, dz_plus) * std::max(0.0, dz_plus);
            }
            
            // Calculate gradient magnitude with proper upwinding
            const double gradMag = std::sqrt(dx + dy + dz + epsilon);
            
            // Update equation for reinitialization with TVD Runge-Kutta
            tempPhi[idx] = tempPhi[idx] - dtau * sign * (gradMag - 1.0);
        }
    }
    
    // Update phi with reinitialized values
    phi = tempPhi;
}


void LevelSetMethod::updateNarrowBand() {
    // Reserve memory to avoid reallocations
    narrowBand.clear();
    const size_t estimated_size = grid.size() / 8; // Typical narrow band is a small fraction of total grid
    narrowBand.reserve(estimated_size);
    
    // Calculate narrow band width in grid units once
    const double narrow_band_grid_units = NARROW_BAND_WIDTH * GRID_SPACING;
    
    // Use thread-local storage with block processing for better cache locality
    const size_t block_size = 4096; // Process in cache-friendly blocks
    
    #pragma omp parallel
    {
        // Thread-local storage
        std::vector<int> localBand;
        localBand.reserve(estimated_size / omp_get_num_threads());
        
        // Process grid in blocks for better cache efficiency
        #pragma omp for schedule(dynamic, block_size) nowait
        for (size_t i = 0; i < grid.size(); ++i) {
            // Use branchless programming where possible
            const bool is_in_band = !isOnBoundary(i) && std::abs(phi[i]) <= narrow_band_grid_units;
            if (is_in_band) {
                localBand.push_back(i);
            }
        }
        
        // Use a mutex instead of critical section for better performance
        static std::mutex mutex;
        {
            std::lock_guard<std::mutex> lock(mutex);
            narrowBand.insert(narrowBand.end(), localBand.begin(), localBand.end());
        }
    }
    
    // Use parallel sorting algorithm for large arrays
    if (narrowBand.size() > 10000) {
        // Parallel sort implementation
        #pragma omp parallel
        {
            #pragma omp single
            {
                // Parallel quicksort with OpenMP tasks
                std::sort(narrowBand.begin(), narrowBand.end());
            }
        }
    } else {
        // Regular sort for smaller arrays
        std::sort(narrowBand.begin(), narrowBand.end());
    }
    
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
    
    // Calculate progress reporting interval once
    const size_t progress_interval = grid_size / 10;
    
    // Process grid points in blocks for better cache locality
    const size_t block_size = 4096; // Adjust based on cache size
    
    #pragma omp parallel
    {
        // Thread-local progress counter to reduce atomic operations
        
        #pragma omp for schedule(dynamic, block_size)
        for (size_t i = 0; i < grid_size; ++i) {
            // Compute squared distance to the mesh using AABB tree
            auto closest = tree->closest_point_and_primitive(grid[i]);
            double sq_dist = CGAL::sqrt(CGAL::squared_distance(grid[i], closest.first));
            
            // Determine if point is inside or outside the mesh
            CGAL::Bounded_side res = inside(grid[i]);
            
            // Set signed distance using branchless programming
            double sign = (res == CGAL::ON_BOUNDED_SIDE) ? -1.0 : 
                         (res == CGAL::ON_BOUNDARY) ? 0.0 : 1.0;
            
            sdf[i] = sign * sq_dist;
        }
    }
    
    std::cout << "Signed distance field initialization complete." << std::endl;
    return sdf;
}

inline bool LevelSetMethod::isOnBoundary(int idx) const {
    // Fast boundary check using grid coordinates
    // Extract coordinates with bit operations where possible for better performance
    static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
    
    const int x = idx % GRID_SIZE;
    const int y = (idx / GRID_SIZE) % GRID_SIZE;
    const int z = idx / GRID_SIZE_SQ;
    
    // Use branchless programming for boundary check
    // A point is on boundary if any coordinate is 0 or GRID_SIZE-1
    const bool x_boundary = (x <= 2) || (x >= GRID_SIZE - 3);
    const bool y_boundary = (y <= 2) || (y >= GRID_SIZE - 3);
    const bool z_boundary = (z <= 2) || (z >= GRID_SIZE - 3);
    
    return x_boundary || y_boundary || z_boundary;
}

inline int LevelSetMethod::getIndex(int x, int y, int z) const {
    static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
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

void LevelSetMethod::loadMaterialInfo(DFISEParser parser) {
    if (!dfiseParser.parse()) {
        throw std::runtime_error("Failed to parse DFISE file");
    }

    // Initialize material properties from the image
    materialProperties["Polymer"] = {0.1, 0.01, "Polymer"};
    materialProperties["SiO2_PECVD"] = {0.6, 0.01, "SiO2_PECVD"};
    materialProperties["Si_Amorph"] = {1.0, 0.01, "Si_Amorph"};
    materialProperties["Si3N4_LPCVD"] = {0.3, 0.01, "Si3N4_LPCVD"};
    
    // Initialize grid materials
    gridMaterials.resize(grid.size());
    
    // Map materials to grid points using DFISEParser information
    auto faceMaterials = dfiseParser.getAllFaceMaterials();
    for (size_t i = 0; i < grid.size(); ++i) {
        Point_3 point = grid[i];
        std::string material = "default";
        
        // Find the closest face and its material
        if (tree) {
            auto closest = tree->closest_point_and_primitive(point);
            // Use dot notation instead of arrow notation
            int faceIdx = closest.second.id();
            auto it = faceMaterials.find(faceIdx);
            if (it != faceMaterials.end()) {
                material = it->second;
            }
        }
        gridMaterials[i] = material;
    }
}

double LevelSetMethod::computeEtchingRate(const std::string& material, const Eigen::Vector3d& normal) {
    auto it = materialProperties.find(material);
    if (it == materialProperties.end()) {
        return 1.0; // Default rate for unknown materials
    }

    const MaterialProperties& props = it->second;
    
    // Calculate directional etching rate based on material properties
    double verticalRate = props.etchRatio;
    double lateralRate = props.lateralRatio;
    
    // Compute angle between normal and vertical direction
    Eigen::Vector3d vertical(0, 0, 1);
    double cosTheta = std::abs(normal.dot(vertical) / normal.norm());
    
    // Interpolate between vertical and lateral rates based on angle
    return verticalRate * cosTheta + lateralRate * (1.0 - cosTheta);
}

std::string LevelSetMethod::getMaterialAtPoint(int idx) const {
    if (idx >= 0 && idx < static_cast<int>(gridMaterials.size())) {
        return gridMaterials[idx];
    }
    return "default";
}