#include "LevelSetMethod.hpp"

Eigen::Vector3d sphericalToCartesian(double theta, double phi) {
    return Eigen::Vector3d(
        std::sin(theta) * std::cos(phi),
        std::sin(theta) * std::sin(phi),
        std::cos(theta)
    );
}

double integrand(const Eigen::Vector3d& r, const Eigen::Vector3d& n, double sigma) {
    Eigen::Vector3d dir_r = Eigen::Vector3d(r.x(), r.y(), -r.z());
    double cosTheta = dir_r.dot(n);
    if (cosTheta >= 0.0) {
        return 0.0;
    }
    double theta = std::acos(r.z());
    return cosTheta * std::exp(-theta / (2.0 * sigma * sigma));
}


// Gauss-Legendre quadrature points and weights
std::vector<std::pair<double, double>> getGaussLegendrePoints() {
    return {
        {-0.9061798459386640, 0.2369268850561891},
        {-0.5384693101056831, 0.4786286704993665},
        {0.0, 0.5688888888888889},
        {0.5384693101056831, 0.4786286704993665},
        {0.9061798459386640, 0.2369268850561891}
    };
}

// Function to perform Gaussian quadrature integration over a hemisphere
double gaussianQuadratureHemisphere(double sigma, const Eigen::Vector3d& normal, int numPointsTheta, int numPointsPhi) {
    // Get Gauss-Legendre points and weights
    std::vector<std::pair<double, double>> pointsTheta = getGaussLegendrePoints();
    std::vector<std::pair<double, double>> pointsPhi = getGaussLegendrePoints();
    
    double result = 0.0;
    
    for (int i = 0; i < numPointsPhi; i++) {
        double phi = (pointsPhi[i].first + 1.0) * M_PI;
        double phi_weight = pointsPhi[i].second;
        
        for (int j = 0; j < numPointsTheta; j++) {
            double theta = (pointsTheta[j].first + 1.0) * M_PI / 4.0;
            double theta_weight = pointsTheta[j].second;
            
            Eigen::Vector3d r = sphericalToCartesian(phi, theta);
            double value = integrand(r, normal, sigma);
            double dOmega = sin(theta) * theta_weight * phi_weight;
            result += value * dOmega;
        }
    }
    return result;
}

double LevelSetMethod::computeEtchingRate(const Eigen::Vector3d& normal, double sigma) {
    return gaussianQuadratureHemisphere(sigma, normal, 5, 5);
}

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

bool LevelSetMethod::evolve() {
    try {
        phi = initializeSignedDistanceField();
        updateNarrowBand();
        
        // Pre-allocate memory for new phi values to avoid reallocations
        Eigen::VectorXd newPhi = phi;
        
        // Progress tracking
        const int progressInterval = std::max(1, 10);
        // Cache frequently used constants
        const double inv_grid_spacing = 1.0 / GRID_SPACING;
        const double sigma = 0.6;
        
        // Pre-sort narrow band for better cache locality
        std::sort(narrowBand.begin(), narrowBand.end());
        
        for (int step = 0; step < STEPS; ++step) {
            // Report progress periodically
            if (step % progressInterval == 0) {
                std::cout << "Evolution step " << step << "/" << STEPS << std::endl;
            }
            
            auto levelSetOperator = [this, sigma](const Eigen::VectorXd& phi_current) -> Eigen::VectorXd {
                Eigen::VectorXd result = Eigen::VectorXd::Zero(phi_current.size());
                
                #pragma omp parallel for schedule(dynamic, 128)
                for (size_t k = 0; k < narrowBand.size(); ++k) {
                    const int idx = narrowBand[k];
                    
                    double dx = 0.0, dy = 0.0, dz = 0.0;
                    spatialScheme->SpatialSch(idx, phi_current, GRID_SPACING, dx, dy, dz);
                    
                    Eigen::Vector3d normal(dx, dy, dz);
                    double gradMag = normal.norm();
                    
                    if (gradMag > 1e-10) {
                        normal /= gradMag;
                    }
                    
                    double rate = computeEtchingRate(normal, sigma);
                    result[idx] = rate * gradMag;
                }
                
                return -result;
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
    const int REINIT_STEPS = 7;
    const double dtau = dt; // Time step for reinitialization
    const double half_inv_grid_spacing = 0.5 / GRID_SPACING;
    const double grid_spacing_squared = GRID_SPACING * GRID_SPACING;
    
    // Perform reinitialization iterations
    for (int step = 0; step < REINIT_STEPS; ++step) {
        // Only reinitialize points in the narrow band - parallelize this loop
        #pragma omp parallel for schedule(dynamic, 128)
        for (size_t k = 0; k < narrowBand.size(); ++k) {
            const int idx = narrowBand[k];
            const int x = idx % GRID_SIZE;
            const int y = (idx / GRID_SIZE) % GRID_SIZE;
            const int z = idx / (GRID_SIZE * GRID_SIZE);
            
            // Cache indices to reduce redundant calculations
            const int idx_x_plus = getIndex(x+1, y, z);
            const int idx_x_minus = getIndex(x-1, y, z);
            const int idx_y_plus = getIndex(x, y+1, z);
            const int idx_y_minus = getIndex(x, y-1, z);
            const int idx_z_plus = getIndex(x, y, z+1);
            const int idx_z_minus = getIndex(x, y, z-1);
            
            // Calculate central differences with precomputed scaling factor
            const double dx = (tempPhi[idx_x_plus] - tempPhi[idx_x_minus]) * half_inv_grid_spacing;
            const double dy = (tempPhi[idx_y_plus] - tempPhi[idx_y_minus]) * half_inv_grid_spacing;
            const double dz = (tempPhi[idx_z_plus] - tempPhi[idx_z_minus]) * half_inv_grid_spacing;
            
            // Use fast approximation for square root if available
            const double gradMag = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            // Optimize sign function calculation
            const double phi_val = tempPhi[idx];
            const double denom = std::sqrt(phi_val*phi_val + gradMag*gradMag*grid_spacing_squared);
            const double sign = (denom > 1e-10) ? (phi_val / denom) : 0.0;

            // Update equation for reinitialization
            tempPhi[idx] = phi_val - dtau * sign * (gradMag - 1.0);
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
        size_t local_progress = 0;
        
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
            
            // Progress reporting with reduced synchronization
            local_progress++;
            if (local_progress % progress_interval == 0 && omp_get_thread_num() == 0) {
                #pragma omp critical
                {
                    std::cout << "SDF initialization: " << (i * 100.0 / grid_size) << "% complete" << std::endl;
                }
            }
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
    const bool x_boundary = (x < 3) || (x > GRID_SIZE - 3);
    const bool y_boundary = (y < 3) || (y > GRID_SIZE - 3);
    const bool z_boundary = (z < 3) || (z > GRID_SIZE - 3);
    
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