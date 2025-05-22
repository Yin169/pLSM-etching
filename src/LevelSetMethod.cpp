#include "LevelSetMethod.hpp"
#include <stdexcept>    // For std::out_of_range
#include <iostream>     // For std::cerr (error reporting)
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/refine.h>
#include <CGAL/Polygon_mesh_processing/smooth_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>

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
        
        Eigen::VectorXd newPhi = phi;
        
        // Progress tracking
        const int progressInterval = 10;
        const double inv_grid_spacing = 1.0 / GRID_SPACING;
        
        
        for (int step = 0; step < STEPS; ++step) {
            // Report progress periodically
            if (step % progressInterval == 0) {
                std::cout << "Evolution step " << step << "/" << STEPS << std::endl;
            }
            
            
            auto levelSetOperator = [this](const Eigen::VectorXd& phi_current) -> Eigen::VectorXd {
                Eigen::VectorXd result = Eigen::VectorXd::Zero(phi_current.size());
                
                #pragma omp parallel for schedule(dynamic, 128)
                for (size_t k = 0; k < narrowBand.size(); ++k) {
                    const int idx = narrowBand[k];
                    
                    // Skip boundary points to avoid instability
                    if (isOnBoundary(idx)) {
                        continue;
                    }
                    
                    // Get material properties for current point
                    std::string material = getMaterialAtPoint(idx);
                    
                    // Calculate spatial derivatives
                    DerivativeOperator Dop;
                    spatialScheme->SpatialSch(idx, phi_current, GRID_SPACING, Dop);
                    
                    
                    Eigen::Vector3d modifiedU_components;
                    const auto it = materialProperties.find(material);

                    if (it != materialProperties.end()) { 
                        const auto& props = it->second;  
                        const double lateral_etch = props.lateralRatio * props.etchRatio; 
                        modifiedU_components << lateral_etch, lateral_etch, props.etchRatio; 
                    } else {
                        modifiedU_components.setZero();  
                    }

                    Eigen::Vector3d modifiedU = modifiedU_components;
                    modifiedU *= -1;
                    
                    double advectionN = std::max(modifiedU.x(), 0.0) * Dop.dxN + 
                                     std::max(modifiedU.y(), 0.0) * Dop.dyN + 
                                     std::max(modifiedU.z(), 0.0) * Dop.dzN;
                    double advectionP = std::min(modifiedU.x(), 0.0) * Dop.dxP + 
                                     std::min(modifiedU.y(), 0.0) * Dop.dyP + 
                                     std::min(modifiedU.z(), 0.0) * Dop.dzP;
                        
                    result[idx] = -(advectionN + advectionP); 
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
            if (step % NARROW_BAND_UPDATE_INTERVAL == 0 && step > 0) {updateNarrowBand();}
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
    const int REINIT_STEPS = 10;
    const double dtau = std::min(dt, 0.1); // Time step for reinitialization
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
    const size_t estimated_size = grid.size();
    narrowBand.reserve(estimated_size);
    
    // Calculate narrow band width in grid units once
    const double narrow_band_grid_units = NARROW_BAND_WIDTH ;
    
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

bool LevelSetMethod::smoothShape(double smoothingFactor = 0.5, int iterations = 5) {
    try {
        if (phi.size() != grid.size()) {
            throw std::runtime_error("Level set function not initialized.");
        }

        std::cout << "Applying level set shape smoothing with factor: " << smoothingFactor 
                  << " and iterations: " << iterations << std::endl;
        
        // Create a copy of the original level set function
        Eigen::VectorXd phi_original = phi;
        Eigen::VectorXd phi_temp = phi;
        
        // Apply multiple iterations of smoothing for better results
        for (int iter = 0; iter < iterations; iter++) {
            std::cout << "Smoothing iteration " << (iter + 1) << "/" << iterations << std::endl;
            
            // Apply Gaussian smoothing to the level set function
            #pragma omp parallel for
            for (int i = 1; i < GRID_SIZE - 1; i++) {
                for (int j = 1; j < GRID_SIZE - 1; j++) {
                    for (int k = 1; k < GRID_SIZE - 1; k++) {
                        int idx = i + j * GRID_SIZE + k * GRID_SIZE * GRID_SIZE;
                        
                        // Apply 3D convolution with a simple kernel
                        double sum = 0.0;
                        double weight_sum = 0.0;
                        
                        // 3x3x3 neighborhood
                        for (int di = -1; di <= 1; di++) {
                            for (int dj = -1; dj <= 1; dj++) {
                                for (int dk = -1; dk <= 1; dk++) {
                                    int ni = i + di;
                                    int nj = j + dj;
                                    int nk = k + dk;
                                    
                                    // Skip out-of-bounds indices
                                    if (ni < 0 || ni >= GRID_SIZE || 
                                        nj < 0 || nj >= GRID_SIZE || 
                                        nk < 0 || nk >= GRID_SIZE) {
                                        continue;
                                    }
                                    
                                    int nidx = ni + nj * GRID_SIZE + nk * GRID_SIZE * GRID_SIZE;
                                    
                                    // Gaussian-like weighting based on distance
                                    double dist = std::sqrt(di*di + dj*dj + dk*dk);
                                    double weight = std::exp(-dist * dist);
                                    
                                    sum += phi[nidx] * weight;
                                    weight_sum += weight;
                                }
                            }
                        }
                        
                        // Weighted average
                        double smoothed_value = sum / weight_sum;
                        
                        // Apply smoothing factor (blend between original and smoothed)
                        phi_temp[idx] = phi[idx] * (1.0 - smoothingFactor) + smoothed_value * smoothingFactor;
                    }
                }
            }
            
            // Update phi with the smoothed values for the next iteration
            phi = phi_temp;
        }
        
        // Preserve the zero level set location by adjusting values near the interface
        // This helps prevent volume loss during smoothing
        #pragma omp parallel for
        for (int i = 0; i < GRID_SIZE; i++) {
            for (int j = 0; j < GRID_SIZE; j++) {
                for (int k = 0; k < GRID_SIZE; k++) {
                    int idx = i + j * GRID_SIZE + k * GRID_SIZE * GRID_SIZE;
                    
                    // If the sign changed during smoothing (crossing the zero level set)
                    if (phi_original[idx] * phi[idx] < 0) {
                        // Adjust the value to be closer to zero but maintain the sign
                        // This helps preserve the original surface location
                        phi[idx] = phi[idx] * 0.5;
                    }
                }
            }
        }
        
        std::cout << "Shape smoothing completed successfully." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in shape smoothing: " << e.what() << std::endl;
        return false;
    }
}

// Enhanced extractSurfaceMeshCGAL method with shape smoothing integration
bool LevelSetMethod::extractSurfaceMeshCGAL(const std::string& filename, 
                                        bool smoothSurface = true, 
                                        bool refineMesh = true, 
                                        bool remeshSurface = true,
                                        int smoothingIterations = 5, 
                                        double targetEdgeLength = -1.0,
                                        bool smoothShape = true,             // New parameter
                                        double shapeSmoothing = 0.5,          // New parameter
                                        int shapeSmoothingIterations = 5) {   // New parameter
    try {
        if (phi.size() != grid.size()) {
            throw std::runtime_error("Level set function not initialized.");
        }
        
        // Apply shape smoothing to the level set function if requested
        if (smoothShape) {
            // Create a backup of the original level set in case smoothing fails
            Eigen::VectorXd phi_backup = phi;
            
            // Apply smoothing to the level set function itself
            bool smoothingSuccess = this->smoothShape(shapeSmoothing, shapeSmoothingIterations);
            
            if (!smoothingSuccess) {
                std::cerr << "Warning: Shape smoothing failed, reverting to original level set." << std::endl;
                phi = phi_backup;
            } else {
                std::cout << "Applied shape smoothing to level set function." << std::endl;
            }
        }
 
        typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
        typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
        typedef Tr::Geom_traits GT;
        typedef GT::Sphere_3 Sphere_3;
        typedef GT::FT FT;
        typedef std::function<FT(typename GT::Point_3)> Function;
        typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;
        typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
    
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
        
        // If refinement is requested, use finer criteria
        double facet_angle = refineMesh ? 25.0 : 30.0;
        double facet_size = refineMesh ? GRID_SPACING * 1.5 : GRID_SPACING * 2.0;
        double facet_distance = refineMesh ? GRID_SPACING * 1.5 : GRID_SPACING * 2.0;
        
        Criteria criteria(facet_angle, facet_size, facet_distance);
        
        // Define the mesh data structure
        Surface_mesh surface_mesh;
        
        std::cout << "Starting surface mesh generation..." << std::endl;
        // Generate the surface mesh
        CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());
        std::cout << "Surface mesh generation completed." << std::endl;
        
        // Convert the complex to a surface mesh
        CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, surface_mesh);
        
        // Get initial mesh statistics
        std::size_t initial_vertices = surface_mesh.number_of_vertices();
        std::size_t initial_faces = surface_mesh.number_of_faces();
        std::cout << "Initial mesh: " << initial_vertices << " vertices, " 
                  << initial_faces << " faces." << std::endl;
        
        // Apply mesh processing if requested
        if (smoothSurface || refineMesh || remeshSurface) {
            // Define property maps for mesh processing
            typedef boost::property_map<Surface_mesh, CGAL::vertex_point_t>::type VPMap;
            VPMap vpmap = get(CGAL::vertex_point, surface_mesh);
            
            // Set target edge length for refinement/remeshing if not specified
            if (targetEdgeLength < 0.0) {
                // Calculate average edge length as default target
                double sum_edge_length = 0.0;
                std::size_t edge_count = 0;
                
                for (auto e : edges(surface_mesh)) {
                    auto v_source = source(e, surface_mesh);
                    auto v_target = target(e, surface_mesh);
                    
                    Point_3 p_source = get(vpmap, v_source);
                    Point_3 p_target = get(vpmap, v_target);
                    
                    sum_edge_length += std::sqrt(CGAL::squared_distance(p_source, p_target));
                    edge_count++;
                }
                
                targetEdgeLength = (edge_count > 0) ? 
                    (sum_edge_length / edge_count) * 0.8 : GRID_SPACING;
            }

            // Apply surface smoothing if requested
            if (smoothSurface) {
                std::cout << "Applying surface smoothing with " << smoothingIterations 
                          << " iterations..." << std::endl;
                
                namespace PMP = CGAL::Polygon_mesh_processing;
                
                // First, ensure mesh is manifold and has no boundaries
                if (!CGAL::is_triangle_mesh(surface_mesh)) {
                    std::cout << "Warning: Input mesh is not triangular, triangulating first..." << std::endl;
                    PMP::triangulate_faces(surface_mesh);
                }
                
                // Close holes if they exist
                std::vector<Surface_mesh::halfedge_index> border_halfedges;
                PMP::extract_boundary_cycles(surface_mesh, std::back_inserter(border_halfedges));
                
                std::size_t num_holes = 0;
                std::vector<Surface_mesh::face_index> new_faces;
                
                // For each border halfedge, triangulate the corresponding hole
                for (auto h : border_halfedges) {
                    std::vector<Surface_mesh::face_index> hole_faces;
                    PMP::triangulate_hole(surface_mesh, h, std::back_inserter(hole_faces));
                    if (!hole_faces.empty()) {
                        num_holes++;
                        new_faces.insert(new_faces.end(), hole_faces.begin(), hole_faces.end());
                    }
                }
                
                if (num_holes > 0) {
                    std::cout << "Closed " << num_holes << " holes with " << new_faces.size() 
                              << " new faces." << std::endl;
                }
                
                // Apply Laplacian smoothing while preserving volume
                PMP::smooth_mesh(surface_mesh, 
                              PMP::parameters::number_of_iterations(smoothingIterations)
                              .use_safety_constraints(true)
                              .vertex_point_map(vpmap));
                
                std::cout << "Surface smoothing completed." << std::endl;
            }
        } 
        // Get final mesh statistics
        std::size_t final_vertices = surface_mesh.number_of_vertices();
        std::size_t final_faces = surface_mesh.number_of_faces();
        
        // Save the surface mesh to a file
        if (!CGAL::IO::write_polygon_mesh(filename, surface_mesh, CGAL::parameters::stream_precision(17))) {
            throw std::runtime_error("Failed to write surface mesh to file.");
        }
        
        std::cout << "Surface mesh extracted and saved to " << filename << std::endl;
        std::cout << "Final surface mesh has " << final_vertices << " vertices and " 
                  << final_faces << " faces" << std::endl;
        
        // Report changes if processing was applied
        if (smoothSurface || refineMesh || remeshSurface) {
            double vertex_change = ((double)final_vertices - initial_vertices) / initial_vertices * 100.0;
            double face_change = ((double)final_faces - initial_faces) / initial_faces * 100.0;
            
            std::cout << "Mesh processing results:" << std::endl;
            std::cout << "  Vertex count: " << initial_vertices << " -> " << final_vertices 
                      << " (" << (vertex_change >= 0 ? "+" : "") << vertex_change << "%)" << std::endl;
            std::cout << "  Face count: " << initial_faces << " -> " << final_faces 
                      << " (" << (face_change >= 0 ? "+" : "") << face_change << "%)" << std::endl;
        }
        
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

bool LevelSetMethod::exportGridMaterialsToCSV(const std::string& filename) {
    try {
        std::ofstream csvFile(filename);
        if (!csvFile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
            return false;
        }
        
        // Write CSV header
        csvFile << "x,y,z,material" << std::endl;
        
        // Write each grid point and its material
        for (size_t i = 0; i < grid.size(); ++i) {
            const Point_3& point = grid[i];
            std::string material = getMaterialAtPoint(i);
            
            csvFile << point.x() << "," 
                    << point.y() << "," 
                    << point.z() << "," 
                    << material << std::endl;
        }
        
        csvFile.close();
        std::cout << "Successfully exported grid materials to " << filename << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error exporting grid materials to CSV: " << e.what() << std::endl;
        return false;
    }
}