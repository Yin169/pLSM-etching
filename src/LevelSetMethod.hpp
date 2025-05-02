#ifndef LEVEL_SET_METHOD_HPP
#define LEVEL_SET_METHOD_HPP

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Surface_mesh_default_criteria_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Surface_mesh_complex_2_in_triangulation_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/IO/polygon_mesh_io.h>

#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Surface_mesh.h>


#include <eigen3/Eigen/Dense>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <functional>
<<<<<<< Updated upstream
#include <bitset>
#include <unordered_map>
=======
#include <unordered_map>
#include <limits>
#include <omp.h>
#include <mutex>
#include <chrono>
>>>>>>> Stashed changes

namespace PMP = CGAL::Polygon_mesh_processing;


typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

class LevelSetMethod {
    public:
        LevelSetMethod(const std::string& filename,
                     int gridSize = 400, 
                     double timeStep = 0.01, 
                     int maxSteps = 80, 
                     int reinitInterval = 5,
                     double narrowBandWidth = 10.0)
            : GRID_SIZE(gridSize),
              dt(timeStep),
              STEPS(maxSteps),
              REINIT_INTERVAL(reinitInterval),
              NARROW_BAND_WIDTH(narrowBandWidth) {
            loadMesh(filename);
            generateGrid();
            precomputeDirections(20, 40);  // 预计算方向向量
        }
    
        CGAL::Bbox_3 calculateBoundingBox() const {
            if (mesh.is_empty()) {
                throw std::runtime_error("Mesh is empty - cannot calculate bounding box");
            }
            return CGAL::Polygon_mesh_processing::bbox(mesh);
        }
<<<<<<< Updated upstream
            
        tree = std::make_unique<AABB_tree>(faces(mesh).first, faces(mesh).second, mesh);
        // tree->build();
        tree->accelerate_distance_queries(); 
    }


    // Process a block of grid points in the narrow band for cache-oblivious evolution
    void processNarrowBandBlock(const std::vector<int>& blockIndices, Eigen::VectorXd& newPhi) {
        // Process points in the block
        for (const auto& idx : blockIndices) {
            // Get grid indices
            int x = idx % GRID_SIZE;
            int y = (idx / GRID_SIZE) % GRID_SIZE;
            int z = idx / (GRID_SIZE * GRID_SIZE);
            
            // Calculate spatial derivatives using central differences
            double dx_forward = (phi[getIndex(x+1, y, z)] - phi[idx]) / GRID_SPACING;
            double dx_backward = (phi[idx] - phi[getIndex(x-1, y, z)]) / GRID_SPACING;
            double dy_forward = (phi[getIndex(x, y+1, z)] - phi[idx]) / GRID_SPACING;
            double dy_backward = (phi[idx] - phi[getIndex(x, y-1, z)]) / GRID_SPACING;
            double dz_forward = (phi[getIndex(x, y, z+1)] - phi[idx]) / GRID_SPACING;
            double dz_backward = (phi[idx] - phi[getIndex(x, y, z-1)]) / GRID_SPACING;
            
            // Calculate gradient magnitude using upwind scheme
            double dx = std::max(dx_backward, 0.0) + std::min(dx_forward, 0.0);
            double dy = std::max(dy_backward, 0.0) + std::min(dy_forward, 0.0);
            double dz = std::max(dz_backward, 0.0) + std::min(dz_forward, 0.0);
            
            double gradMag = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            // Calculate extension speed F based on the first equation
            // Assuming gravity direction is (0, 0, -1) and sigma = 0.5
            double nx = dx / (gradMag + 1e-10);
            double ny = dy / (gradMag + 1e-10);
            double nz = dz / (gradMag + 1e-10);
            
            // Gravity direction (unit vector pointing downward)
            double gx = 0.0;
            double gy = 0.0;
            double gz = -1.0;
            
            // Calculate theta (angle between normal and gravity direction)
            double dotProduct = nx*gx + ny*gy + nz*gz;
            double theta = std::acos(std::min(std::max(dotProduct, -1.0), 1.0));
            
            // Calculate extension speed F
            double sigma = 14.0; // Parameter controlling angular spread
            double F = dotProduct * std::exp(-theta/(2*sigma*sigma));
            
            // Update level set function using the level set equation
            newPhi[idx] = phi[idx] - dt * (F * gradMag);
        }
    }
    
    // Process a block of grid points for cache-oblivious reinitialization
    void processReinitBlock(const std::vector<int>& blockIndices, Eigen::VectorXd& tempPhi, double dtau) {
        // Process points in the block
        for (const auto& idx : blockIndices) {
            int x = idx % GRID_SIZE;
            int y = (idx / GRID_SIZE) % GRID_SIZE;
            int z = idx / (GRID_SIZE * GRID_SIZE);
            
            double dx = (tempPhi[getIndex(x+1, y, z)] - tempPhi[getIndex(x-1, y, z)]) / (2*GRID_SPACING);
            double dy = (tempPhi[getIndex(x, y+1, z)] - tempPhi[getIndex(x, y-1, z)]) / (2*GRID_SPACING);
            double dz = (tempPhi[getIndex(x, y, z+1)] - tempPhi[getIndex(x, y, z-1)]) / (2*GRID_SPACING);
            
            double gradMag = std::sqrt(dx*dx + dy*dy + dz*dz);
           
            // Sign function
            double sign = tempPhi[idx] / std::sqrt(tempPhi[idx]*tempPhi[idx] + gradMag*gradMag*GRID_SPACING*GRID_SPACING); 

            // Update equation for reinitialization
            tempPhi[idx] = tempPhi[idx] - dtau * sign * (gradMag - 1.0);
        }
    }
    
    // Recursive subdivision for cache-oblivious processing
    void recursiveSubdivision(const std::vector<int>& indices, int start, int end, 
                             Eigen::VectorXd& newPhi, bool isReinit = false, 
                             Eigen::VectorXd* tempPhi = nullptr, double dtau = 0.0) {
        // Base case: small enough block to process directly
        const int BLOCK_SIZE = 64; // Adjust based on cache size
        if (end - start <= BLOCK_SIZE) {
            std::vector<int> blockIndices(indices.begin() + start, indices.begin() + end);
            if (isReinit && tempPhi) {
                processReinitBlock(blockIndices, *tempPhi, dtau);
            } else {
                processNarrowBandBlock(blockIndices, newPhi);
            }
            return;
        }
        
        // Recursive case: divide and conquer
        int mid = start + (end - start) / 2;
        recursiveSubdivision(indices, start, mid, newPhi, isReinit, tempPhi, dtau);
        recursiveSubdivision(indices, mid, end, newPhi, isReinit, tempPhi, dtau);
    }

    bool evolve() {
        try {
            // Initialize the signed distance field
            phi = initializeSignedDistanceField();
            
            // Initialize narrow band
            updateNarrowBand();
            
            // Main evolution loop
            for (int step = 0; step < STEPS; ++step) {
                // Create a copy of the current level set
                Eigen::VectorXd newPhi = phi;
                
                // Process narrow band using cache-oblivious recursive subdivision
                if (!narrowBand.empty()) {
                    recursiveSubdivision(narrowBand, 0, narrowBand.size(), newPhi);
                }
                
                phi = newPhi;
                
                // Reinitialization to maintain signed distance property
                if (step % REINIT_INTERVAL == 0 && step > 0) {
                    reinitialize();
                    // Update narrow band after reinitialization
                    updateNarrowBand();
                }

                if (step % 10 == 0) {
                    std::cout << "Step " << step << " completed. Narrow band size: " << narrowBand.size() << std::endl;
                }
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error during evolution: " << e.what() << std::endl;
            return false;
        }
    }

    void reinitialize() {
        // Create a temporary copy of phi
        Eigen::VectorXd tempPhi = phi;
        
        // Number of iterations for reinitialization
        const int REINIT_STEPS = 7;
        const double dtau = dt; // Time step for reinitialization
        
        // Perform reinitialization iterations
        for (int step = 0; step < REINIT_STEPS; ++step) {
            // Process narrow band using cache-oblivious recursive subdivision
            if (!narrowBand.empty()) {
                recursiveSubdivision(narrowBand, 0, narrowBand.size(), phi, true, &tempPhi, dtau);
            }
            
            // Copy updated values back to tempPhi for next iteration
            tempPhi = phi;
        }
    }


    bool saveResult(const std::string& filename) {
        try {
            std::ofstream output(filename);
            if (!output.is_open()) {
                std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
                return false;
            }
            
            output << "x,y,z,value" << std::endl;
            
            // Write the SDF values with coordinates
            for (size_t i = 0; i < grid.size(); ++i) {
                output << grid[i].x() << "," << grid[i].y() << "," << grid[i].z() << "," << phi[i] << std::endl;
            }
            
            output.close();
            std::cout << "Results saved to " << filename << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error saving results: " << e.what() << std::endl;
            return false;
        }
    }

private:
    // Remove BOX_SIZE from configuration parameters
    const int GRID_SIZE;
    double GRID_SPACING;  // Now calculated based on mesh bounds
    const double dt;
    const int STEPS;
    const int REINIT_INTERVAL;
    const double NARROW_BAND_WIDTH;
    double BOX_SIZE = -1.0;
    
    // Data structures
    Mesh mesh;
    std::unique_ptr<AABB_tree> tree;
    std::vector<Point_3> grid;
    Eigen::VectorXd phi;
    std::vector<int> narrowBand; // Indices of grid points in the narrow band
    std::vector<uint64_t> mortonCodes; // Morton codes for Z-order traversal
    std::unordered_map<uint64_t, size_t> mortonToIndex; // Maps morton code to grid index

    void updateNarrowBand() {
        narrowBand.clear();
        
        // Create a vector of pairs (morton code, grid index) for points in narrow band
        std::vector<std::pair<uint64_t, size_t>> narrowBandPoints;
        
        // First identify all points in the narrow band
        for (size_t i = 0; i < grid.size(); ++i) {
            if (isOnBoundary(i)) continue;
            
            if (std::abs(phi[i]) <= NARROW_BAND_WIDTH * GRID_SPACING) {
                // Get the morton code for this point
                uint64_t mortonCode = mortonCodes[i];
                narrowBandPoints.emplace_back(mortonCode, i);
            }
        }
        
        // Sort narrow band points by Morton code for cache-oblivious traversal
        std::sort(narrowBandPoints.begin(), narrowBandPoints.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Extract the indices in Z-order
        narrowBand.reserve(narrowBandPoints.size());
        for (const auto& point : narrowBandPoints) {
            narrowBand.push_back(point.second);
        }
=======
    
        bool extractSurfaceMeshCGAL(const std::string& filename);
    
        void loadMesh(const std::string& filename) {
            if (!PMP::IO::read_polygon_mesh(filename, mesh) || is_empty(mesh) || !CGAL::is_closed(mesh) || !is_triangle_mesh(mesh)) {
                std::cerr << "Error: Could not open file " << filename << std::endl;
                return;
            }
                
            tree = std::make_unique<AABB_tree>(faces(mesh).first, faces(mesh).second, mesh);
            tree->accelerate_distance_queries(); 
        }
    
        bool evolve() {
            try {
                phi = initializeSignedDistanceField();
                updateNarrowBand();
                
                // Pre-allocate memory for new phi values to avoid reallocations
                Eigen::VectorXd newPhi = phi;
                
                // Progress tracking
                int progressInterval = std::max(1, 10);
                
                for (int step = 0; step < STEPS; ++step) {
                    // Report progress periodically
                    if (step % progressInterval == 0) {
                        std::cout << "Evolution step " << step << "/" << STEPS << std::endl;
                    }
                    
                    // Use schedule(guided) for better load balancing with varying workloads
                    #pragma omp parallel for schedule(guided)
                    for (size_t k = 0; k < narrowBand.size(); ++k) {
                        int idx = narrowBand[k];
                        int x = idx % GRID_SIZE;
                        int y = (idx / GRID_SIZE) % GRID_SIZE;
                        int z = idx / (GRID_SIZE * GRID_SIZE);
                        
                        // 空间导数计算
                        double dx_forward = (phi[getIndex(x+1, y, z)] - phi[idx]) / GRID_SPACING;
                        double dx_backward = (phi[idx] - phi[getIndex(x-1, y, z)]) / GRID_SPACING;
                        double dy_forward = (phi[getIndex(x, y+1, z)] - phi[idx]) / GRID_SPACING;
                        double dy_backward = (phi[idx] - phi[getIndex(x, y-1, z)]) / GRID_SPACING;
                        double dz_forward = (phi[getIndex(x, y, z+1)] - phi[idx]) / GRID_SPACING;
                        double dz_backward = (phi[idx] - phi[getIndex(x, y, z-1)]) / GRID_SPACING;
                        
                        // 梯度计算
                        double dx = std::max(dx_backward, 0.0) + std::min(dx_forward, 0.0);
                        double dy = std::max(dy_backward, 0.0) + std::min(dy_forward, 0.0);
                        double dz = std::max(dz_backward, 0.0) + std::min(dz_forward, 0.0);
                        double gradMag = std::sqrt(dx*dx + dy*dy + dz*dz);
                        
                        // Avoid division by zero when computing normal
                        const double EPSILON = 1e-10;
                        double nx = dx / (gradMag + EPSILON);
                        double ny = dy / (gradMag + EPSILON);
                        double nz = dz / (gradMag + EPSILON);
                        Eigen::Vector3d normal(nx, ny, nz);
                        
                        // Compute etching rate with optimized parameters
                        const double sigma = 0.01; // Etching parameter
                        double F = computeEtchingRate(idx, grid[idx], normal, sigma);
                        
                        // Update level set value
                        newPhi[idx] = phi[idx] - dt * F * gradMag;
                    }
                    
                    // Swap phi and newPhi (more efficient than copying)
                    phi.swap(newPhi);
                    
                    // Reinitialize periodically to maintain signed distance property
                    if (step % REINIT_INTERVAL == 0 && step > 0) {
                        reinitialize();
                        updateNarrowBand();
                    }
                }
                
                std::cout << "Evolution completed successfully." << std::endl;
                return true;
            } catch (const std::exception& e) {
                std::cerr << "Error during evolution: " << e.what() << std::endl;
                return false;
            }
        }

        void reinitialize() {
            // Create a temporary copy of phi
            Eigen::VectorXd tempPhi = phi;
            
            // Number of iterations for reinitialization
            const int REINIT_STEPS = 7;
            const double dtau = dt; // Time step for reinitialization
            
            // Perform reinitialization iterations
            for (int step = 0; step < REINIT_STEPS; ++step) {
                // Only reinitialize points in the narrow band
                for (const auto& idx : narrowBand) {
                    int x = idx % GRID_SIZE;
                    int y = (idx / GRID_SIZE) % GRID_SIZE;
                    int z = idx / (GRID_SIZE * GRID_SIZE);
                    
                    double dx = (tempPhi[getIndex(x+1, y, z)] - tempPhi[getIndex(x-1, y, z)]) / (2*GRID_SPACING);
                    double dy = (tempPhi[getIndex(x, y+1, z)] - tempPhi[getIndex(x, y-1, z)]) / (2*GRID_SPACING);
                    double dz = (tempPhi[getIndex(x, y, z+1)] - tempPhi[getIndex(x, y, z-1)]) / (2*GRID_SPACING);
                    
                    double gradMag = std::sqrt(dx*dx + dy*dy + dz*dz);
                   
                    // Sign function
                    double sign = tempPhi[idx] / std::sqrt(tempPhi[idx]*tempPhi[idx] + gradMag*gradMag*GRID_SPACING*GRID_SPACING); 
    
                    // Update equation for reinitialization
                    tempPhi[idx] = tempPhi[idx] - dtau * sign * (gradMag - 1.0);
                }
            }
            
            // Update phi with reinitialized values
            phi = tempPhi;
        }

    private:
        const int GRID_SIZE;
        double GRID_SPACING;
        const double dt;
        const int STEPS;
        const int REINIT_INTERVAL;
        const double NARROW_BAND_WIDTH;
        double BOX_SIZE = -1.0;
        
        // Grid origin coordinates for faster lookups
        double gridOriginX = 0.0;
        double gridOriginY = 0.0;
        double gridOriginZ = 0.0;
        
        Mesh mesh;
        std::unique_ptr<AABB_tree> tree;
        std::vector<Point_3> grid;
        Eigen::VectorXd phi;
        std::vector<int> narrowBand;
        
        // Precomputed data for direction sampling
        std::vector<Eigen::Vector3d> precomputed_directions;
        std::vector<double> precomputed_dOmega;
    
        void precomputeDirections(int num_theta, int num_phi) {
            // Optimize sampling for better coverage with fewer samples
            const double d_theta = (M_PI/2) / num_theta;
            const double d_phi = (2*M_PI) / num_phi;
            
            // Reserve memory to avoid reallocations
            const int total_samples = num_theta * num_phi;
            precomputed_directions.clear();
            precomputed_directions.reserve(total_samples);
            precomputed_dOmega.clear();
            precomputed_dOmega.reserve(total_samples);
            
            std::cout << "Precomputing " << total_samples << " direction vectors..." << std::endl;
            
            // Use OpenMP for parallel computation
            #pragma omp parallel
            {
                // Thread-local storage for intermediate results
                std::vector<Eigen::Vector3d> local_directions;
                std::vector<double> local_dOmega;
                local_directions.reserve(total_samples / omp_get_num_threads());
                local_dOmega.reserve(total_samples / omp_get_num_threads());
                
                #pragma omp for schedule(static)
                for (int i = 0; i < num_theta; ++i) {
                    double theta = i * d_theta;
                    double sin_theta = std::sin(theta);
                    double cos_theta = std::cos(theta);
                    double solid_angle = sin_theta * d_theta * d_phi;
                    
                    for (int j = 0; j < num_phi; ++j) {
                        double phi = j * d_phi;
                        double sin_phi = std::sin(phi);
                        double cos_phi = std::cos(phi);
                        
                        // Precompute trig functions to avoid redundant calculations
                        Eigen::Vector3d r(
                            sin_theta * cos_phi,
                            sin_theta * sin_phi,
                            cos_theta
                        );
                        
                        // Store in thread-local vectors
                        local_directions.push_back(r.normalized());
                        local_dOmega.push_back(solid_angle);
                    }
                }
                
                // Merge thread-local results into global vectors
                #pragma omp critical
                {
                    precomputed_directions.insert(precomputed_directions.end(), 
                                                local_directions.begin(), 
                                                local_directions.end());
                    precomputed_dOmega.insert(precomputed_dOmega.end(), 
                                           local_dOmega.begin(), 
                                           local_dOmega.end());
                }
            }
            
            std::cout << "Direction precomputation complete. Total directions: " 
                      << precomputed_directions.size() << std::endl;
        }
    
        double computeEtchingRate(int idx, const Point_3& pos, const Eigen::Vector3d& normal, double sigma) {
            double total_F = 0.0;
            const size_t num_samples = precomputed_directions.size();
            
            // Optimize computation with vectorization hints
            #pragma omp parallel for reduction(+:total_F) schedule(static, 16)
            for (size_t s = 0; s < num_samples; ++s) {
                const auto& r = precomputed_directions[s];
                double dot = r.dot(normal);
                    
                if (dot <= 0) continue;
                    
                // Precompute expensive operations
                double theta = std::acos(dot);
                double sigma_squared = sigma * sigma;
                double exp_term = std::exp(-theta/(2.0 * sigma_squared));
                    
                // Accumulate contribution
                total_F += dot * exp_term * precomputed_dOmega[s];
            }
            
            return total_F;
        }
    
    
    void updateNarrowBand() {
        // Reserve memory to avoid reallocations
        narrowBand.clear();
        narrowBand.reserve(grid.size() / 8); // Typical narrow band is a small fraction of total grid
        
        // Use a more efficient approach with spatial locality
        #pragma omp parallel
        {
            std::vector<int> localBand;
            localBand.reserve(grid.size() / (8 * omp_get_num_threads()));
            
            #pragma omp for nowait
            for (size_t i = 0; i < grid.size(); ++i) {
                if (!isOnBoundary(i) && std::abs(phi[i]) <= NARROW_BAND_WIDTH * GRID_SPACING) {
                    localBand.push_back(i);
                }
            }
            
            #pragma omp critical
            {
                narrowBand.insert(narrowBand.end(), localBand.begin(), localBand.end());
            }
        }
        
        // Sort for better cache locality during evolution
        std::sort(narrowBand.begin(), narrowBand.end());
>>>>>>> Stashed changes
        
        std::cout << "Narrow band updated. Size: " << narrowBand.size() 
                  << " (" << (narrowBand.size() * 100.0 / grid.size()) << "% of grid)" << std::endl;
    }

    // Morton encoding (Z-order curve) functions for cache-oblivious traversal
    inline uint64_t expandBits(uint32_t v) const {
        uint64_t x = v & 0x1fffff; // 21 bits (enough for grids up to 2^7 = 128^3)
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
    }
    
    inline uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) const {
        return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
    }
    
    inline void mortonDecode(uint64_t code, uint32_t& x, uint32_t& y, uint32_t& z) const {
        x = y = z = 0;
        for (uint32_t i = 0; i < 21; ++i) { // 21 bits per dimension
            x |= ((code >> (3*i)) & 1) << i;
            y |= ((code >> (3*i+1)) & 1) << i;
            z |= ((code >> (3*i+2)) & 1) << i;
        }
    }
    
    void generateGrid() {
        if (mesh.is_empty()) {
            throw std::runtime_error("Mesh not loaded - cannot generate grid");
        }
        
        std::cout << "Generating grid with size " << GRID_SIZE << "x" << GRID_SIZE << "x" << GRID_SIZE << "..." << std::endl;
        
        // Calculate bounding box with optimized padding
        CGAL::Bbox_3 bbox = calculateBoundingBox();
        double dx = bbox.xmax() - bbox.xmin();
        double dy = bbox.ymax() - bbox.ymin();
        double dz = bbox.zmax() - bbox.zmin();
        
        // Add adaptive padding based on mesh size
        double max_dim = std::max({dx, dy, dz});
        double padding = 0.1 * max_dim;
        
        // Calculate grid boundaries
        double xmin = bbox.xmin() - padding;
        double xmax = bbox.xmax() + padding;
        double ymin = bbox.ymin() - padding;
        double ymax = bbox.ymax() + padding;
        double zmin = bbox.zmin() - padding;
        double zmax = bbox.zmax() + padding;
        
        // Calculate grid spacing based on largest dimension
        BOX_SIZE = std::max({xmax-xmin, ymax-ymin, zmax-zmin});
        GRID_SPACING = BOX_SIZE / (GRID_SIZE - 1);
        
<<<<<<< Updated upstream
        // Clear and reserve space for grid points and morton codes
        grid.clear();
        grid.reserve(GRID_SIZE * GRID_SIZE * GRID_SIZE);
        mortonCodes.clear();
        mortonCodes.reserve(GRID_SIZE * GRID_SIZE * GRID_SIZE);
        mortonToIndex.clear();
        
        // Generate grid points with Z-order curve (Morton ordering)
        std::vector<std::pair<uint64_t, size_t>> pointsWithMorton;
        pointsWithMorton.reserve(GRID_SIZE * GRID_SIZE * GRID_SIZE);
        
        // First generate all points and their Morton codes
        size_t idx = 0;
        for (int z = 0; z < GRID_SIZE; ++z) {
            double pz = zmin + z * GRID_SPACING;
            for (int y = 0; y < GRID_SIZE; ++y) {
                double py = ymin + y * GRID_SPACING;
                for (int x = 0; x < GRID_SIZE; ++x) {
                    double px = xmin + x * GRID_SPACING;
                    grid.emplace_back(px, py, pz);
                    
                    // Calculate Morton code for this point
                    uint64_t mortonCode = mortonEncode(x, y, z);
                    mortonCodes.push_back(mortonCode);
                    mortonToIndex[mortonCode] = idx;
                    pointsWithMorton.emplace_back(mortonCode, idx);
                    idx++;
=======
        // Store grid origin for faster lookups
        gridOriginX = xmin;
        gridOriginY = ymin;
        gridOriginZ = zmin;
        
        // Preallocate memory for grid points
        const size_t total_points = static_cast<size_t>(GRID_SIZE) * GRID_SIZE * GRID_SIZE;
        grid.clear();
        grid.reserve(total_points);
        
        // Generate grid points in parallel for better performance
        #pragma omp parallel
        {
            // Thread-local storage for grid points
            std::vector<Point_3> local_grid;
            local_grid.reserve(total_points / omp_get_num_threads());
            
            #pragma omp for schedule(dynamic, 8) collapse(2) nowait
            for (int z = 0; z < GRID_SIZE; ++z) {
                for (int y = 0; y < GRID_SIZE; ++y) {
                    // Precompute z and y coordinates
                    double pz = zmin + z * GRID_SPACING;
                    double py = ymin + y * GRID_SPACING;
                    
                    // Generate points for this y-z slice
                    for (int x = 0; x < GRID_SIZE; ++x) {
                        double px = xmin + x * GRID_SPACING;
                        local_grid.emplace_back(px, py, pz);
                    }
>>>>>>> Stashed changes
                }
            }
            
            // Merge thread-local results into global grid
            #pragma omp critical
            {
                grid.insert(grid.end(), local_grid.begin(), local_grid.end());
            }
        }
        
<<<<<<< Updated upstream
        // Sort grid points by Morton code for cache-oblivious traversal
        // This is optional as we keep the original grid ordering for compatibility
        // but we'll use the sorted order for traversal in the evolution methods
        std::sort(pointsWithMorton.begin(), pointsWithMorton.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
    }
    

    // Process a block of grid points for SDF initialization
    void processSdfBlock(const std::vector<size_t>& blockIndices, Eigen::VectorXd& sdf, 
                        const CGAL::Side_of_triangle_mesh<Mesh, Kernel>& inside) {
        for (const auto& i : blockIndices) {
            // Compute squared distance to the mesh
=======
        std::cout << "Grid generation complete. Total points: " << grid.size() << std::endl;
    }
    

    Eigen::VectorXd initializeSignedDistanceField() {
        if (!tree) {
            throw std::runtime_error("AABB tree not initialized. Load a mesh first.");
        }
        
        std::cout << "Initializing signed distance field..." << std::endl;
        
        // Pre-allocate memory for the signed distance field
        const size_t grid_size = grid.size();
        Eigen::VectorXd sdf(grid_size);
        
        // Create inside/outside classifier once (thread-safe in CGAL)
        CGAL::Side_of_triangle_mesh<Mesh, Kernel> inside(mesh);
        
        // Use dynamic scheduling for better load balancing
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t i = 0; i < grid_size; ++i) {
            // Compute squared distance to the mesh using AABB tree
>>>>>>> Stashed changes
            auto closest = tree->closest_point_and_primitive(grid[i]);
            double sq_dist = CGAL::sqrt(CGAL::squared_distance(grid[i], closest.first));
            
            // Determine if point is inside or outside the mesh
            CGAL::Bounded_side res = inside(grid[i]);
            
            // Set signed distance based on inside/outside classification
            if (res == CGAL::ON_BOUNDED_SIDE) {
                sdf[i] = -sq_dist; // Inside (negative)
            } else if (res == CGAL::ON_BOUNDARY) {
                sdf[i] = 0.0;      // On boundary
            } else { 
                sdf[i] = sq_dist;   // Outside (positive)
            }
<<<<<<< Updated upstream
        }
    }
    
    // Recursive subdivision for cache-oblivious SDF initialization
    void recursiveSdfSubdivision(const std::vector<size_t>& indices, size_t start, size_t end, 
                               Eigen::VectorXd& sdf, const CGAL::Side_of_triangle_mesh<Mesh, Kernel>& inside) {
        // Base case: small enough block to process directly
        const size_t BLOCK_SIZE = 64; // Adjust based on cache size
        if (end - start <= BLOCK_SIZE) {
            std::vector<size_t> blockIndices(indices.begin() + start, indices.begin() + end);
            processSdfBlock(blockIndices, sdf, inside);
            return;
        }
        
        // Recursive case: divide and conquer
        size_t mid = start + (end - start) / 2;
        
        #pragma omp task shared(sdf) if(end - start > 1000)
        recursiveSdfSubdivision(indices, start, mid, sdf, inside);
        
        #pragma omp task shared(sdf) if(end - start > 1000)
        recursiveSdfSubdivision(indices, mid, end, sdf, inside);
        
        #pragma omp taskwait
    }

    Eigen::VectorXd initializeSignedDistanceField() {
        if (!tree) {
            throw std::runtime_error("AABB tree not initialized. Load a mesh first.");
        }
        
        Eigen::VectorXd sdf(grid.size());
        CGAL::Side_of_triangle_mesh<Mesh, Kernel> inside(mesh);

        // Create a vector of indices sorted by Morton code for cache-oblivious traversal
        std::vector<size_t> sortedIndices;
        sortedIndices.reserve(grid.size());
        
        // Create pairs of (morton code, grid index)
        std::vector<std::pair<uint64_t, size_t>> pointsWithMorton;
        pointsWithMorton.reserve(grid.size());
        
        for (size_t i = 0; i < grid.size(); ++i) {
            pointsWithMorton.emplace_back(mortonCodes[i], i);
        }
        
        // Sort by Morton code
        std::sort(pointsWithMorton.begin(), pointsWithMorton.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Extract sorted indices
        for (const auto& point : pointsWithMorton) {
            sortedIndices.push_back(point.second);
        }
        
        // Process grid points using cache-oblivious recursive subdivision
        #pragma omp parallel
        {
            #pragma omp single
            recursiveSdfSubdivision(sortedIndices, 0, sortedIndices.size(), sdf, inside);
=======
            
            // Progress reporting for long computations
            if (i % (grid_size / 10) == 0 && omp_get_thread_num() == 0) {
                #pragma omp critical
                {
                    std::cout << "SDF initialization: " << (i * 100.0 / grid_size) << "% complete" << std::endl;
                }
            }
>>>>>>> Stashed changes
        }
        
        std::cout << "Signed distance field initialization complete." << std::endl;
        return sdf;
    }
    

    inline bool isOnBoundary(int idx) const {
        // Fast boundary check using grid coordinates
        // Extract coordinates with bit operations where possible for better performance
        const int x = idx % GRID_SIZE;
        const int y = (idx / GRID_SIZE) % GRID_SIZE;
        const int z = idx / (GRID_SIZE * GRID_SIZE);
        
        // Check if any coordinate is on the boundary of the grid
        // Using bitwise OR for potentially faster evaluation
        return (x == 0) | (x == GRID_SIZE - 1) | 
               (y == 0) | (y == GRID_SIZE - 1) | 
               (z == 0) | (z == GRID_SIZE - 1);
    }
    

    inline int getIndex(int x, int y, int z) const {
        // Fast boundary check with branch prediction hints
        if (__builtin_expect(x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE && z >= 0 && z < GRID_SIZE, 1)) {
            // Most common case - point is within bounds
            return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
        } else {
            // Boundary handling for out-of-bounds access
            x = std::max(0, std::min(x, GRID_SIZE - 1));
            y = std::max(0, std::min(y, GRID_SIZE - 1));
            z = std::max(0, std::min(z, GRID_SIZE - 1));
            return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
        }
    }
    
    

};


bool LevelSetMethod::extractSurfaceMeshCGAL(const std::string& filename) {
    try {
        if (phi.size() != grid.size()) {
            throw std::runtime_error("Level set function not initialized.");
        }
        
        std::cout << "Preparing to extract surface mesh..." << std::endl;
 
        typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
        typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
        typedef Tr::Geom_traits GT;
        typedef GT::Sphere_3 Sphere_3;
        typedef GT::FT FT;
        typedef std::function<FT(typename GT::Point_3)> Function;
        typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;
    
<<<<<<< Updated upstream
        // Define the implicit function for the zero level set with cache-oblivious optimization
=======
        // Define the implicit function for the zero level set with optimized implementation
>>>>>>> Stashed changes
        class LevelSetImplicitFunction {
        private:
            const std::vector<Point_3>& grid;
            const Eigen::VectorXd& phi;
            const int GRID_SIZE;
            const double GRID_SPACING;
            const double gridOriginX, gridOriginY, gridOriginZ;
            const std::unordered_map<uint64_t, size_t>& mortonToIndex; // For fast spatial lookups
            
            // Cache for recently accessed values to improve locality
            mutable std::unordered_map<uint64_t, FT> valueCache;
            mutable size_t cacheHits = 0;
            mutable size_t cacheMisses = 0;
            const size_t MAX_CACHE_SIZE = 4096; // Adjust based on expected usage pattern
            
            // Cache for frequently accessed values
            mutable std::unordered_map<size_t, double> valueCache;
            mutable std::mutex cacheMutex;
            
            // Hash function for 3D coordinates
            size_t hashCoords(int x, int y, int z) const {
                // Simple hash function for 3D coordinates
                return (static_cast<size_t>(x) * 73856093) ^ 
                       (static_cast<size_t>(y) * 19349663) ^ 
                       (static_cast<size_t>(z) * 83492791);
            }
            
        public:
            LevelSetImplicitFunction(const std::vector<Point_3>& grid, const Eigen::VectorXd& phi, 
                                    int gridSize, double gridSpacing,
                                    const std::unordered_map<uint64_t, size_t>& mortonToIndex)
                : grid(grid), phi(phi), GRID_SIZE(gridSize), GRID_SPACING(gridSpacing),
<<<<<<< Updated upstream
                  gridOriginX(grid[0].x()), gridOriginY(grid[0].y()), gridOriginZ(grid[0].z()),
                  mortonToIndex(mortonToIndex) {
            }
            
            ~LevelSetImplicitFunction() {
                // Report cache statistics
                if (cacheHits + cacheMisses > 0) {
                    double hitRate = static_cast<double>(cacheHits) / (cacheHits + cacheMisses) * 100.0;
                    std::cout << "Cache statistics: " << cacheHits << " hits, " 
                              << cacheMisses << " misses (" << hitRate << "% hit rate)" << std::endl;
                }
            }
                
            FT operator()(const Point_3& p) const {
=======
                  gridOriginX(grid[0].x()), gridOriginY(grid[0].y()), gridOriginZ(grid[0].z()) {
                // Reserve space for cache to avoid rehashing
                valueCache.reserve(10000);
            }
                
            FT operator()(const Point_3& p) const {
                // Fast grid-based lookup with caching for frequently accessed points
                
>>>>>>> Stashed changes
                // Calculate grid indices based on point position
                int x = std::round((p.x() - gridOriginX) / GRID_SPACING);
                int y = std::round((p.y() - gridOriginY) / GRID_SPACING);
                int z = std::round((p.z() - gridOriginZ) / GRID_SPACING);
                
                // Check cache first for exact grid points
                size_t hash = hashCoords(x, y, z);
                {
                    std::lock_guard<std::mutex> lock(cacheMutex);
                    auto it = valueCache.find(hash);
                    if (it != valueCache.end()) {
                        return it->second;
                    }
                }
                
                // Clamp to grid boundaries
                x = std::max(0, std::min(x, GRID_SIZE - 1));
                y = std::max(0, std::min(y, GRID_SIZE - 1));
                z = std::max(0, std::min(z, GRID_SIZE - 1));
                
                // Use Morton code for better spatial locality
                uint64_t mortonCode = expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
                
<<<<<<< Updated upstream
                // Check cache first
                auto cacheIt = valueCache.find(mortonCode);
                if (cacheIt != valueCache.end()) {
                    cacheHits++;
                    return cacheIt->second;
                }
                cacheMisses++;
                
                // Try to find the exact point using morton code lookup
                auto indexIt = mortonToIndex.find(mortonCode);
                if (indexIt != mortonToIndex.end()) {
                    size_t idx = indexIt->second;
                    // Cache the result
                    if (valueCache.size() >= MAX_CACHE_SIZE) {
                        valueCache.clear(); // Simple eviction policy: clear all when full
                    }
                    valueCache[mortonCode] = phi[idx];
                    return phi[idx];
                }
                
                // Fallback to trilinear interpolation for points not exactly on the grid
=======
                // Bounds check for direct lookup
                if (idx >= 0 && idx < static_cast<int>(phi.size())) {
                    double value = phi[idx];
                    // Cache the result for future lookups
                    {
                        std::lock_guard<std::mutex> lock(cacheMutex);
                        // Limit cache size to prevent memory issues
                        if (valueCache.size() < 10000) {
                            valueCache[hash] = value;
                        }
                    }
                    return value;
                }
                
                // Fallback to trilinear interpolation for points outside the grid
>>>>>>> Stashed changes
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
                
                // Get the eight corners of the cell using Morton codes for better locality
                uint64_t code000 = expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
                uint64_t code001 = expandBits(x) | (expandBits(y) << 1) | (expandBits(z+1) << 2);
                uint64_t code010 = expandBits(x) | (expandBits(y+1) << 1) | (expandBits(z) << 2);
                uint64_t code011 = expandBits(x) | (expandBits(y+1) << 1) | (expandBits(z+1) << 2);
                uint64_t code100 = expandBits(x+1) | (expandBits(y) << 1) | (expandBits(z) << 2);
                uint64_t code101 = expandBits(x+1) | (expandBits(y) << 1) | (expandBits(z+1) << 2);
                uint64_t code110 = expandBits(x+1) | (expandBits(y+1) << 1) | (expandBits(z) << 2);
                uint64_t code111 = expandBits(x+1) | (expandBits(y+1) << 1) | (expandBits(z+1) << 2);
                
                // Get indices from Morton codes
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
                
<<<<<<< Updated upstream
                // Interpolate along z
                FT result = v0 * (1 - fz) + v1 * fz;
                
                // Cache the interpolated result
                if (valueCache.size() >= MAX_CACHE_SIZE) {
                    valueCache.clear(); // Simple eviction policy: clear all when full
                }
                valueCache[mortonCode] = result;
                
                return result;
            }
            
            // Helper function to expand bits for Morton code
            inline uint64_t expandBits(uint32_t v) const {
                uint64_t x = v & 0x1fffff; // 21 bits (enough for grids up to 2^7 = 128^3)
                x = (x | x << 32) & 0x1f00000000ffff;
                x = (x | x << 16) & 0x1f0000ff0000ff;
                x = (x | x << 8) & 0x100f00f00f00f00f;
                x = (x | x << 4) & 0x10c30c30c30c30c3;
                x = (x | x << 2) & 0x1249249249249249;
                return x;
=======
                // Interpolate along z and return final value
                double result = v0 * (1 - fz) + v1 * fz;
                return result;
>>>>>>> Stashed changes
            }
        };

        // Create the implicit function with grid parameters and morton code mapping
        LevelSetImplicitFunction implicitFunction(grid, phi, GRID_SIZE, GRID_SPACING, mortonToIndex);

        // Wrap the implicit function with the corrected type
        Function function = [&implicitFunction](const GT::Point_3& p) {
            return implicitFunction(Point_3(p.x(), p.y(), p.z()));
        };
        
        // Create triangulation with optimized parameters
        Tr tr;
        C2t3 c2t3(tr);
        
        // Adjust bounding sphere to better match the data
        double boundingSphereRadius = BOX_SIZE * 0.6; // Slightly smaller for better focus
        Surface_3 surface(function, 
                         Sphere_3(CGAL::ORIGIN, boundingSphereRadius*boundingSphereRadius), 
                         1e-6); // Increased precision
        
        // Adjust mesh criteria for better performance/quality tradeoff
        typedef CGAL::Surface_mesh_default_criteria_3<Tr> Criteria;
        // Adjust angle bound, radius bound and distance bound for better quality
        Criteria criteria(25.0, GRID_SPACING * 2.5, GRID_SPACING * 2.0);
        
        // Define the mesh data structure
        typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
        Surface_mesh surface_mesh;
        
        std::cout << "Starting surface mesh generation..." << std::endl;
        
        // Generate the surface mesh with progress reporting
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Generate the surface mesh
        CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        std::cout << "Surface mesh generation completed in " << duration << " seconds." << std::endl;
        std::cout << "Triangulation has " << tr.number_of_vertices() << " vertices." << std::endl;
        
        // Convert the complex to a surface mesh
        std::cout << "Converting to surface mesh..." << std::endl;
        CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, surface_mesh);
        
<<<<<<< Updated upstream
        // Save the surface mesh to a file
=======
        // Save the surface mesh to a file with high precision
        std::cout << "Saving mesh to file..." << std::endl;
>>>>>>> Stashed changes
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

#endif