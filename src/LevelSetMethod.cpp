#include <omp.h>
#include <iostream>
#include <cmath>

#include "LevelSetMethod.hpp"

// Constructor implementation
LevelSetMethod::LevelSetMethod(const std::string& filename,
                             int gridSize, 
                             double timeStep, 
                             int maxSteps, 
                             int reinitInterval,
                             double narrowBandWidth)
    : GRID_SIZE(gridSize),
      GRID_SPACING(0.0),  // Will be calculated after loading mesh
      dt(timeStep),
      STEPS(maxSteps),
      REINIT_INTERVAL(reinitInterval),
      NARROW_BAND_WIDTH(narrowBandWidth) {
    
    // Load the mesh
    loadMesh(filename);
    
    // Generate the grid
    generateGrid();
    
    // Initialize the signed distance field
    phi = initializeSignedDistanceField();
    
    // Initialize the narrow band
    updateNarrowBand();
}

CGAL::Bbox_3 LevelSetMethod::calculateBoundingBox() const {
    return CGAL::bbox_3(mesh.points().begin(), mesh.points().end());
}

bool LevelSetMethod::extractSurfaceMeshCGAL(const std::string& filename) {
    try {
        // Create a mesh from the level set
        Mesh outputMesh;
        
        // Implementation of surface extraction from level set
        // This would convert the phi values back to a surface mesh
        
        // Save the mesh
        if (!CGAL::IO::write_polygon_mesh(filename, outputMesh, 
                                       CGAL::parameters::stream_precision(17))) {
            std::cerr << "Failed to write surface mesh to file." << std::endl;
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error extracting surface mesh: " << e.what() << std::endl;
        return false;
    }
}

void LevelSetMethod::loadMesh(const std::string& filename) {
    // Load the mesh using CGAL
    if (!CGAL::IO::read_polygon_mesh(filename, mesh) || 
        is_empty(mesh) || 
        !is_triangle_mesh(mesh)) {
        throw std::runtime_error("Failed to load mesh or mesh is not a triangle mesh.");
    }
    
    // Calculate the bounding box
    CGAL::Bbox_3 bbox = calculateBoundingBox();
    
    // Calculate the box size based on the bounding box
    double xSize = bbox.xmax() - bbox.xmin();
    double ySize = bbox.ymax() - bbox.ymin();
    double zSize = bbox.zmax() - bbox.zmin();
    BOX_SIZE = std::max({xSize, ySize, zSize}) * 1.1; // Add 10% margin
    
    // Calculate the grid spacing
    GRID_SPACING = BOX_SIZE / GRID_SIZE;
    
    // Set the grid origin (bottom-left-front corner of the bounding box)
    gridOriginX = bbox.xmin() - (BOX_SIZE - xSize) / 2.0;
    gridOriginY = bbox.ymin() - (BOX_SIZE - ySize) / 2.0;
    gridOriginZ = bbox.zmin() - (BOX_SIZE - zSize) / 2.0;
    
    // Build the AABB tree for fast distance queries
    tree = std::make_unique<AABB_tree>(faces(mesh).begin(), faces(mesh).end(), mesh);
    tree->accelerate_distance_queries();
}

bool LevelSetMethod::evolve() {
    try {
        // Main evolution loop
        for (int step = 0; step < STEPS; ++step) {
            // Perform level set evolution for one time step
            
            // For each point in the narrow band
            #pragma omp parallel for
            for (size_t i = 0; i < narrowBand.size(); ++i) {
                int idx = narrowBand[i];
                
                // Skip points that are too far from the interface
                if (std::abs(phi[idx]) > NARROW_BAND_WIDTH) {
                    continue;
                }
                
                // Calculate the etching rate based on the current level set
                double rate = CalculateEtchingRate(14.0); // sigma = 14.0
                
                // Update the level set value
                phi[idx] += rate * dt;
            }
            
            // Reinitialize the signed distance field periodically
            if (step % REINIT_INTERVAL == 0) {
                reinitialize();
                updateNarrowBand();
            }
            
            // Print progress
            if (step % 10 == 0) {
                std::cout << "Step " << step << " completed." << std::endl;
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during evolution: " << e.what() << std::endl;
        return false;
    }
}

void LevelSetMethod::reinitialize() {
    // Reinitialize the level set to maintain the signed distance property
    // This is typically done using a PDE-based approach or by recalculating distances
    
    // For simplicity, we'll just recalculate the signed distance field
    phi = initializeSignedDistanceField();
}

double LevelSetMethod::CalculateEtchingRate(double sigma) {
    // Calculate the etching rate based on the current level set
    // This would typically depend on the local geometry (e.g., curvature, normal direction)
    
    // For a simple model, we can use a constant rate
    return 1.0;
}

void LevelSetMethod::updateNarrowBand() {
    // Update the narrow band (the set of grid points near the interface)
    narrowBand.clear();
    
    // Add all points within the narrow band width to the narrow band
    for (int i = 0; i < phi.size(); ++i) {
        if (std::abs(phi[i]) < NARROW_BAND_WIDTH) {
            narrowBand.push_back(i);
        }
    }
}

void LevelSetMethod::generateGrid() {
    // Generate a uniform grid covering the bounding box
    grid.clear();
    grid.reserve(GRID_SIZE * GRID_SIZE * GRID_SIZE);
    
    for (int z = 0; z < GRID_SIZE; ++z) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                double xPos = gridOriginX + x * GRID_SPACING;
                double yPos = gridOriginY + y * GRID_SPACING;
                double zPos = gridOriginZ + z * GRID_SPACING;
                grid.emplace_back(xPos, yPos, zPos);
            }
        }
    }
}

Eigen::VectorXd LevelSetMethod::initializeSignedDistanceField() {
    // Initialize the signed distance field
    Eigen::VectorXd phi(grid.size());
    
    // Use CGAL's Side_of_triangle_mesh to determine inside/outside
    CGAL::Side_of_triangle_mesh<Mesh, Kernel> inside_tester(mesh);
    
    // Calculate the signed distance for each grid point
    #pragma omp parallel for
    for (size_t i = 0; i < grid.size(); ++i) {
        // Calculate the unsigned distance
        double distance = std::sqrt(tree->squared_distance(grid[i]));
        
        // Determine the sign based on whether the point is inside or outside
        CGAL::Bounded_side side = inside_tester(grid[i]);
        
        // Set the signed distance
        phi[i] = (side == CGAL::ON_BOUNDED_SIDE) ? -distance : distance;
    }
    
    return phi;
}

bool LevelSetMethod::isOnBoundary(int idx) const {
    static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
        
    const int x = idx % GRID_SIZE;
    const int y = (idx / GRID_SIZE) % GRID_SIZE;
    const int z = idx / GRID_SIZE_SQ;
        
    const bool x_boundary = (x == 0) || (x == GRID_SIZE - 1);
    const bool y_boundary = (y == 0) || (y == GRID_SIZE - 1);
    const bool z_boundary = (z == 0) || (z == GRID_SIZE - 1);
        
    return x_boundary || y_boundary || z_boundary;
}

int LevelSetMethod::getIndex(int x, int y, int z) const {
    // Convert 3D coordinates to linear index
    return z * GRID_SIZE * GRID_SIZE + y * GRID_SIZE + x;
}

// Main function and OpenMP test function preserved from original file
void testOpenMP() {
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
    
    #pragma omp parallel
    {
        #pragma omp critical
        std::cout << "Hello from thread " << omp_get_thread_num() 
                  << " of " << omp_get_num_threads() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string inputFile = "initial_struct_600_600.obj";
    std::string outputFile = "final_sdf.csv";
    std::string surfaceFile = "result.obj";
    
    testOpenMP();
       
    LevelSetMethod levelSet(inputFile, 400, 0.01, 10000, 10);

    // Run the level set evolution
    std::cout << "Running level set evolution..." << std::endl;
    if (!levelSet.evolve()) {
        std::cerr << "Evolution failed. Exiting." << std::endl;
        return 1;
    }
        
    std::cout << "Saving surface mesh to " << surfaceFile << "..." << std::endl;
    if (!levelSet.extractSurfaceMeshCGAL(surfaceFile)) {
        std::cerr << "Failed to save surface mesh. Exiting." << std::endl;
        return 1;
    }

    std::cout << "Level set method completed successfully." << std::endl;
    return 0;
}

