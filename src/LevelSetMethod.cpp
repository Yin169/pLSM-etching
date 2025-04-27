#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace PMP = CGAL::Polygon_mesh_processing;


typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;


class LevelSetMethod {
public:

    LevelSetMethod(int gridSize = 30, double boxSize = 1400.0, 
                  double timeStep = 0.01, int maxSteps = 50, 
                  int reinitInterval = 5)
        : GRID_SIZE(gridSize), 
          BOX_SIZE(boxSize),
          GRID_SPACING(boxSize / (gridSize - 1)),
          dt(timeStep),
          STEPS(maxSteps),
          REINIT_INTERVAL(reinitInterval) {
        
        // Generate the computational grid
        generateGrid();
    }


    bool loadMesh(const std::string& filename) {
        try {
            if (!PMP::IO::read_polygon_mesh(filename, mesh) || is_empty(mesh) || !is_triangle_mesh(mesh)) {
                std::cerr << "Error: Could not open file " << filename << std::endl;
                return false;
            }
            
            // Build AABB tree for efficient distance queries
            tree = std::make_unique<AABB_tree>(faces(mesh).first, faces(mesh).second, mesh);
            tree->build();
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading mesh: " << e.what() << std::endl;
            return false;
        }
    }


    bool evolve() {
        try {
            // Initialize the signed distance field
            phi = initializeSignedDistanceField();
            
            // Main evolution loop
            for (int step = 0; step < STEPS; ++step) {
                // Create a copy of the current level set
                Eigen::VectorXd newPhi = phi;
                
                
                phi = newPhi;
                

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
    // Configuration parameters
    const int GRID_SIZE;
    const double BOX_SIZE;
    const double GRID_SPACING;
    const double dt;
    const int STEPS;
    const int REINIT_INTERVAL;
    
    // Data structures
    Mesh mesh;
    std::unique_ptr<AABB_tree> tree;
    std::vector<Point_3> grid;
    Eigen::VectorXd phi;
    

    void generateGrid() {
        grid.clear();
        grid.reserve(GRID_SIZE * GRID_SIZE * GRID_SIZE);
        
        double halfBox = BOX_SIZE / 2.0;
        for (int x = 0; x < GRID_SIZE; ++x) {
            for (int y = 0; y < GRID_SIZE; ++y) {
                for (int z = 0; z < GRID_SIZE; ++z) {
                    double px = -halfBox + x * GRID_SPACING;
                    double py = -halfBox + y * GRID_SPACING;
                    double pz = -halfBox + z * GRID_SPACING;
                    grid.emplace_back(px, py, pz);
                }
            }
        }
    }
    

    Eigen::VectorXd initializeSignedDistanceField() {
        if (!tree) {
            throw std::runtime_error("AABB tree not initialized. Load a mesh first.");
        }
        
        Eigen::VectorXd sdf(grid.size());
        CGAL::Side_of_triangle_mesh<Mesh, Kernel> inside(mesh);

        for (size_t i = 0; i < grid.size(); ++i) {
            // Compute squared distance to the mesh
            auto closest = tree->closest_point_and_primitive(grid[i]);
            double sq_dist = CGAL::sqrt(CGAL::squared_distance(grid[i], closest.first));
            
            CGAL::Bounded_side res = inside(grid[i]);
            
            // Set the signed distance (negative inside, positive outside)
            sdf[i] = (res == CGAL::ON_BOUNDED_SIDE) ? -std::sqrt(sq_dist) : std::sqrt(sq_dist);
        }
        
        return sdf;
    }
    

    bool isOnBoundary(int idx) const {
        int x = idx % GRID_SIZE;
        int y = (idx / GRID_SIZE) % GRID_SIZE;
        int z = idx / (GRID_SIZE * GRID_SIZE);
        
        return x == 0 || x == GRID_SIZE - 1 || 
               y == 0 || y == GRID_SIZE - 1 || 
               z == 0 || z == GRID_SIZE - 1;
    }
    

    int getIndex(int x, int y, int z) const {
        // Boundary check
        x = std::max(0, std::min(x, GRID_SIZE - 1));
        y = std::max(0, std::min(y, GRID_SIZE - 1));
        z = std::max(0, std::min(z, GRID_SIZE - 1));
        
        return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
    }
    
    

};


int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        std::string inputFile = "./data/initial_struct.obj";
        std::string outputFile = "final_sdf.csv";
        
        if (argc > 1) inputFile = argv[1];
        if (argc > 2) outputFile = argv[2];
        
        // Create level set method instance with default parameters
        LevelSetMethod levelSet;
        
        // Load the input mesh
        std::cout << "Loading mesh from " << inputFile << "..." << std::endl;
        if (!levelSet.loadMesh(inputFile)) {
            std::cerr << "Failed to load mesh. Exiting." << std::endl;
            return 1;
        }
        
        // Run the level set evolution
        std::cout << "Running level set evolution..." << std::endl;
        if (!levelSet.evolve()) {
            std::cerr << "Evolution failed. Exiting." << std::endl;
            return 1;
        }
        
        // Save the results
        std::cout << "Saving results to " << outputFile << "..." << std::endl;
        if (!levelSet.saveResult(outputFile)) {
            std::cerr << "Failed to save results. Exiting." << std::endl;
            return 1;
        }
        
        std::cout << "Level set method completed successfully." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}