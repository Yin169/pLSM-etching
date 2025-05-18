// Example usage of the Cache-Oblivious Level Set Method
#include "LevelSetMethodCacheOblivious.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <mesh_file> <org_file> <material_csv_file>" << std::endl;
        return 1;
    }
    
    const std::string meshFile = argv[1];
    const std::string orgFile = argv[2];
    const std::string materialCsvFile = argv[3];
    
    // Parameters for the level set method
    const int gridSize = 100;
    const double timeStep = 0.01;
    const int maxSteps = 80;
    const int reinitInterval = 5;
    const int narrowBandInterval = 100;
    const double narrowBandWidth = 10.0;
    const int numThreads = -1; // Use all available threads
    const double curvatureWeight = 0.0;
    
    try {
        std::cout << "Initializing Cache-Oblivious Level Set Method..." << std::endl;
        
        // Create an instance of the cache-oblivious level set method
        LevelSetMethodCacheOblivious levelSet(
            meshFile,
            orgFile,
            materialCsvFile,
            gridSize,
            timeStep,
            maxSteps,
            reinitInterval,
            narrowBandInterval,
            narrowBandWidth,
            numThreads,
            curvatureWeight,
            SpatialSchemeType::UPWIND,  // Use WENO for better accuracy
            TimeSchemeType::FORWARD_EULER  // Use RK3 for better stability
        );
        
        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();
        
        // Run the evolution
        std::cout << "Starting level set evolution with cache-oblivious algorithm..." << std::endl;
        bool success = levelSet.evolve();
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        if (success) {
            std::cout << "Evolution completed successfully in " << elapsed.count() << " seconds." << std::endl;
            
            levelSet.extractSurfaceMeshCGAL("output_mesh.obj");
            
            return 0;
        } else {
            std::cerr << "Evolution failed." << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}