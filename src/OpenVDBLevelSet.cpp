#include "OpenVDBLevelSet.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Default parameters
    std::string inputFile = "initial_struct_600_600.obj";
    std::string outputFile = "etched_result.obj";
    double voxelSize = 2.0;        // Size of voxels in the grid
    double timeStep = 0.5;         // Time step for evolution
    int maxSteps = 100;           // Number of evolution steps
    double narrowBandWidth = 3.0;  // Width of narrow band in voxel units
    
    // Parse command line arguments if provided
    if (argc > 1) inputFile = argv[1];
    if (argc > 2) outputFile = argv[2];
    if (argc > 3) voxelSize = std::stod(argv[3]);
    if (argc > 4) maxSteps = std::stoi(argv[4]);
    
    // Initialize the OpenVDB level set method
    OpenVDBLevelSet levelSet(voxelSize, timeStep, maxSteps, narrowBandWidth);
    
    // Load the input mesh
    std::cout << "Loading mesh from " << inputFile << "..." << std::endl;
    if (!levelSet.loadMesh(inputFile)) {
        std::cerr << "Failed to load mesh. Exiting." << std::endl;
        return 1;
    }
    
    // Run the level set evolution (etching simulation)
    std::cout << "Running etching simulation..." << std::endl;
    if (!levelSet.evolve()) {
        std::cerr << "Evolution failed. Exiting." << std::endl;
        return 1;
    }
    
    // Extract and save the resulting surface mesh
    std::cout << "Saving etched surface mesh to " << outputFile << "..." << std::endl;
    if (!levelSet.extractSurfaceMesh(outputFile)) {
        std::cerr << "Failed to save surface mesh. Exiting." << std::endl;
        return 1;
    }
    
    std::cout << "Etching simulation completed successfully." << std::endl;
    return 0;
}