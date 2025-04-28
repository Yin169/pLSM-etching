#include "LevelSetMethod.hpp"



int main(int argc, char* argv[]) {
    std::string inputFile = "initial_struct_600_600.obj";
    std::string outputFile = "final_sdf.csv";
    std::string surfaceFile = "result.obj";
        
        
    LevelSetMethod levelSet(200, 1400.0, 0.01, 2000, 5);
        
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
        
    // std::cout << "Saving results to " << outputFile << "..." << std::endl;
    // if (!levelSet.saveResult(outputFile)) {
    //     std::cerr << "Failed to save results. Exiting." << std::endl;
    //     return 1;
    // }
        
    std::cout << "Saving surface mesh to " << surfaceFile << "..." << std::endl;
    if (!levelSet.extractSurfaceMeshCGAL(surfaceFile)) {
        std::cerr << "Failed to save surface mesh. Exiting." << std::endl;
        return 1;
    }

    std::cout << "Level set method completed successfully." << std::endl;
    return 0;
}

