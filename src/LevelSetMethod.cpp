#include "LevelSetMethod.hpp"



int main(int argc, char* argv[]) {
    std::string inputFile = "initial_struct_600_600.obj";
    std::string outputFile = "final_sdf.csv";
    std::string surfaceFile = "result.obj";
        
        
    LevelSetMethod levelSet(inputFile, 200, 0.001, 1000, 100);

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

