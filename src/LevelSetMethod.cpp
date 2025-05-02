#include <omp.h>
#include <iostream>
#include <cmath>

#include "LevelSetMethod.hpp"



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
       
    LevelSetMethod levelSet(inputFile,80, 0.1, 2000, 77);

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

