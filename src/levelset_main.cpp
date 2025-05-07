#include <omp.h>
#include <iostream>
#include <cmath>

#include "LevelSetMethod.hpp"
#include "LevelSetMethod.cpp"


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
    std::string surfaceFile = "result.obj";
    
    testOpenMP();
       
    LevelSetMethod levelSet(inputFile, 
        200,    // gridSize
        0.001,   // timeStep
        200,    // maxSteps
        5,      // reinitInterval
        100,    // narrowBandInterval
        10.0,   // narrowBandWidth
        -1,     // numThreads (auto)
        0.0,    // curvatureWeight 
        Eigen::Vector3d(-0.01, -0.01, -1.0),  // velocity
        SpatialSchemeType::UPWIND,
        TimeSchemeType::FORWARD_EULER);

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

