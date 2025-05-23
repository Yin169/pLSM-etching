#include <omp.h>
#include <iostream>
#include <cmath>

#include "convert.hpp"
#include "alphawrap.hpp"
#include "OBJToBNDConverter.hpp"

#include "LevelSetMethod.hpp"
#include "LevelSetMethod.cpp"

int main(int argc, char* argv[]) {

    const std::string inFile = argv[1];
    const std::string outputFile = argv[2];
    std::cout << "Input file: " << inFile << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;

    Convert(inFile);
    Wrapper("data/initial_struct.obj", 600, 600);

    std::string inputFile = "initial_struct_600_600.obj";
    std::string materialCsvFile = "data/initial_struct_test.csv";
    std::string orgFile = "data/initial_struct.obj";
    
    LevelSetMethod levelSet(
        inputFile,
        orgFile,
        materialCsvFile, 
        300,    // gridSize
        0.1,   // timeStep
        400,    // maxSteps
        10,      // reinitInterval
        20,    // narrowBandInterval
        100.0,   // narrowBandWidth
        1.0,  // curvatureThreshold
        -1,     // numThreads (auto)
        SpatialSchemeType::UPWIND,
        TimeSchemeType::FORWARD_EULER);

    // case 1
    levelSet.setMaterialProperties("Polymer", 0.1, 0.01);
    levelSet.setMaterialProperties( "SiO2_PECVD", 0.6, 0.01);
    levelSet.setMaterialProperties("Si_Amorph", 1.0, 0.01);
    std::string surfaceFile = "Silicon_etch.obj";
    std::string outputBNDfile = "Silicon_etch.bnd";
    surfaceFile = outputFile + surfaceFile;
    outputBNDfile = outputFile + outputBNDfile;

    std::cout << "Running level set evolution..." << std::endl;
    if (!levelSet.evolve()) {
        std::cerr << "Evolution failed. Exiting." << std::endl; return 1;
    }
    std::cout << "Saving surface mesh to " << surfaceFile << "..." << std::endl;
    if (!levelSet.extractSurfaceMeshCGAL(surfaceFile)) {
        std::cerr << "Failed to save surface mesh. Exiting." << std::endl;
        return 1;
    }

    int saveBND = ConvertOBJToBND(surfaceFile, outputBNDfile); 
    if (saveBND == 0) {
        std::cout << "Conversion completed successfully." << std::endl;
    } else {
        std::cerr << "Conversion failed." << std::endl;
    }

    // case 2
    levelSet.setMaterialProperties("Si3N4_LPCVD", 0.3, 0.01);
    levelSet.setMaterialProperties("Polymer", 1, 0.01);
    levelSet.setMaterialProperties("SiO2_PECVD", 0.6, 0.01);
    levelSet.setSTEPS(700);  // maxSteps
    surfaceFile = "Polymer_etch.obj";
    outputBNDfile = "Polymer_etch.bnd";
    surfaceFile = outputFile + surfaceFile;
    outputBNDfile = outputFile + outputBNDfile;

    std::cout << "Running level set evolution..." << std::endl;
    if (!levelSet.evolve()) {
        std::cerr << "Evolution failed. Exiting." << std::endl; return 1;
    }
    std::cout << "Saving surface mesh to " << surfaceFile << "..." << std::endl;
    if (!levelSet.extractSurfaceMeshCGAL(surfaceFile)) {
        std::cerr << "Failed to save surface mesh. Exiting." << std::endl;
        return 1;
    }

    saveBND = ConvertOBJToBND(surfaceFile, outputBNDfile); 
    if (saveBND == 0) {
        std::cout << "Conversion completed successfully." << std::endl;
    } else {
        std::cerr << "Conversion failed." << std::endl;
    }

    // case 3
    levelSet.clearMaterialProperties();
    levelSet.setMaterialProperties("Si3N4_LPCVD", 1, 0.01);
    levelSet.setMaterialProperties("Polymer", 0.1, 0.01);  
    levelSet.setMaterialProperties("SiO2_PECVD", 0.4, 0.01); 
    levelSet.setMaterialProperties("SiO2_Thermal", 0.7, 0.01);  
    levelSet.setMaterialProperties("Si_Amorph", 0.35, 0.01); 
    levelSet.setMaterialProperties("Si_Xtal", 0.35, 0.01);
    levelSet.setSTEPS(1300);  // maxSteps
    surfaceFile = "Nitride_etch.obj";
    outputBNDfile = "Nitride_etch.bnd"; 
    surfaceFile = outputFile + surfaceFile;
    outputBNDfile = outputFile + outputBNDfile;

    std::cout << "Running level set evolution..." << std::endl;
    if (!levelSet.evolve()) {
        std::cerr << "Evolution failed. Exiting." << std::endl; return 1;
    }
    std::cout << "Saving surface mesh to " << surfaceFile << "..." << std::endl;
    if (!levelSet.extractSurfaceMeshCGAL(surfaceFile)) {
        std::cerr << "Failed to save surface mesh. Exiting." << std::endl;
        return 1;
    }

    saveBND = ConvertOBJToBND(surfaceFile, outputBNDfile); 
    if (saveBND == 0) {
        std::cout << "Conversion completed successfully." << std::endl;
    } else {
        std::cerr << "Conversion failed." << std::endl;
    }

    std::cout << "Level set method completed successfully." << std::endl;
    
    return 0;
}
