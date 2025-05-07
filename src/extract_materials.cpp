#include <iostream>
#include <string>
#include "DFISEParser.cpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file.bnd> [output_obj_file] [output_csv_file]" << std::endl;
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_obj_file = (argc > 2) ? argv[2] : "output.obj";
    std::string output_csv_file = (argc > 3) ? argv[3] : "materials.csv";
    
    // Parse the DF-ISE file
    std::cout << "Parsing file: " << input_file << std::endl;
    DFISEParser parser(input_file);
    parser.parse();
    
    // Print some statistics
    std::cout << "Number of vertices: " << parser.getVertices().size() << std::endl;
    std::cout << "Number of edges: " << parser.getEdges().size() << std::endl;
    std::cout << "Number of faces: " << parser.getFaces().size() << std::endl;
    std::cout << "Number of elements: " << parser.getElements().size() << std::endl;
    std::cout << "Number of regions: " << parser.getRegions().size() << std::endl;
    std::cout << "Number of materials: " << parser.getMaterials().size() << std::endl;
    
    // Get vertex-to-material mapping
    auto vertex_materials = parser.getVertexMaterials();
    std::cout << "Number of vertices with material information: " << vertex_materials.size() << std::endl;
    
    // Export to OBJ file with material information
    std::cout << "Exporting to OBJ file: " << output_obj_file << std::endl;
    parser.exportToObj(output_obj_file);
    
    // Export materials to CSV file
    std::cout << "Exporting materials to CSV file: " << output_csv_file << std::endl;
    parser.exportMaterialsToCSV(output_csv_file);
    
    return 0;
}