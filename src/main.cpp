#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <regex>
#include <sstream>
#include <cmath>
#include "DFISEParser.cpp"

int main() {
    DFISEParser parser("/Users/yincheangng/worksapce/Github/EDA_competition/data/initial_struct.bnd");
    parser.parse();
    
    // Print some basic information
    std::cout << "File version: " << parser.getInfoDouble()["version"] << std::endl;
    std::cout << "Type: " << parser.getInfoString()["type"] << std::endl;
    std::cout << "Dimension: " << parser.getInfoInt()["dimension"] << std::endl;
    std::cout << "Number of vertices: " << parser.getInfoInt()["nb_vertices"] << std::endl;
    std::cout << "Number of edges: " << parser.getInfoInt()["nb_edges"] << std::endl;
    std::cout << "Number of faces: " << parser.getInfoInt()["nb_faces"] << std::endl;
    std::cout << "Number of elements: " << parser.getInfoInt()["nb_elements"] << std::endl;
    std::cout << "Number of regions: " << parser.getInfoInt()["nb_regions"] << std::endl;
    
    // Print the first few vertices
    auto vertices = parser.getVertices();
    if (!vertices.empty()) {
        std::cout << "\nFirst 5 vertices:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), vertices.size()); ++i) {
            std::cout << "Vertex " << i+1 << ": [";
            for (size_t j = 0; j < vertices[i].size(); ++j) {
                std::cout << vertices[i][j];
                if (j < vertices[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Print the first few edges
    auto edges = parser.getEdges();
    if (!edges.empty()) {
        std::cout << "\nFirst 5 edges:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), edges.size()); ++i) {
            std::cout << "Edge " << i+1 << ": [";
            for (size_t j = 0; j < edges[i].size(); ++j) {
                std::cout << edges[i][j];
                if (j < edges[i].size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // Print the regions and materials
    std::cout << "\nRegions:" << std::endl;
    auto regions = parser.getRegions();
    for (size_t i = 0; i < regions.size(); ++i) {
        std::cout << "Region " << i+1 << ": " << regions[i] << std::endl;
    }
    
    std::cout << "\nMaterials:" << std::endl;
    auto materials = parser.getMaterials();
    for (size_t i = 0; i < materials.size(); ++i) {
        std::cout << "Material " << i+1 << ": " << materials[i] << std::endl;
    }
    
    return 0;
}