#ifndef OBJ_TO_BND_CONVERTER_H
#define OBJ_TO_BND_CONVERTER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <regex>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <set>

class OBJToBNDConverter {
public:
    // Constructor
    OBJToBNDConverter(const std::string& objPath) : obj_file_path(objPath) {}
    
    // Method to parse the OBJ file
    bool parseOBJ() {
        std::ifstream file(obj_file_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << obj_file_path << std::endl;
            return false;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            
            if (token == "v") { // Vertex
                double x, y, z;
                if (iss >> x >> y >> z) {
                    std::vector<double> vertex = {x, y, z};
                    vertices.push_back(vertex);
                }
            } else if (token == "f") { // Face
                std::vector<int> face;
                std::string vertex_str;
                while (iss >> vertex_str) {
                    // OBJ format can have vertex/texture/normal indices
                    // We only care about vertex indices
                    std::istringstream vertex_iss(vertex_str);
                    std::string vertex_index_str;
                    std::getline(vertex_iss, vertex_index_str, '/');
                    
                    try {
                        // OBJ indices are 1-based, convert to 0-based
                        int vertex_index = std::stoi(vertex_index_str) - 1;
                        face.push_back(vertex_index);
                    } catch (const std::exception& e) {
                        std::cerr << "Error parsing face index: " << e.what() << std::endl;
                    }
                }
                
                if (face.size() >= 3) { // Valid face must have at least 3 vertices
                    faces.push_back(face);
                }
            } else if (token == "mtllib") { // Material library
                iss >> mtl_lib;
            } else if (token == "usemtl") { // Use material
                iss >> current_material;
                if (!current_material.empty()) {
                    materials.insert(current_material);
                }
            } else if (token == "g" || token == "o") { // Group or object name
                iss >> current_group;
                if (!current_group.empty()) {
                    regions.insert(current_group);
                }
            }
        }
        
        file.close();
        
        // Create edges from faces
        createEdgesFromFaces();
        
        // Create elements (if needed)
        createElementsFromFaces();
        
        return !vertices.empty() && !faces.empty();
    }
    
    // Method to export to BND format
    bool exportToBND(const std::string& output_path) {
        std::ofstream file(output_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << output_path << " for writing" << std::endl;
            return false;
        }
        
        // Write Info section
        file << "Info {" << std::endl;
        file << "    version = 1.0" << std::endl;
        file << "    type = grid" << std::endl;
        file << "    dimension = 3" << std::endl;
        file << "    nb_vertices = " << vertices.size() << std::endl;
        file << "    nb_edges = " << edges.size() << std::endl;
        file << "    nb_faces = " << faces.size() << std::endl;
        file << "    nb_elements = " << elements.size() << std::endl;
        file << "    nb_regions = " << regions.size() << std::endl;
        
        // Write materials list
        file << "    materials = [";
        bool first_material = true;
        for (const auto& material : materials) {
            if (!first_material) {
                file << " ";
            }
            file << "\"" << material << "\"";
            first_material = false;
        }
        file << "]" << std::endl;
        
        // Write regions list
        file << "    regions = [";
        bool first_region = true;
        for (const auto& region : regions) {
            if (!first_region) {
                file << " ";
            }
            file << "\"" << region << "\"";
            first_region = false;
        }
        file << "]" << std::endl;
        file << "}" << std::endl;
        
        // Write Data section
        file << "Data {" << std::endl;
        
        // Write CoordSystem
        file << "    CoordSystem {" << std::endl;
        file << "        translate = [0 0 0]" << std::endl;
        file << "        transform = [1 0 0 0 1 0 0 0 1]" << std::endl;
        file << "    }" << std::endl;
        
        // Write Vertices
        file << "    Vertices (" << vertices.size() << ") {" << std::endl;
        for (const auto& vertex : vertices) {
            file << "        " << vertex[0] << " " << vertex[1] << " " << vertex[2] << std::endl;
        }
        file << "    }" << std::endl;
        
        // Write Edges
        file << "    Edges (" << edges.size() << ") {" << std::endl;
        for (const auto& edge : edges) {
            file << "        " << edge[0] << " " << edge[1] << std::endl;
        }
        file << "    }" << std::endl;
        
        // Write Faces
        file << "    Faces (" << faces.size() << ") {" << std::endl;
        for (const auto& face : faces) {
            file << "        " << face.size();
            for (int edge_idx : face_to_edges[&face]) {
                file << " " << edge_idx;
            }
            file << std::endl;
        }
        file << "    }" << std::endl;
        
        // Write Elements (if any)
        if (!elements.empty()) {
            file << "    Elements (" << elements.size() << ") {" << std::endl;
            for (const auto& element : elements) {
                file << "        10 0"; // Assuming tetrahedron type
                for (int face_idx : element) {
                    file << " " << face_idx;
                }
                file << std::endl;
            }
            file << "    }" << std::endl;
        }
        
        // Write Regions (if any)
        for (const auto& region : regions) {
            file << "    Region (\"" << region << "\") {" << std::endl;
            file << "        material = " << (materials.empty() ? "unknown" : *materials.begin()) << std::endl;
            
            // Write Elements for this region
            file << "        Elements (" << elements.size() << ") {" << std::endl;
            for (size_t i = 0; i < elements.size(); ++i) {
                file << "            " << i << std::endl;
            }
            file << "        }" << std::endl;
            file << "    }" << std::endl;
        }
        
        file << "}" << std::endl;
        
        file.close();
        std::cout << "Successfully exported to BND format: " << output_path << std::endl;
        return true;
    }
    
private:
    std::string obj_file_path;
    std::string mtl_lib;
    std::string current_material;
    std::string current_group;
    
    std::vector<std::vector<double>> vertices;
    std::vector<std::vector<int>> edges;
    std::vector<std::vector<int>> faces;
    std::vector<std::vector<int>> elements;
    std::set<std::string> materials;
    std::set<std::string> regions;
    
    // Maps to store relationships
    std::map<const std::vector<int>*, std::vector<int>> face_to_edges;
    
    // Helper method to create edges from faces
    void createEdgesFromFaces() {
        std::map<std::pair<int, int>, int> edge_map; // Maps vertex pairs to edge indices
        
        for (const auto& face : faces) {
            std::vector<int> face_edges;
            
            for (size_t i = 0; i < face.size(); ++i) {
                int v1 = face[i];
                int v2 = face[(i + 1) % face.size()]; // Wrap around to the first vertex
                
                // Ensure v1 < v2 for consistent edge representation
                if (v1 > v2) {
                    std::swap(v1, v2);
                }
                
                std::pair<int, int> edge_key(v1, v2);
                
                // Check if this edge already exists
                auto it = edge_map.find(edge_key);
                if (it == edge_map.end()) {
                    // Create a new edge
                    int edge_idx = edges.size();
                    edges.push_back({v1, v2});
                    edge_map[edge_key] = edge_idx;
                    face_edges.push_back(edge_idx);
                } else {
                    // Use existing edge
                    face_edges.push_back(it->second);
                }
            }
            
            face_to_edges[&face] = face_edges;
        }
    }
    
    // Helper method to create elements from faces (if needed)
    void createElementsFromFaces() {
        // In this simple implementation, we'll create one element per face
        // This is a simplification and might not be appropriate for all models
        for (size_t i = 0; i < faces.size(); ++i) {
            elements.push_back({static_cast<int>(i)});
        }
    }
};

// Function to convert OBJ to BND
int ConvertOBJToBND(const std::string& inputFile, const std::string& outputFile) {
    OBJToBNDConverter converter(inputFile);
    
    if (!converter.parseOBJ()) {
        std::cerr << "Failed to parse OBJ file" << std::endl;
        return 1;
    }
    
    if (!converter.exportToBND(outputFile)) {
        std::cerr << "Failed to export to BND format" << std::endl;
        return 1;
    }
    
    return 0;
}

#endif