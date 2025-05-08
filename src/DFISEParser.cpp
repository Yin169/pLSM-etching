#ifndef DFISE_PARSER_H
#define DFISE_PARSER_H
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <regex>
#include <sstream>
#include <cmath>
#include <cstring>
#include <set>

class DFISEParser {
private:
    std::string file_path;
    std::map<std::string, double> info_double;
    std::map<std::string, int> info_int;
    std::map<std::string, std::string> info_string;
    std::vector<std::vector<double>> vertices;
    std::vector<std::vector<int>> edges;
    std::vector<std::vector<int>> faces;
    std::vector<std::vector<int>> elements;
    std::vector<std::string> regions;
    std::vector<std::string> materials;
    std::map<std::string, std::vector<double>> coord_system_vector;
    std::map<std::string, std::vector<std::vector<double>>> coord_system_matrix;
    
    // New data structures for region-element-material mapping
    struct RegionInfo {
        std::string name;
        std::string material;
        std::vector<int> elements;
    };
    std::vector<RegionInfo> region_info;
    std::map<int, std::string> element_to_material;
    std::map<int, std::string> face_to_material;
    
    // Helper function to extract content between braces
    std::string extractBetweenBraces(const std::string& content, const std::string& section) {
        // Find the starting position of the section
        std::regex section_pattern(section + "\\s*\\{");
        std::smatch section_match;
        if (!std::regex_search(content, section_match, section_pattern)) {
            return "";
        }
        
        size_t start_pos = section_match.position() + section_match.length();
        size_t pos = start_pos;
        int brace_count = 1; // We've already found the opening brace
        
        // Find the matching closing brace by counting opening and closing braces
        while (pos < content.length() && brace_count > 0) {
            if (content[pos] == '{') {
                brace_count++;
            } else if (content[pos] == '}') {
                brace_count--;
            }
            pos++;
        }
        
        // If we found the matching closing brace, return the content between braces
        if (brace_count == 0) {
            return content.substr(start_pos, pos - start_pos - 1);
        }
        
        return "";
    }

    // Helper function to extract content between parentheses
    std::string extractBetweenParentheses(const std::string& content, const std::string& section) {
        std::regex pattern(section + "\\s*\\(\\d+\\)\\s*\\{([^{}]*)\\}");
        std::smatch matches;
        if (std::regex_search(content, matches, pattern)) {
            return matches[1].str();
        }
        return "";
    }

    // Helper function to extract vector values
    std::vector<double> extractVector(const std::string& text) {
        std::vector<double> result;
        std::regex pattern("\\[(.*?)\\]");
        std::smatch matches;
        if (std::regex_search(text, matches, pattern)) {
            std::string values = matches[1].str();
            std::istringstream iss(values);
            double value;
            while (iss >> value) {
                result.push_back(value);
            }
        }
        return result;
    }

    // Helper function to extract matrix values
    std::vector<std::vector<double>> extractMatrix(const std::string& text) {
        std::vector<std::vector<double>> result;
        std::regex pattern("\\[(.*?)\\]");
        std::smatch matches;
        if (std::regex_search(text, matches, pattern)) {
            std::string values = matches[1].str();
            std::istringstream iss(values);
            std::string line;
            while (std::getline(iss, line)) {
                std::vector<double> row;
                std::istringstream line_iss(line);
                double value;
                while (line_iss >> value) {
                    row.push_back(value);
                }
                if (!row.empty()) {
                    result.push_back(row);
                }
            }
        }
        return result;
    }

    // Helper function to extract string list
    std::vector<std::string> extractStringList(const std::string& text) {
        std::vector<std::string> result;
        std::regex pattern("\\[(.*?)\\]");
        std::smatch matches;
        if (std::regex_search(text, matches, pattern)) {
            std::string values = matches[1].str();
            // Remove quotes and split by spaces or newlines
            std::regex word_pattern("\"([^\"]*)\"");
            auto words_begin = std::sregex_iterator(values.begin(), values.end(), word_pattern);
            auto words_end = std::sregex_iterator();
            for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                std::smatch match = *i;
                result.push_back(match[1].str());
            }
        }
        return result;
    }

    void parseInfo(const std::string& info_text) {
        // Extract version
        std::regex version_pattern("version\\s*=\\s*([\\d\\.]+)");
        std::smatch version_match;
        if (std::regex_search(info_text, version_match, version_pattern)) {
            info_double["version"] = std::stod(version_match[1].str());
        }

        // Extract type
        std::regex type_pattern("type\\s*=\\s*(\\w+)");
        std::smatch type_match;
        if (std::regex_search(info_text, type_match, type_pattern)) {
            info_string["type"] = type_match[1].str();
        }

        // Extract dimension
        std::regex dim_pattern("dimension\\s*=\\s*(\\d+)");
        std::smatch dim_match;
        if (std::regex_search(info_text, dim_match, dim_pattern)) {
            info_int["dimension"] = std::stoi(dim_match[1].str());
        }

        // Extract number of vertices
        std::regex vertices_pattern("nb_vertices\\s*=\\s*(\\d+)");
        std::smatch vertices_match;
        if (std::regex_search(info_text, vertices_match, vertices_pattern)) {
            info_int["nb_vertices"] = std::stoi(vertices_match[1].str());
        }

        // Extract number of edges
        std::regex edges_pattern("nb_edges\\s*=\\s*(\\d+)");
        std::smatch edges_match;
        if (std::regex_search(info_text, edges_match, edges_pattern)) {
            info_int["nb_edges"] = std::stoi(edges_match[1].str());
        }

        // Extract number of faces
        std::regex faces_pattern("nb_faces\\s*=\\s*(\\d+)");
        std::smatch faces_match;
        if (std::regex_search(info_text, faces_match, faces_pattern)) {
            info_int["nb_faces"] = std::stoi(faces_match[1].str());
        }

        // Extract number of elements
        std::regex elements_pattern("nb_elements\\s*=\\s*(\\d+)");
        std::smatch elements_match;
        if (std::regex_search(info_text, elements_match, elements_pattern)) {
            info_int["nb_elements"] = std::stoi(elements_match[1].str());
        }

        // Extract number of regions
        std::regex regions_pattern("nb_regions\\s*=\\s*(\\d+)");
        std::smatch regions_match;
        if (std::regex_search(info_text, regions_match, regions_pattern)) {
            info_int["nb_regions"] = std::stoi(regions_match[1].str());
        }

        // Extract regions
        std::regex regions_list_pattern("regions\\s*=\\s*\\[(.*?)\\]");
        std::smatch regions_list_match;
        if (std::regex_search(info_text, regions_list_match, regions_list_pattern)) {
            std::string regions_text = regions_list_match[1].str();
            regions = extractStringList(regions_text);
        }

        // Extract materials
        std::regex materials_pattern("materials\\s*=\\s*\\[(.*?)\\]");
        std::smatch materials_match;
        if (std::regex_search(info_text, materials_match, materials_pattern)) {
            std::string materials_text = materials_match[1].str();
            materials = extractStringList(materials_text);
        }
    }

    void parseData(const std::string& data_text) {
        // Extract CoordSystem
        std::string coord_text = extractBetweenBraces(data_text, "CoordSystem");
        if (!coord_text.empty()) {
            parseCoordSystem(coord_text);
        }

        // Extract Vertices
        std::string vertices_text = extractBetweenParentheses(data_text, "Vertices");
        if (!vertices_text.empty()) {
            parseVertices(vertices_text);
        }

        // Extract Edges
        std::string edges_text = extractBetweenParentheses(data_text, "Edges");
        if (!edges_text.empty()) {
            parseEdges(edges_text);
        }

        // Extract Faces
        std::string faces_text = extractBetweenParentheses(data_text, "Faces");
        if (!faces_text.empty()) {
            parseFaces(faces_text);
        }

        // Extract Elements
        std::string elements_text = extractBetweenParentheses(data_text, "Elements");
        if (!elements_text.empty()) {
            parseElements(elements_text);
        }
        
        // Parse Regions
        parseRegions(data_text);
        
        // Map materials to faces
        mapMaterialsToFaces();
    }
    
    // New method to parse regions
    void parseRegions(const std::string& data_text) {
        // This pattern matches the entire region block including nested braces
        std::regex region_pattern("Region\\s*\\(\\s*\"([^\"]*)\"\\s*\\)\\s*\\{([^{}]*)\\}");
        std::smatch region_matches;
        std::string::const_iterator search_start(data_text.cbegin());
        
        while (std::regex_search(search_start, data_text.cend(), region_matches, region_pattern)) {
            std::string region_name = region_matches[1].str();
            std::string region_content = region_matches[2].str();
            
            // Extract material
            std::regex material_pattern("material\\s*=\\s*(\\w+)");
            std::smatch material_match;
            std::string material_name;
            if (std::regex_search(region_content, material_match, material_pattern)) {
                material_name = material_match[1].str();
            }
            
            // Extract elements - modified pattern to match your data format
            std::regex elements_pattern("Elements\\s*\\(\\s*\\d+\\s*\\)\\s*\\{\\s*([^{}]*)\\s*\\}");
            std::smatch elements_match;
            std::vector<int> region_elements;
            if (std::regex_search(region_content, elements_match, elements_pattern)) {
                std::string elements_list = elements_match[1].str();
                std::cout << "Region: " << region_name << ", Material: " << material_name << ", Elements: " << elements_list << std::endl;
                std::istringstream iss(elements_list);
                int element_idx;
                while (iss >> element_idx) {
                    region_elements.push_back(element_idx);
                    // Map element to material
                    element_to_material[element_idx] = material_name;
                    std::cout << "  Mapped element " << element_idx << " to material " << material_name << std::endl;
                }
            }
            
            // Add region info
            RegionInfo info;
            info.name = region_name;
            info.material = material_name;
            info.elements = region_elements;
            region_info.push_back(info);
            
            // Move to next region
            search_start = region_matches.suffix().first;
        }
        
        // Debug output - check how many regions were parsed
        std::cout << "Parsed " << region_info.size() << " regions" << std::endl;
        
        // If no regions were found, try an alternative approach with a more flexible pattern
        if (region_info.empty()) {
            std::cout << "No regions found with standard pattern, trying alternative approach..." << std::endl;
            
            // Alternative pattern that can handle more complex nested structures
            std::regex alt_region_pattern("Region\\s*\\(\\s*\"([^\"]*)\"\\s*\\)\\s*\\{");
            std::smatch alt_region_match;
            search_start = data_text.cbegin();
            
            while (std::regex_search(search_start, data_text.cend(), alt_region_match, alt_region_pattern)) {
                std::string region_name = alt_region_match[1].str();
                size_t region_start = search_start - data_text.cbegin() + alt_region_match.position() + alt_region_match.length();
                
                // Find the closing brace for this region
                size_t pos = region_start;
                int brace_count = 1;
                while (pos < data_text.length() && brace_count > 0) {
                    if (data_text[pos] == '{') {
                        brace_count++;
                    } else if (data_text[pos] == '}') {
                        brace_count--;
                    }
                    pos++;
                }
                
                if (brace_count == 0) {
                    std::string region_content = data_text.substr(region_start, pos - region_start - 1);
                    
                    // Extract material
                    std::regex material_pattern("material\\s*=\\s*(\\w+)");
                    std::smatch material_match;
                    std::string material_name;
                    if (std::regex_search(region_content, material_match, material_pattern)) {
                        material_name = material_match[1].str();
                    }
                    
                    // Extract elements with a more flexible pattern
                    std::regex elements_pattern("Elements\\s*\\(\\s*\\d+\\s*\\)\\s*\\{\\s*([^{}]*)\\s*\\}");
                    std::smatch elements_match;
                    std::vector<int> region_elements;
                    if (std::regex_search(region_content, elements_match, elements_pattern)) {
                        std::string elements_list = elements_match[1].str();
                        std::cout << "Region: " << region_name << ", Material: " << material_name << ", Elements: " << elements_list << std::endl;
                        std::istringstream iss(elements_list);
                        int element_idx;
                        while (iss >> element_idx) {
                            region_elements.push_back(element_idx);
                            // Map element to material
                            element_to_material[element_idx] = material_name;
                            std::cout << "  Mapped element " << element_idx << " to material " << material_name << std::endl;
                        }
                    }
                    
                    // Add region info
                    RegionInfo info;
                    info.name = region_name;
                    info.material = material_name;
                    info.elements = region_elements;
                    region_info.push_back(info);
                }
                
                // Move to next region
                search_start = data_text.cbegin() + pos;
            }
            
            std::cout << "Parsed " << region_info.size() << " regions with alternative approach" << std::endl;
        }
    }
    
    // New method to map materials to faces
    void mapMaterialsToFaces() {
        // For each element, find the faces that belong to it and assign the material
        for (size_t elem_idx = 0; elem_idx < elements.size(); ++elem_idx) {
            const auto& element = elements[elem_idx];
            
            // Check if this element has a material assigned
            auto material_it = element_to_material.find(elem_idx);
            if (material_it == element_to_material.end()) {
                continue;  // Skip elements without material
            }
            
            std::string material = material_it->second;
            
            // In DF-ISE format, elements typically have a specific structure
            // The first value is often the element type, followed by face indices
            // We need to determine which values in the element are face indices
            
            // Skip the first value (element type descriptor)
            for (size_t i = 1; i < element.size(); ++i) {
                int face_idx = element[i];
                
                // Check if this is a valid face index
                if (face_idx >= 0 && face_idx < static_cast<int>(faces.size())) {
                    face_to_material[face_idx] = material;
                }
                // Handle negative indices (which might indicate orientation)
                else if (face_idx < 0 && -face_idx - 1 < static_cast<int>(faces.size())) {
                    face_to_material[-face_idx - 1] = material;
                }
            }
        }
        
        // Debug output - check how many faces have materials assigned
        std::cout << "Assigned materials to " << face_to_material.size() << " faces out of " << faces.size() << std::endl;
    }
    
    void parseCoordSystem(const std::string& coord_text) {
        // Extract translate
        std::regex translate_pattern("translate\\s*=\\s*\\[(.*?)\\]");
        std::smatch translate_match;
        if (std::regex_search(coord_text, translate_match, translate_pattern)) {
            std::string translate_text = translate_match[0].str();
            coord_system_vector["translate"] = extractVector(translate_text);
        }

        // Extract transform
        std::regex transform_pattern("transform\\s*=\\s*\\[(.*?)\\]");
        std::smatch transform_match;
        if (std::regex_search(coord_text, transform_match, transform_pattern)) {
            std::string transform_text = transform_match[0].str();
            coord_system_matrix["transform"] = extractMatrix(transform_text);
        }
    }

    void parseVertices(const std::string& vertices_text) {
        std::istringstream iss(vertices_text);
        std::string line;
        while (std::getline(iss, line)) {
            std::vector<double> coords;
            std::istringstream line_iss(line);
            double coord;
            while (line_iss >> coord) {
                coords.push_back(coord);
            }
            if (coords.size() == 3) {
                vertices.push_back(coords);
            }
        }
    }

    void parseEdges(const std::string& edges_text) {
        std::istringstream iss(edges_text);
        std::string line;
        while (std::getline(iss, line)) {
            std::vector<int> indices;
            std::istringstream line_iss(line);
            int index;
            while (line_iss >> index) {
                indices.push_back(index);
            }
            if (indices.size() == 2) {
                edges.push_back(indices);
            }
        }
    }

    void parseFaces(const std::string& faces_text) {
        std::istringstream iss(faces_text);
        std::string line;
        while (std::getline(iss, line)) {
            std::vector<int> indices;
            std::istringstream line_iss(line);
            int index;
            while (line_iss >> index) {
                indices.push_back(index);
            }
            if (!indices.empty()) {
                faces.push_back(indices);
            }
        }
    }

    void parseElements(const std::string& elements_text) {
        std::istringstream iss(elements_text);
        std::string line;
        std::vector<int> current_element;
        bool in_element = false;

        while (std::getline(iss, line)) {
            std::istringstream line_iss(line);
            std::vector<int> line_numbers;
            int number;
            
            // Read all numbers from the current line
            while (line_iss >> number) {
                line_numbers.push_back(number);
            }
            
            // Skip empty lines
            if (line_numbers.empty()) continue;
            
            // Check if this is the start of a new element
            if (line_numbers[0] == 10 && line_numbers.size() == 12) {
                // If we were already processing an element, add it to elements
                if (!current_element.empty()) {
                    elements.push_back(current_element);
                }
                
                // Start new element, skip the first two numbers (10 and descriptor)
                current_element.clear();
                for (size_t i = 2; i < line_numbers.size(); i++) {
                    current_element.push_back(line_numbers[i]);
                }
                in_element = true;
            }
            // If we're in an element, add all numbers from this line
            else if (in_element) {
                current_element.insert(current_element.end(), line_numbers.begin(), line_numbers.end());
            }
        }
        
        // Add the last element if there is one
        if (!current_element.empty()) {
            elements.push_back(current_element);
        }
    }
    

public:
    DFISEParser(const std::string& file_path) : file_path(file_path) {}

    DFISEParser* parse() {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << file_path << std::endl;
            return this;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        // Extract the Info section
        std::string info_text = extractBetweenBraces(content, "Info");
        if (!info_text.empty()) {
            parseInfo(info_text);
        }

        // Extract the Data section
        std::string data_text = extractBetweenBraces(content, "Data");
        if (!data_text.empty()) {
            parseData(data_text);
        }

        return this;
    }

    std::vector<std::vector<double>> getVertices() const {
        return vertices;
    }

    std::vector<std::vector<int>> getEdges() const {
        return edges;
    }

    std::vector<std::vector<int>> getFaces() const {
        return faces;
    }

    std::vector<std::vector<int>> getElements() const {
        return elements;
    }

    std::map<std::string, double> getInfoDouble() const {
        return info_double;
    }

    std::map<std::string, int> getInfoInt() const {
        return info_int;
    }

    std::map<std::string, std::string> getInfoString() const {
        return info_string;
    }

    std::vector<std::string> getRegions() const {
        return regions;
    }

    std::vector<std::string> getMaterials() const {
        return materials;
    }
    
    // Enhanced OBJ export with material information
    bool exportToObj(const std::string& output_file) {
        std::ofstream file(output_file);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << output_file << " for writing" << std::endl;
            return false;
        }
        
        // Write OBJ header with information
        file << "# Wavefront OBJ file exported from DF-ISE format" << std::endl;
        file << "# Original file: " << file_path << std::endl;
        file << "# Exported by DFISEParser" << std::endl;
        file << "# Vertices: " << vertices.size() << std::endl;
        file << "# Faces: " << faces.size() << std::endl;
        file << "# Materials: " << materials.size() << std::endl;
        file << std::endl;
        
        // Create MTL file for materials
        std::string mtl_filename = output_file.substr(0, output_file.find_last_of('.')) + ".mtl";
        file << "mtllib " << mtl_filename << std::endl << std::endl;
        
        // Write vertices (v x y z)
        for (const auto& vertex : vertices) {
            if (vertex.size() == 3) {
                file << "v " << vertex[0] << " " << vertex[1] << " " << vertex[2] << std::endl;
            }
        }
        file << std::endl;
        
        // Group faces by material
        std::map<std::string, std::vector<size_t>> material_to_faces;
        for (size_t i = 0; i < faces.size(); ++i) {
            std::string material = getMaterialForFace(i);
            material_to_faces[material].push_back(i);
        }
        
        // Write faces grouped by material
        for (const auto& mat_faces : material_to_faces) {
            file << "g " << mat_faces.first << std::endl;
            file << "usemtl " << mat_faces.first << std::endl;
            
            for (size_t face_idx : mat_faces.second) {
                const auto& face = faces[face_idx];
                if (!face.empty()) {
                    file << "f";
                    std::vector<int> face_vertices;
                    
                    // Process each edge in face (skip first element if it's count)
                    for (size_t i = 1; i < face.size(); ++i) {
                        int edge_idx = face[i];
                        bool reverse = edge_idx < 0;
                        int abs_edge_idx;
                        if (reverse) {
                            abs_edge_idx = -edge_idx - 1;                    
                        } else {
                            abs_edge_idx = edge_idx;
                        }
                        
                        if (abs_edge_idx >= 0 && abs_edge_idx < edges.size()) {
                            const auto& edge = edges[abs_edge_idx];
                            if (edge.size() == 2) {
                                // Add vertices in correct order based on edge direction
                                if (reverse) {
                                    face_vertices.push_back(edge[1]);
                                    face_vertices.push_back(edge[0]);
                                } else {
                                    face_vertices.push_back(edge[0]);
                                    face_vertices.push_back(edge[1]);
                                }
                            }
                        }
                    }
                    
                    // Remove consecutive duplicates while maintaining order
                    std::vector<int> unique_vertices;
                    for (auto& v : face_vertices) {
                        if (unique_vertices.empty() || unique_vertices.back() != v) {
                            unique_vertices.push_back(v);
                        }
                    }
                    
                    // Write final vertex indices
                    for (int v : unique_vertices) {
                        file << " " << (v + 1);
                    }
                    file << std::endl;
                }
            }
        }
        
        // If there are no faces but we have edges, write edges as lines
        if (faces.empty() && !edges.empty()) {
            for (const auto& edge : edges) {
                if (edge.size() == 2) {
                    // Convert to 1-based indexing for OBJ format
                    file << "l " << (edge[0] + 1) << " " << (edge[1] + 1) << std::endl;
                }
            }
        }
        
        file.close();
        
        // Create MTL file with basic material definitions
        std::ofstream mtl_file(mtl_filename);
        if (mtl_file.is_open()) {
            mtl_file << "# Material definitions for " << output_file << std::endl;
            mtl_file << "# Generated by DFISEParser" << std::endl << std::endl;
            
            // Create a unique set of materials
            std::set<std::string> unique_materials;
            for (const auto& region : region_info) {
                unique_materials.insert(region.material);
            }
            
            // Add unknown material if needed
            if (material_to_faces.find("unknown") != material_to_faces.end()) {
                unique_materials.insert("unknown");
            }
            
            // Write material definitions
            for (const std::string& material : unique_materials) {
                mtl_file << "newmtl " << material << std::endl;
                
                // Generate a pseudo-random color based on material name for visualization
                unsigned int hash = 0;
                for (char c : material) {
                    hash = hash * 101 + c;
                }
                float r = (hash % 255) / 255.0f;
                float g = ((hash / 255) % 255) / 255.0f;
                float b = ((hash / 255 / 255) % 255) / 255.0f;
                
                mtl_file << "Ka " << r << " " << g << " " << b << std::endl;  // ambient color
                mtl_file << "Kd " << r << " " << g << " " << b << std::endl;  // diffuse color
                mtl_file << "Ks 0.0 0.0 0.0" << std::endl;                   // specular color
                mtl_file << "d 1.0" << std::endl;                            // transparency
                mtl_file << "illum 1" << std::endl << std::endl;             // illumination model
            }
            
            mtl_file.close();
            std::cout << "Successfully created material file: " << mtl_filename << std::endl;
        }
        
        std::cout << "Successfully exported to OBJ format: " << output_file << std::endl;
        return true;
    }

    std::vector<RegionInfo> getRegionInfo() const {
        return region_info;
    }

    std::map<int, std::string> getElementToMaterial() const {
        return element_to_material;
    }

    std::map<int, std::string> getFaceToMaterial() const {
        return face_to_material;
    }

    std::string getMaterialForFace(int face_idx) const {
        auto it = face_to_material.find(face_idx);
        if (it != face_to_material.end()) {
            return it->second;
        }
        return "unknown";
    }


};
#endif
