#ifndef OBJ_TO_BND_CONVERTER_H
#define OBJ_TO_BND_CONVERTER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

struct Vector3 {
    double x, y, z;
    Vector3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    Vector3 cross(const Vector3& other) const {
        return Vector3(y * other.z - z * other.y, 
                      z * other.x - x * other.z, 
                      x * other.y - y * other.x);
    }
    double dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    Vector3 normalize() const {
        double len = sqrt(x*x + y*y + z*z);
        return len > 0 ? Vector3(x/len, y/len, z/len) : Vector3();
    }
};

struct Edge {
    int v0, v1;
    Edge(int v0, int v1) : v0(std::min(v0, v1)), v1(std::max(v0, v1)) {}
    bool operator<(const Edge& other) const {
        return v0 < other.v0 || (v0 == other.v0 && v1 < other.v1);
    }
    bool operator==(const Edge& other) const {
        return v0 == other.v0 && v1 == other.v1;
    }
};

struct Face {
    std::vector<int> vertices;
    std::vector<int> edges;
    Vector3 normal;
    bool is_oriented = false;
};

class ObjToDfiseConverter {
private:
    std::vector<Vector3> vertices;
    std::vector<Edge> edges;
    std::vector<Face> faces;
    std::map<Edge, int> edge_map;
    std::string material_name;
    std::string region_name;

    // Parse OBJ file
    bool parseObjFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open OBJ file: " << filename << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string prefix;
            iss >> prefix;

            if (prefix == "v") {
                // Vertex
                double x, y, z;
                iss >> x >> y >> z;
                vertices.push_back(Vector3(x, y, z));
            }
            else if (prefix == "f") {
                // Face
                Face face;
                std::string vertex_str;
                while (iss >> vertex_str) {
                    // Handle vertex/texture/normal format (v/vt/vn or v//vn or just v)
                    size_t slash_pos = vertex_str.find('/');
                    int vertex_idx;
                    if (slash_pos != std::string::npos) {
                        vertex_idx = std::stoi(vertex_str.substr(0, slash_pos));
                    } else {
                        vertex_idx = std::stoi(vertex_str);
                    }
                    
                    // OBJ indices are 1-based, convert to 0-based
                    vertex_idx--;
                    if (vertex_idx < 0 || vertex_idx >= vertices.size()) {
                        std::cerr << "Error: Invalid vertex index in face" << std::endl;
                        return false;
                    }
                    face.vertices.push_back(vertex_idx);
                }
                
                if (face.vertices.size() >= 3) {
                    faces.push_back(face);
                }
            }
        }

        file.close();
        return true;
    }

    // Create edges from faces and build edge map
    void buildEdges() {
        std::set<Edge> unique_edges;
        
        for (auto& face : faces) {
            for (size_t i = 0; i < face.vertices.size(); ++i) {
                int v0 = face.vertices[i];
                int v1 = face.vertices[(i + 1) % face.vertices.size()];
                Edge edge(v0, v1);
                unique_edges.insert(edge);
            }
        }

        // Convert set to vector and create edge map
        edges.assign(unique_edges.begin(), unique_edges.end());
        for (size_t i = 0; i < edges.size(); ++i) {
            edge_map[edges[i]] = i;
        }
    }

    // Assign edges to faces with proper orientation
    void assignEdgesToFaces() {
        for (auto& face : faces) {
            face.edges.clear();
            
            // Calculate face normal for orientation check
            if (face.vertices.size() >= 3) {
                Vector3 v0 = vertices[face.vertices[0]];
                Vector3 v1 = vertices[face.vertices[1]];
                Vector3 v2 = vertices[face.vertices[2]];
                face.normal = (v1 - v0).cross(v2 - v0).normalize();
            }

            // Assign edges with proper orientation
            for (size_t i = 0; i < face.vertices.size(); ++i) {
                int va = face.vertices[i];
                int vb = face.vertices[(i + 1) % face.vertices.size()];
                
                Edge edge(va, vb);
                auto it = edge_map.find(edge);
                if (it != edge_map.end()) {
                    int edge_idx = it->second;
                    
                    // Check if edge needs to be inverted for proper orientation
                    // DF-ISE requires counterclockwise orientation when looking from outside
                    if (edge.v0 == va && edge.v1 == vb) {
                        // Edge is in correct direction
                        face.edges.push_back(edge_idx);
                    } else {
                        // Edge needs to be inverted (negative index)
                        face.edges.push_back(-(edge_idx + 1));
                    }
                }
            }
            face.is_oriented = true;
        }
    }

    // Ensure consistent face orientation (all faces should have outward-pointing normals)
    void ensureConsistentOrientation() {
        // This is a simplified approach - for complex meshes, a more sophisticated
        // algorithm would be needed to ensure global consistency
        for (auto& face : faces) {
            if (face.vertices.size() >= 3) {
                // Calculate centroid
                Vector3 centroid;
                for (int vid : face.vertices) {
                    centroid.x += vertices[vid].x;
                    centroid.y += vertices[vid].y;
                    centroid.z += vertices[vid].z;
                }
                centroid.x /= face.vertices.size();
                centroid.y /= face.vertices.size();
                centroid.z /= face.vertices.size();

                // Simple heuristic: assume faces should point outward from origin
                // For more complex cases, you might need mesh analysis
                Vector3 to_center = Vector3(0, 0, 0) - centroid;
                if (face.normal.dot(to_center) > 0) {
                    // Face is pointing inward, reverse it
                    std::reverse(face.vertices.begin(), face.vertices.end());
                    std::reverse(face.edges.begin(), face.edges.end());
                    
                    // Flip edge orientations
                    for (int& edge_idx : face.edges) {
                        if (edge_idx >= 0) {
                            edge_idx = -(edge_idx + 1);
                        } else {
                            edge_idx = (-edge_idx - 1);
                        }
                    }
                    
                    // Recalculate normal
                    Vector3 v0 = vertices[face.vertices[0]];
                    Vector3 v1 = vertices[face.vertices[1]];
                    Vector3 v2 = vertices[face.vertices[2]];
                    face.normal = (v1 - v0).cross(v2 - v0).normalize();
                }
            }
        }
    }

    // Write DF-ISE boundary file
    bool writeDfiseFile(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot create output file: " << filename << std::endl;
            return false;
        }

        // Write Info block
        file << "DF-ISE text\n\n";
        file << "Info {\n";
        file << "  version = 1.0\n";
        file << "  type = boundary\n";
        file << "  dimension = 3\n";
        file << "  nb_vertices = " << vertices.size() << "\n";
        file << "  nb_edges = " << edges.size() << "\n";
        file << "  nb_faces = " << faces.size() << "\n";
        file << "  nb_elements = 0\n";
        file << "  nb_regions = 1\n";
        file << "  regions = [ \"" << region_name << "\" ]\n";
        file << "  materials = [ " << material_name << " ]\n";
        file << "}\n\n";

        // Write Data block
        file << "Data {\n\n";

        // Write coordinate system (identity)
        file << "  CoordSystem {\n";
        file << "    translate = [ 0.0 0.0 0.0 ]\n";
        file << "    transform = [ 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 ]\n";
        file << "  }\n\n";

        // Write vertices
        file << "  Vertices (" << vertices.size() << ") {\n";
        for (const auto& vertex : vertices) {
            file << "    " << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
        }
        file << "  }\n\n";

        // Write edges
        file << "  Edges (" << edges.size() << ") {\n";
        for (const auto& edge : edges) {
            file << "    " << edge.v0 << " " << edge.v1 << "\n";
        }
        file << "  }\n\n";

        // Write faces
        file << "  Faces (" << faces.size() << ") {\n";
        for (const auto& face : faces) {
            file << "    " << face.edges.size();
            for (int edge_idx : face.edges) {
                file << " " << edge_idx;
            }
            file << "\n";
        }
        file << "  }\n\n";

        // Write locations (all external for boundary)
        file << "  Locations (" << faces.size() << ") {\n";
        for (size_t i = 0; i < faces.size(); ++i) {
            file << "    e\n";  // external
        }
        file << "  }\n\n";

        // Write elements (empty for boundary)
        file << "  Elements (0) {\n";
        file << "  }\n\n";

        // Write region
        file << "  Region (\"" << region_name << "\") {\n";
        file << "    material = " << material_name << "\n";
        file << "    Faces (" << faces.size() << ") {";
        for (size_t i = 0; i < faces.size(); ++i) {
            if (i % 10 == 0) file << "\n     ";
            file << " " << i;
        }
        file << "\n    }\n";
        file << "  }\n\n";

        file << "}\n";
        file.close();
        return true;
    }

public:
    ObjToDfiseConverter(const std::string& material = "Silicon", const std::string& region = "object") 
        : material_name(material), region_name(region) {}

    bool convert(const std::string& obj_filename, const std::string& bnd_filename) {
        std::cout << "Parsing OBJ file: " << obj_filename << std::endl;
        if (!parseObjFile(obj_filename)) {
            return false;
        }

        std::cout << "Loaded " << vertices.size() << " vertices and " << faces.size() << " faces" << std::endl;

        std::cout << "Building edges..." << std::endl;
        buildEdges();
        std::cout << "Created " << edges.size() << " edges" << std::endl;

        std::cout << "Assigning edges to faces with proper orientation..." << std::endl;
        assignEdgesToFaces();

        std::cout << "Ensuring consistent face orientation..." << std::endl;
        ensureConsistentOrientation();

        std::cout << "Writing DF-ISE boundary file: " << bnd_filename << std::endl;
        if (!writeDfiseFile(bnd_filename)) {
            return false;
        }

        std::cout << "Conversion completed successfully!" << std::endl;
        return true;
    }

    void setMaterial(const std::string& material) { material_name = material; }
    void setRegion(const std::string& region) { region_name = region; }
};

#endif