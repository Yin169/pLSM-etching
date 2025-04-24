#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <regex>
#include <sstream>
#include <cmath>

// VTK includes for 3D visualization
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkLine.h>
#include <vtkAxesActor.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkLookupTable.h>

// For plotting, we would need a C++ plotting library
// This implementation focuses on the parser functionality

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
        while (std::getline(iss, line)) {
            std::vector<int> indices;
            std::istringstream line_iss(line);
            int index;
            while (line_iss >> index) {
                indices.push_back(index);
            }
            if (!indices.empty()) {
                elements.push_back(indices);
            }
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

    // Note: In C++, we would need a separate plotting library
    // This function is a placeholder for the equivalent Python function
    void plotGeometry(bool show_vertices = true, bool show_edges = true, int vertex_size = 10,
                     const std::string& edge_color = "blue", const std::string& vertex_color = "red") {
        // Create a renderer, render window, and interactor
        vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
        vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
        renderWindow->AddRenderer(renderer);
        vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = 
            vtkSmartPointer<vtkRenderWindowInteractor>::New();
        renderWindowInteractor->SetRenderWindow(renderWindow);
        
        // Set background color to white
        renderer->SetBackground(1.0, 1.0, 1.0);
        
        // Create a lookup table for colors
        vtkSmartPointer<vtkLookupTable> colorLookupTable = vtkSmartPointer<vtkLookupTable>::New();
        colorLookupTable->SetNumberOfTableValues(2);
        
        // Set colors for vertices and edges
        double vertexRGB[3] = {1.0, 0.0, 0.0}; // Red by default
        double edgeRGB[3] = {0.0, 0.0, 1.0};   // Blue by default
        
        // Convert color strings to RGB
        if (vertex_color == "red") {
            vertexRGB[0] = 1.0; vertexRGB[1] = 0.0; vertexRGB[2] = 0.0;
        } else if (vertex_color == "green") {
            vertexRGB[0] = 0.0; vertexRGB[1] = 1.0; vertexRGB[2] = 0.0;
        } else if (vertex_color == "blue") {
            vertexRGB[0] = 0.0; vertexRGB[1] = 0.0; vertexRGB[2] = 1.0;
        }
        
        if (edge_color == "red") {
            edgeRGB[0] = 1.0; edgeRGB[1] = 0.0; edgeRGB[2] = 0.0;
        } else if (edge_color == "green") {
            edgeRGB[0] = 0.0; edgeRGB[1] = 1.0; edgeRGB[2] = 0.0;
        } else if (edge_color == "blue") {
            edgeRGB[0] = 0.0; edgeRGB[1] = 0.0; edgeRGB[2] = 1.0;
        }
        
        colorLookupTable->SetTableValue(0, vertexRGB[0], vertexRGB[1], vertexRGB[2], 1.0);
        colorLookupTable->SetTableValue(1, edgeRGB[0], edgeRGB[1], edgeRGB[2], 1.0);
        colorLookupTable->Build();
        
        // Plot vertices if requested
        if (show_vertices && !vertices.empty()) {
            // Create points for vertices
            vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
            vtkSmartPointer<vtkCellArray> vertices_vtk = vtkSmartPointer<vtkCellArray>::New();
            
            // Add each vertex as a point
            for (size_t i = 0; i < vertices.size(); ++i) {
                if (vertices[i].size() == 3) {
                    vtkIdType id = points->InsertNextPoint(vertices[i][0], vertices[i][1], vertices[i][2]);
                    vertices_vtk->InsertNextCell(1, &id);
                }
            }
            
            // Create a polydata object
            vtkSmartPointer<vtkPolyData> pointsPolyData = vtkSmartPointer<vtkPolyData>::New();
            pointsPolyData->SetPoints(points);
            pointsPolyData->SetVerts(vertices_vtk);
            
            // Create a mapper and actor for vertices
            vtkSmartPointer<vtkPolyDataMapper> pointsMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            pointsMapper->SetInputData(pointsPolyData);
            pointsMapper->ScalarVisibilityOff();
            
            vtkSmartPointer<vtkActor> pointsActor = vtkSmartPointer<vtkActor>::New();
            pointsActor->SetMapper(pointsMapper);
            pointsActor->GetProperty()->SetColor(vertexRGB[0], vertexRGB[1], vertexRGB[2]);
            pointsActor->GetProperty()->SetPointSize(vertex_size);
            
            // Add the actor to the renderer
            renderer->AddActor(pointsActor);
        }
        
        // Plot edges if requested
        if (show_edges && !edges.empty()) {
            // Create a vtkCellArray to store the lines
            vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
            vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();
            
            // Add all vertices to points
            for (size_t i = 0; i < vertices.size(); ++i) {
                if (vertices[i].size() == 3) {
                    points->InsertNextPoint(vertices[i][0], vertices[i][1], vertices[i][2]);
                }
            }
            
            // Add each edge as a line
            for (size_t i = 0; i < edges.size(); ++i) {
                if (edges[i].size() == 2) {
                    int v1 = edges[i][0];
                    int v2 = edges[i][1];
                    
                    // Check if indices are valid
                    if (0 <= v1 && v1 < static_cast<int>(vertices.size()) && 
                        0 <= v2 && v2 < static_cast<int>(vertices.size())) {
                        vtkSmartPointer<vtkLine> line = vtkSmartPointer<vtkLine>::New();
                        line->GetPointIds()->SetId(0, v1);
                        line->GetPointIds()->SetId(1, v2);
                        lines->InsertNextCell(line);
                    }
                }
            }
            
            // Create a polydata object
            vtkSmartPointer<vtkPolyData> linesPolyData = vtkSmartPointer<vtkPolyData>::New();
            linesPolyData->SetPoints(points);
            linesPolyData->SetLines(lines);
            
            // Create a mapper and actor for edges
            vtkSmartPointer<vtkPolyDataMapper> linesMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
            linesMapper->SetInputData(linesPolyData);
            linesMapper->ScalarVisibilityOff();
            
            vtkSmartPointer<vtkActor> linesActor = vtkSmartPointer<vtkActor>::New();
            linesActor->SetMapper(linesMapper);
            linesActor->GetProperty()->SetColor(edgeRGB[0], edgeRGB[1], edgeRGB[2]);
            linesActor->GetProperty()->SetLineWidth(1.0);
            
            // Add the actor to the renderer
            renderer->AddActor(linesActor);
        }
        
        // Add axes for reference
        vtkSmartPointer<vtkAxesActor> axes = vtkSmartPointer<vtkAxesActor>::New();
        axes->SetTotalLength(1.0, 1.0, 1.0);
        axes->SetShaftType(0);
        axes->SetAxisLabels(1);
        axes->SetCylinderRadius(0.02);
        
        // Add a title to the renderer
        vtkSmartPointer<vtkTextActor> textActor = vtkSmartPointer<vtkTextActor>::New();
        textActor->SetInput("3D Geometry from DF-ISE File");
        textActor->GetTextProperty()->SetFontSize(24);
        textActor->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
        textActor->SetPosition(10, 10);
        renderer->AddActor2D(textActor);
        
        // Add the axes to the renderer
        renderer->AddActor(axes);
        
        // Reset camera to show all actors
        renderer->ResetCamera();
        
        // Set up the render window and start the interaction
        renderWindow->SetSize(1000, 800);
        renderWindow->SetWindowName("DFISEParser Geometry Visualization");
        renderWindowInteractor->Initialize();
        renderWindow->Render();
        renderWindowInteractor->Start();
    }
};