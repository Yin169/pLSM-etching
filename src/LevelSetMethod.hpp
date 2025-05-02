#ifndef LEVEL_SET_METHOD_HPP
#define LEVEL_SET_METHOD_HPP

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Surface_mesh_default_criteria_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Surface_mesh_complex_2_in_triangulation_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/IO/polygon_mesh_io.h>

#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Surface_mesh.h>

#include <eigen3/Eigen/Dense>
#include <memory>
#include <vector>
#include <string>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

class LevelSetMethod {
public:
    LevelSetMethod(const std::string& filename,
                int gridSize = 400, 
                double timeStep = 0.01, 
                int maxSteps = 80, 
                int reinitInterval = 5,
                double narrowBandWidth = 10.0)
        : GRID_SIZE(gridSize),
        dt(timeStep),
        STEPS(maxSteps),
        REINIT_INTERVAL(reinitInterval),
        NARROW_BAND_WIDTH(narrowBandWidth) {
        loadMesh(filename);
        generateGrid();
    }
    
    // Public methods
    CGAL::Bbox_3 calculateBoundingBox() const;
    bool extractSurfaceMeshCGAL(const std::string& filename);
    void loadMesh(const std::string& filename);
    bool evolve();
    void reinitialize();

private:
    // Private member variables
    const int GRID_SIZE;
    double GRID_SPACING;
    const double dt;
    const int STEPS;
    const int REINIT_INTERVAL;
    const double NARROW_BAND_WIDTH;
    double BOX_SIZE = -1.0;
    
    // Grid origin coordinates for faster lookups
    double gridOriginX = 0.0;
    double gridOriginY = 0.0;
    double gridOriginZ = 0.0;
    
    Mesh mesh;
    std::unique_ptr<AABB_tree> tree;
    std::vector<Point_3> grid;
    Eigen::VectorXd phi;
    std::vector<int> narrowBand;
    
    // Private methods
    double CalculateEtchingRate(double sigma);
    void updateNarrowBand();
    void generateGrid();
    Eigen::VectorXd initializeSignedDistanceField();
    bool isOnBoundary(int idx) const;
    int getIndex(int x, int y, int z) const;
};

CGAL::Bbox_3 LevelSetMethod::calculateBoundingBox() const {
    if (mesh.is_empty()) {
        throw std::runtime_error("Mesh is empty - cannot calculate bounding box");
    }
    return CGAL::Polygon_mesh_processing::bbox(mesh);
}

void LevelSetMethod::loadMesh(const std::string& filename) {
    if (!PMP::IO::read_polygon_mesh(filename, mesh) || is_empty(mesh) || !CGAL::is_closed(mesh) || !is_triangle_mesh(mesh)) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
        
    tree = std::make_unique<AABB_tree>(faces(mesh).first, faces(mesh).second, mesh);
    tree->accelerate_distance_queries(); 
}

bool LevelSetMethod::evolve() {
    try {
        phi = initializeSignedDistanceField();
        updateNarrowBand();
        
        // Pre-allocate memory for new phi values to avoid reallocations
        Eigen::VectorXd newPhi = phi;
        
        // Progress tracking
        const int progressInterval = std::max(1, 10);
        // Cache frequently used constants
        const double inv_grid_spacing = 1.0 / GRID_SPACING;
        const double half_inv_grid_spacing = 0.5 * inv_grid_spacing;
        const double sigma = 0.01; // Etching parameter
        
        // Pre-sort narrow band for better cache locality
        std::sort(narrowBand.begin(), narrowBand.end());
        
        for (int step = 0; step < STEPS; ++step) {
            // Report progress periodically
            if (step % progressInterval == 0) {
                std::cout << "Evolution step " << step << "/" << STEPS << std::endl;
            }
            
            // Use dynamic scheduling with larger chunk size for better cache utilization
            #pragma omp parallel for schedule(dynamic, 128)
            for (size_t k = 0; k < narrowBand.size(); ++k) {
                const int idx = narrowBand[k];
                const int x = idx % GRID_SIZE;
                const int y = (idx / GRID_SIZE) % GRID_SIZE;
                const int z = idx / (GRID_SIZE * GRID_SIZE);
