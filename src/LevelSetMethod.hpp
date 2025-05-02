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
        precomputeDirections(5, 10);
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
    
    std::vector<Eigen::Vector3d> precomputed_directions;
    std::vector<double> precomputed_dOmega;

    // Private methods
    void precomputeDirections(int num_theta, int num_phi);
    double computeEtchingRate(const Eigen::Vector3d& normal, double sigma);
    void updateNarrowBand();
    void generateGrid();
    Eigen::VectorXd initializeSignedDistanceField();
    bool isOnBoundary(int idx) const;
    int getIndex(int x, int y, int z) const;
};


#endif // LEVEL_SET_METHOD_HPP