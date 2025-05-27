#ifndef LEVEL_SET_METHOD_HPP
#define LEVEL_SET_METHOD_HPP

#define CGAL_PMP_USE_CERES_SOLVER
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
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
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

class SpatialScheme;
class UpwindScheme;
class WENOScheme;
class TimeScheme;
class ForwardEulerScheme;
class RungeKutta3Scheme;
class BackwardEulerScheme;
class ImplicitCrankNicolsonScheme;

// Enum for spatial scheme types
enum class SpatialSchemeType {
    UPWIND,
};

// Enum for time scheme types
enum class TimeSchemeType {
    BACKWARD_EULER,  // Implicit method
};

class LevelSetMethod {
public:
    // Constructor that accepts a CSV file for material information
    LevelSetMethod(
                const std::string& meshFile,
                int gridSize = 400, 
                double timeStep = 0.01, 
                int maxSteps = 80, 
                int reinitInterval = 5,
                double curvatureWeight = 0.0,
                int numThreads = -1,
                SpatialSchemeType spatialSchemeType = SpatialSchemeType::UPWIND)
        : GRID_SIZE(gridSize),
        dt(timeStep),
        STEPS(maxSteps),
        REINIT_INTERVAL(reinitInterval),
        CURVATURE_WEIGHT(curvatureWeight){

        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        }
        
        // Load mesh and material information
        loadMesh(meshFile);
        generateGrid();
        phi = initializeSignedDistanceField();
        
        switch (spatialSchemeType) {
            case SpatialSchemeType::UPWIND:
                spatialScheme = std::static_pointer_cast<SpatialScheme>(std::make_shared<UpwindScheme>(gridSize));
                break;
            default:
                spatialScheme = std::static_pointer_cast<SpatialScheme>(std::make_shared<UpwindScheme>(gridSize));
                break;
        }
        
        // Always use Backward Euler scheme for time integration
        backwardEuler = std::make_shared<BackwardEulerScheme>(dt, GRID_SPACING);
    }
    ~LevelSetMethod() = default;
    
    CGAL::Bbox_3 calculateBoundingBox() const;
    bool smoothShape(double smoothingFactor, int iterations);
    bool extractSurfaceMeshCGAL(const std::string& filename, 
        bool smoothSurface, 
        bool refineMesh, 
        bool remeshSurface,
        int smoothingIterations, 
        double targetEdgeLength,
        bool smoothShape,             // New parameter
        double shapeSmoothing,          // New parameter
        int shapeSmoothingIterations);
    void loadMesh(const std::string& filename);
    bool evolve();
    void reinitialize();
    bool exportGridMaterialsToCSV(const std::string& filename);
    void setGridMaterial(const std::string& material, const double zmax, const double zmin) {
        // Input validation
        if (zmax < zmin) {
            throw std::invalid_argument("zmax must be greater than or equal to zmin");
        }
        if (material.empty()) {
            throw std::invalid_argument("material name cannot be empty");
        }
        if (gridMaterials.empty() || grid.empty()) {
            throw std::runtime_error("grid materials or grid points not initialized");
        }

        // Use OpenMP for parallel processing
        #pragma omp parallel for
        for (size_t i = 0; i < gridMaterials.size(); ++i) {
            const double z = grid[i].z();
            // Use inclusive range check and avoid string comparison in tight loop
            if (z <= zmax && z > zmin && phi[i] <= 0.0) {
                gridMaterials[i] = material;
            }
        }
    }
    void setMaterialProperties(const std::string& material, double etchRatio, double lateralRatio){
        materialProperties[material].etchRatio = etchRatio;
        materialProperties[material].lateralRatio = lateralRatio;
        materialProperties[material].name = material;
    }
    void clearMaterialProperties() {
        materialProperties.clear();
    }
    void setSTEPS(int steps){
        STEPS = steps;
    }
    void updateU(){
        Ux = Eigen::VectorXd::Zero(phi.size());
        Uy = Eigen::VectorXd::Zero(phi.size());
        Uz = Eigen::VectorXd::Zero(phi.size());
        
        #pragma omp parallel for schedule(static, 128)
        for (int idx = 0; idx < static_cast<int>(phi.size()); ++idx) {
            std::string material = getMaterialAtPoint(idx);
            Eigen::Vector3d modifiedU_components;
            
            const auto it_mat = materialProperties.find(material);
            if (it_mat != materialProperties.end()) {
                const auto& props = it_mat->second;
                const double lateral_etch = props.lateralRatio * props.etchRatio;
                modifiedU_components << lateral_etch, lateral_etch, props.etchRatio;
            } else {
                modifiedU_components.setZero();
            }
                
            Ux(idx) = -modifiedU_components.x();
            Uy(idx) = -modifiedU_components.y();
            Uz(idx) = -modifiedU_components.z();
        }
    }

private:
    const int GRID_SIZE;
    double GRID_SPACING;
    const double dt;
    int STEPS;
    const int REINIT_INTERVAL;
    const double CURVATURE_WEIGHT;
    double BOX_SIZE = -1.0;
    
    double gridOriginX = 0.0;
    double gridOriginY = 0.0;
    double gridOriginZ = 0.0;
   
    std::shared_ptr<SpatialScheme> spatialScheme;
    std::shared_ptr<BackwardEulerScheme> backwardEuler;

    Mesh mesh;
    std::unique_ptr<AABB_tree> tree;
    std::vector<Point_3> grid;
    Eigen::VectorXd phi;
    Eigen::VectorXd Ux, Uy, Uz;
    
    // Add material related members
    struct MaterialProperties {
        double etchRatio;
        double lateralRatio;
        std::string name;
    };
    
    std::unordered_map<std::string, MaterialProperties> materialProperties;
    std::vector<std::string> gridMaterials; // Store material for each grid point
    
    // Add new methods
    std::string getMaterialAtPoint(int idx) const;
    double computeMeanCurvature(int idx, const Eigen::VectorXd& phi);
    void generateGrid();
    Eigen::VectorXd initializeSignedDistanceField();
    bool isOnBoundary(int idx) const;
    int getIndex(int x, int y, int z) const;
    void updateNarrowBand(); // Empty implementation kept for compatibility

};

struct DerivativeOperator{
    double dxN;
    double dyN;
    double dzN;
    double dxP;
    double dyP;
    double dzP;
};

class SpatialScheme{
    public:
        SpatialScheme(double gridSize): GRID_SIZE(gridSize) {};
        virtual ~SpatialScheme() = default;
    
        inline int getIndex(int x, int y, int z) const {
            static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
            return x + y * GRID_SIZE + z * GRID_SIZE_SQ;
        }
        
        virtual void SpatialSch(int idx, const Eigen::VectorXd& phi, double spacing, DerivativeOperator& Dop) = 0;
    
    protected:
        const int GRID_SIZE;       
};

class UpwindScheme : public SpatialScheme {
    public:
        UpwindScheme(double gridSize) : SpatialScheme(gridSize) {}
        
        void SpatialSch(int idx, const Eigen::VectorXd& phi, double spacing, DerivativeOperator& Dop) override {
            int x = idx % GRID_SIZE;
            int y = (idx / GRID_SIZE) % GRID_SIZE;
            int z = idx / (GRID_SIZE * GRID_SIZE);

            double dxN = computeUpwindDerivativeN(phi, spacing, x, y, z, 0);
            double dyN = computeUpwindDerivativeN(phi, spacing, x, y, z, 1);
            double dzN = computeUpwindDerivativeN(phi, spacing, x, y, z, 2);
            double dxP = computeUpwindDerivativeP(phi, spacing, x, y, z, 0);
            double dyP = computeUpwindDerivativeP(phi, spacing, x, y, z, 1);
            double dzP = computeUpwindDerivativeP(phi, spacing, x, y, z, 2);
            Dop = {dxN, dyN, dzN, dxP, dyP, dzP};
        }

        private:

        std::vector<double> getStencil(const Eigen::VectorXd& phi, int x, int y, int z, int direction) const {
            std::vector<int> stencil;
            if (direction == 0) {
                stencil = {
                    getIndex(x-1, y, z),
                    getIndex(x, y, z),
                    getIndex(x+1, y, z)
                };
            } else if (direction == 1) {
                stencil = {
                    getIndex(x, y-1, z),
                    getIndex(x, y, z),
                    getIndex(x, y+1, z)
                };
            } else {
                stencil = {
                    getIndex(x, y, z-1),
                    getIndex(x, y, z),
                    getIndex(x, y, z+1)
                };
            }
            
            std::vector<double> v(3);
            for (int i = 0; i < 3; i++) {
                v[i] = phi[stencil[i]];
            }
            return v;
        }

        double computeUpwindDerivativeN(const Eigen::VectorXd& phi, double spacing, int x, int y, int z, int direction) const {
            std::vector<double> v = getStencil(phi, x, y, z, direction);
            double forward_derivative = computUpwind(v[0], v[1], v[2], true, spacing);
            double backward_derivative = computUpwind(v[0], v[1], v[2], false, spacing);
            return std::max(forward_derivative, 0.0) + std::min(backward_derivative, 0.0);
        }
        double computeUpwindDerivativeP(const Eigen::VectorXd& phi, double spacing, int x, int y, int z, int direction) const {
            std::vector<double> v = getStencil(phi, x, y, z, direction);
            double forward_derivative = computUpwind(v[0], v[1], v[2], true, spacing);
            double backward_derivative = computUpwind(v[0], v[1], v[2], false, spacing);
            return std::max(backward_derivative, 0.0) + std::min(forward_derivative, 0.0);
        }

        double computUpwind(double v0, double v1, double v2, bool forward, double h) const {
            // Proper upwind scheme implementation
            if (forward) {
                return (v1 - v0) / h; // Backward difference
            } else {
                return (v2 - v1) / h; // Forward difference
            }
        }
};

class TimeScheme {
public:
    TimeScheme(double timeStep, double GRID_SPACING) : dt(timeStep), dx(GRID_SPACING) {}
    virtual ~TimeScheme() = default;
    
    virtual Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                                   const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) = 0;
    
protected:
    const double dt;
    const double dx;
};

class BackwardEulerScheme : public TimeScheme {
public:
    BackwardEulerScheme(double timeStep, double GRID_SPACING = 1.0) : TimeScheme(timeStep, GRID_SPACING) {}
    
    // Standard interface for TimeScheme
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        // This is a placeholder implementation that will never be called
        // The actual implementation is in the specialized version below
        return phi;
    }
    
    // Specialized version for backward Euler with velocity components
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const Eigen::VectorXd& Ux, 
                           const Eigen::VectorXd& Uy, 
                           const Eigen::VectorXd& Uz,
                           const std::shared_ptr<SpatialScheme>& spatialScheme,
                           double spacing,
                           int gridSize) {
        const int n = phi.size();
        
        // Create the system matrix for implicit time stepping
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(7 * n); // Each row has at most 7 non-zero entries (center + 6 neighbors)
        
        // Create right-hand side vector
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
        
        // Small regularization term to improve matrix conditioning
        const double epsilon = 1e-10;
        
        // Build the linear system (I - dt*L)phi^{n+1} = phi^n
        #pragma omp parallel for
        for (int idx = 0; idx < n; idx++) {
            // Get spatial derivatives at this point
            DerivativeOperator Dop;
            spatialScheme->SpatialSch(idx, phi, spacing, Dop);
            
            // Calculate upwind derivatives based on velocity sign
            double dxTerm = (Ux(idx) > 0) ? Ux(idx) * Dop.dxN : Ux(idx) * Dop.dxP;
            double dyTerm = (Uy(idx) > 0) ? Uy(idx) * Dop.dyN : Uy(idx) * Dop.dyP;
            double dzTerm = (Uz(idx) > 0) ? Uz(idx) * Dop.dzN : Uz(idx) * Dop.dzP;
            
            // Total velocity contribution
            double velocityTerm = dxTerm + dyTerm + dzTerm;
            
            // Right-hand side is just the current phi value
            b(idx) = phi(idx);
            
            // Thread-safe insertion into triplet list
            #pragma omp critical
            {
                int x = idx % gridSize;
                int y = (idx / gridSize) % gridSize;
                int z = idx / (gridSize * gridSize);
                
                // Check if this is a boundary point
                bool isBoundary = (x == 0 || x == gridSize-1 || 
                                  y == 0 || y == gridSize-1 || 
                                  z == 0 || z == gridSize-1);
                
                if (isBoundary) {
                    // For boundary points, use identity equation (phi^{n+1} = phi^n)
                    tripletList.push_back(T(idx, idx, 1.0));
                } else {
                    // Diagonal term: 1 + dt*regularization
                    // Note: we're using 1.0 instead of (1.0 + dt * velocityTerm) to avoid potential instability
                    double diagTerm = 1.0;

                    // Add connections to neighboring cells based on velocity direction
                    // X direction
                    if (Ux(idx) <= 0) { // Flow from right to left
                        diagTerm -= dt * Ux(idx) / spacing; 
                        tripletList.push_back(T(idx, idx+1, dt * Ux(idx) / spacing));
                    } else if (Ux(idx) > 0) { // Flow from left to right
                        diagTerm += dt * Ux(idx) / spacing;
                        tripletList.push_back(T(idx, idx-1, dt * Ux(idx) / spacing));
                    }
                    
                    // Y direction
                    if (Uy(idx) <= 0) { // Flow from top to bottom
                        diagTerm -= dt * Uy(idx) / spacing;
                        tripletList.push_back(T(idx, idx+gridSize, dt * Uy(idx) / spacing));
                    } else if (Uy(idx) > 0) { // Flow from bottom to top
                        diagTerm += dt * Uy(idx) / spacing;
                        tripletList.push_back(T(idx, idx-gridSize, dt * Uy(idx) / spacing));
                    }
                    
                    // Z direction
                    if (Uz(idx) <= 0) { // Flow from front to back
                        diagTerm -= dt * Uz(idx) / spacing;
                        tripletList.push_back(T(idx, idx+gridSize*gridSize, dt * Uz(idx) / spacing));
                    } else if (Uz(idx) > 0) { // Flow from back to front
                        diagTerm += dt * Uz(idx) / spacing;
                        tripletList.push_back(T(idx, idx-gridSize*gridSize, dt * Uz(idx) / spacing));
                    }

                    tripletList.push_back(T(idx, idx, diagTerm));
                }
            }
        }
        
        // Create sparse matrix from triplets
        Eigen::SparseMatrix<double> A(n, n);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        
        // Use BiCGSTAB solver which is more robust for non-symmetric matrices
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
        
        // Configure solver for better robustness
        solver.setMaxIterations(1000);
        solver.setTolerance(1e-6);
        
        // Compute the preconditioner
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Matrix decomposition failed with error: " << solver.error() << std::endl;
            throw std::runtime_error("Decomposition failed");
        }
        
        // Solve the system
        Eigen::VectorXd phi_next = solver.solve(b);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed with error: " << solver.error() << std::endl;
            std::cerr << "Iterations: " << solver.iterations() << ", estimated error: " << solver.error() << std::endl;
            throw std::runtime_error("Solver failed");
        }
        
        return phi_next;
    }
};

#endif // LEVEL_SET_METHOD_HPP