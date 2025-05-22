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
    ENO,
    WENO
};

// Enum for time scheme types
enum class TimeSchemeType {
    FORWARD_EULER,
    RUNGE_KUTTA_3,
    BACKWARD_EULER,  // Implicit method
    CRANK_NICOLSON   // Second-order implicit method
};

class LevelSetMethod {
public:
    // Constructor that accepts a CSV file for material information
    LevelSetMethod(
                const std::string& meshFile,
                const std::string& orgFile,
                const std::string& materialCsvFile,
                int gridSize = 400, 
                double timeStep = 0.01, 
                int maxSteps = 80, 
                int reinitInterval = 5,
                int narrowBandInterval = 100,
                double narrowBandWidth = 10.0,
                double curvatureWeight = 0.0,
                int numThreads = -1,
                SpatialSchemeType spatialSchemeType = SpatialSchemeType::UPWIND,
                TimeSchemeType timeSchemeType = TimeSchemeType::FORWARD_EULER)
        : GRID_SIZE(gridSize),
        dt(timeStep),
        STEPS(maxSteps),
        REINIT_INTERVAL(reinitInterval),
        NARROW_BAND_UPDATE_INTERVAL(narrowBandInterval),
        NARROW_BAND_WIDTH(narrowBandWidth),
        CURVATURE_WEIGHT(curvatureWeight){

        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        }
        
        // Load mesh and material information
        loadMesh(meshFile);
        generateGrid();
        loadMaterialInfo(materialCsvFile, orgFile);
        phi = initializeSignedDistanceField();
        
        switch (spatialSchemeType) {
            case SpatialSchemeType::UPWIND:
                spatialScheme = std::static_pointer_cast<SpatialScheme>(std::make_shared<UpwindScheme>(gridSize));
                break;
            case SpatialSchemeType::WENO:
                spatialScheme = std::static_pointer_cast<SpatialScheme>(std::make_shared<WENOScheme>(gridSize));
                break;
            default:
                spatialScheme = std::static_pointer_cast<SpatialScheme>(std::make_shared<UpwindScheme>(gridSize));
                break;
        }
        
        switch (timeSchemeType) {
            case TimeSchemeType::FORWARD_EULER:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<ForwardEulerScheme>(dt));
                break;
            case TimeSchemeType::RUNGE_KUTTA_3:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<RungeKutta3Scheme>(dt));
                break;
            case TimeSchemeType::BACKWARD_EULER:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<BackwardEulerScheme>(dt));
                break;
            case TimeSchemeType::CRANK_NICOLSON:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<ImplicitCrankNicolsonScheme>(dt));
                break;
            default:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<ForwardEulerScheme>(dt));
                break;
        }
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

private:
    const int GRID_SIZE;
    double GRID_SPACING;
    const double dt;
    int STEPS;
    const int REINIT_INTERVAL;
    const int NARROW_BAND_UPDATE_INTERVAL;
    const double NARROW_BAND_WIDTH;
    const double CURVATURE_WEIGHT;
    double BOX_SIZE = -1.0;
    
    double gridOriginX = 0.0;
    double gridOriginY = 0.0;
    double gridOriginZ = 0.0;
   
    std::shared_ptr<SpatialScheme> spatialScheme;
    std::shared_ptr<TimeScheme> timeScheme;

    Mesh mesh;
    std::unique_ptr<AABB_tree> tree;
    std::vector<Point_3> grid;
    Eigen::VectorXd phi;
    std::vector<int> narrowBand;
    
    // Add material related members
    struct MaterialProperties {
        double etchRatio;
        double lateralRatio;
        std::string name;
    };
    
    std::unordered_map<std::string, MaterialProperties> materialProperties;
    std::vector<std::string> gridMaterials; // Store material for each grid point
    
    // Add new methods
    void loadMaterialInfo(const std::string& csvFilename, const std::string& meshFilename);
    double computeEtchingRate(const std::string& material, const Eigen::Vector3d& normal);
    std::string getMaterialAtPoint(int idx) const;

    double computeEtchingRate(const Eigen::Vector3d& normal, double sigma);
    double computeMeanCurvature(int idx, const Eigen::VectorXd& phi);
    void updateNarrowBand();
    void generateGrid();
    Eigen::VectorXd initializeSignedDistanceField();
    bool isOnBoundary(int idx) const;
    int getIndex(int x, int y, int z) const;
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

class WENOScheme : public SpatialScheme {
public:
    WENOScheme(double gridSize) : SpatialScheme(gridSize) {}
    
    void SpatialSch(int idx, const Eigen::VectorXd& phi, double spacing, DerivativeOperator& Dop) override {
        int x = idx % GRID_SIZE;
        int y = (idx / GRID_SIZE) % GRID_SIZE;
        int z = idx / (GRID_SIZE * GRID_SIZE);

        double dxN = computeWENODerivativeN(phi, spacing, x, y, z, 0);
        double dyN = computeWENODerivativeN(phi, spacing, x, y, z, 1);
        double dzN = computeWENODerivativeN(phi, spacing, x, y, z, 2);
        double dxP = computeWENODerivativeP(phi, spacing, x, y, z, 0);
        double dyP = computeWENODerivativeP(phi, spacing, x, y, z, 1);
        double dzP = computeWENODerivativeP(phi, spacing, x, y, z, 2);
        Dop = {dxN, dyN, dzN, dxP, dyP, dzP};
    }

private:
    std::vector<double> getWideStencil(const Eigen::VectorXd& phi, int x, int y, int z, int direction) const {
        std::vector<int> stencil;
        if (direction == 0) {
            stencil = {
                getIndex(x-2, y, z),
                getIndex(x-1, y, z),
                getIndex(x, y, z),
                getIndex(x+1, y, z),
                getIndex(x+2, y, z),
                getIndex(x+3, y, z), 
            };
        } else if (direction == 1) {
            stencil = {
                getIndex(x, y-2, z),
                getIndex(x, y-1, z),
                getIndex(x, y, z),
                getIndex(x, y+1, z),
                getIndex(x, y+2, z),
                getIndex(x, y+3, z)
            };
        } else {
            stencil = {
                getIndex(x, y, z-2),
                getIndex(x, y, z-1),
                getIndex(x, y, z),
                getIndex(x, y, z+1),
                getIndex(x, y, z+2),
                getIndex(x, y, z+3)
            };
        }
        
        std::vector<double> v(6);
        for (int i = 0; i < 6; i++) {
            v[i] = phi[stencil[i]];
        }
        return v;
    }

    double computeWENODerivativeN(const Eigen::VectorXd& phi, double spacing, int x, int y, int z, int direction) const {
        std::vector<double> v = getWideStencil(phi, x, y, z, direction);
        double forward_derivative = computeWENO(v, true, spacing);
        double backward_derivative = computeWENO(v, false, spacing);
        return std::max(forward_derivative, 0.0) + std::min(backward_derivative, 0.0);
    }
    
    double computeWENODerivativeP(const Eigen::VectorXd& phi, double spacing, int x, int y, int z, int direction) const {
        std::vector<double> v = getWideStencil(phi, x, y, z, direction);
        double forward_derivative = computeWENO(v, true, spacing);
        double backward_derivative = computeWENO(v, false, spacing);
        return std::max(backward_derivative, 0.0) + std::min(forward_derivative, 0.0);
    }

    double computeWENO(const std::vector<double>& v, bool forward, double h) const {
        const double eps = 1e-6; // Small value to avoid division by zero
        
        if (forward) {
            double beta0 = 13.0/12.0 * std::pow(v[0] - 2.0*v[1] + v[2], 2) + 1.0/4.0 * std::pow(v[0] - 4.0*v[1] + v[2], 2);
            double beta1 = 13.0/12.0 * std::pow(v[1] - 2.0*v[2] + v[3], 2) + 1.0/4.0 * std::pow(v[1] - v[3], 2);
            double beta2 = 13.0/12.0 * std::pow(v[2] - 2.0*v[3] + v[4], 2) + 1.0/4.0 * std::pow(3.0*v[2] - 4.0*v[3] + v[4], 2);
            
            double alpha0 = 0.1 / std::pow(eps + beta0, 2);
            double alpha1 = 0.6 / std::pow(eps + beta1, 2);
            double alpha2 = 0.3 / std::pow(eps + beta2, 2);
            
            double sum_alpha = alpha0 + alpha1 + alpha2;
            double w0 = alpha0 / sum_alpha;
            double w1 = alpha1 / sum_alpha;
            double w2 = alpha2 / sum_alpha;
            
            double q0 = (2.0*v[0] - 7.0*v[1] + 11.0*v[2]) / 6.0;
            double q1 = (-v[1] + 5.0*v[2] + 2.0*v[3]) / 6.0;
            double q2 = (2.0*v[2] + 5.0*v[3] - v[4]) / 6.0;
            
            double derivative = (w0 * q0 + w1 * q1 + w2 * q2) / h;
            
            return derivative;
        } else {
            double beta0 = 13.0/12.0 * std::pow(v[1] - 2.0*v[2] + v[3], 2) + 1.0/4.0 * std::pow(v[1] - 4.0*v[2] + 3.0*v[3], 2);
            double beta1 = 13.0/12.0 * std::pow(v[2] - 2.0*v[3] + v[4], 2) + 1.0/4.0 * std::pow(v[2] - v[4], 2);
            double beta2 = 13.0/12.0 * std::pow(v[3] - 2.0*v[4] + v[5], 2) + 1.0/4.0 * std::pow(v[3] - 4.0*v[4] + v[5], 2);
            
            double alpha0 = 0.1 / std::pow(eps + beta0, 2);
            double alpha1 = 0.6 / std::pow(eps + beta1, 2);
            double alpha2 = 0.3 / std::pow(eps + beta2, 2);
            
            double sum_alpha = alpha0 + alpha1 + alpha2;
            double w0 = alpha0 / sum_alpha;
            double w1 = alpha1 / sum_alpha;
            double w2 = alpha2 / sum_alpha;
            
            double q0 = (-1.0*v[1] + 5.0*v[2] + 2.0*v[3]) / 6.0;
            double q1 = (2.0*v[2] + 5.0*v[3] - 1.0*v[4]) / 6.0;
            double q2 = (11.0*v[3] - 7.0*v[4] + 2.0*v[5]) / 6.0;
            
            double derivative = (w0 * q0 + w1 * q1 + w2 * q2) / h;
            
            return derivative;
        }
    }
};

class TimeScheme {
public:
    TimeScheme(double timeStep) : dt(timeStep) {}
    virtual ~TimeScheme() = default;
    
    virtual Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                                   const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) = 0;
    
protected:
    const double dt;
};

class ForwardEulerScheme : public TimeScheme {
public:
    ForwardEulerScheme(double timeStep) : TimeScheme(timeStep) {}
    
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        return phi + dt * L(phi);
    }
};

class RungeKutta3Scheme : public TimeScheme {
public:
    RungeKutta3Scheme(double timeStep) : TimeScheme(timeStep) {}
    
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        Eigen::VectorXd k1 = L(phi);
        Eigen::VectorXd phi1 = phi + dt * k1/2;
        
        Eigen::VectorXd k2 = L(phi1);
        Eigen::VectorXd phi2 = phi + dt * (-k1 + 2 * k2);
        
        Eigen::VectorXd k3 = L(phi2);
        return phi + dt * (k1 + 4 * k2 + k3) / 6;
    }
};

class BackwardEulerScheme : public TimeScheme {
public:
    BackwardEulerScheme(double timeStep, double tolerance = 1e-6, int maxIterations = 100)
        : TimeScheme(timeStep), tol(tolerance), maxIter(maxIterations) {}
    
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        // Backward Euler: phi^{n+1} = phi^n + dt * L(phi^{n+1})
        // We can solve this using two approaches:
        // 1. Fixed-point iteration (simpler but may converge slowly)
        // 2. Newton's method with sparse linear solver (faster convergence but more complex)
        
        // For level set methods, fixed-point iteration with relaxation often works well
        // and avoids the need to compute Jacobians
        
        const size_t n = phi.size();
        Eigen::VectorXd phi_next = phi;  // Initial guess
        
        // Implement fixed-point iteration with adaptive relaxation
        double relaxation = 0.8;  // Initial relaxation factor
        double prev_residual = std::numeric_limits<double>::max();
        
        for (int iter = 0; iter < maxIter; ++iter) {
            // Compute the operator at current solution estimate
            Eigen::VectorXd L_phi_next = L(phi_next);
            
            // Compute the residual: r = phi_next - (phi + dt*L(phi_next))
            Eigen::VectorXd rhs = phi + dt * L_phi_next;
            Eigen::VectorXd residual_vec = phi_next - rhs;
            double residual_norm = residual_vec.norm() / std::max(1.0, phi_next.norm());
            
            // Print convergence information every few iterations
            std::cout << "Iteration " << iter << ", residual: " << residual_norm << std::endl;
            
            // Check for convergence
            if (residual_norm < tol) {
                std::cout << "Backward Euler converged in " << iter << " iterations" << std::endl;
                break;
            }
            
            // Adaptive relaxation - increase if converging, decrease if diverging
            if (iter > 0) {
                if (residual_norm < prev_residual) {
                    // Converging, can slightly increase relaxation
                    relaxation = std::min(0.95, relaxation * 1.05);
                } else {
                    // Diverging, decrease relaxation
                    relaxation = std::max(0.2, relaxation * 0.7);
                }
            }
            prev_residual = residual_norm;
            
            // Update solution with relaxation
            phi_next = phi_next - relaxation * residual_vec;
            
        }
        
        return phi_next;
    }
    
private:
    const double tol;         // Convergence tolerance
    const int maxIter;        // Maximum number of iterations
};

// Alternative implementation using a sparse linear solver approach
class ImplicitCrankNicolsonScheme : public TimeScheme {
public:
    ImplicitCrankNicolsonScheme(double timeStep, double tolerance = 1e-6, int maxIterations = 100)
        : TimeScheme(timeStep), tol(tolerance), maxIter(maxIterations) {}
    
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        // Crank-Nicolson: phi^{n+1} = phi^n + 0.5*dt*(L(phi^n) + L(phi^{n+1}))
        // This is second-order accurate in time
        
        const size_t n = phi.size();
        Eigen::VectorXd phi_next = phi;  // Initial guess
        
        // Compute the explicit part once
        Eigen::VectorXd L_phi = L(phi);
        Eigen::VectorXd explicit_part = phi + 0.5 * dt * L_phi;
        
        // Iterative solution for the implicit part
        for (int iter = 0; iter < maxIter; ++iter) {
            // Compute the operator at current solution estimate
            Eigen::VectorXd L_phi_next = L(phi_next);
            
            // Compute the residual: r = phi_next - (explicit_part + 0.5*dt*L(phi_next))
            Eigen::VectorXd rhs = explicit_part + 0.5 * dt * L_phi_next;
            Eigen::VectorXd residual_vec = phi_next - rhs;
            double residual_norm = residual_vec.norm() / std::max(1.0, phi_next.norm());
            
            // Check for convergence
            if (residual_norm < tol) {
                std::cout << "Crank-Nicolson converged in " << iter << " iterations" << std::endl;
                break;
            }
            
            // Update solution with relaxation
            const double relaxation = 0.7;  // Relaxation factor for Crank-Nicolson
            phi_next = phi_next - relaxation * residual_vec;
        }
        
        return phi_next;
    }
    
private:
    const double tol;         // Convergence tolerance
    const int maxIter;        // Maximum number of iterations
};

#endif // LEVEL_SET_METHOD_HPP