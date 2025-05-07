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
#include <functional>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mutex>

namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

class SpatialScheme;
class UpwindScheme;
class ENOScheme;
class WENOScheme;
class TimeScheme;
class ForwardEulerScheme;
class BackwardEulerScheme;
class CrankNicolsonScheme;
class RungeKutta3Scheme;

// Enum for spatial scheme types
enum class SpatialSchemeType {
    UPWIND,
    ENO,
    WENO
};

// Enum for time scheme types
enum class TimeSchemeType {
    FORWARD_EULER,
    BACKWARD_EULER,
    CRANK_NICOLSON,
    RUNGE_KUTTA_3
};

class LevelSetMethod {
public:
    LevelSetMethod(const std::string& filename,
                int gridSize = 400, 
                double timeStep = 0.01, 
                int maxSteps = 80, 
                int reinitInterval = 5,
                int narrowBandInterval = 100,
                double narrowBandWidth = 10.0,
                int numThreads = -1,
                double curvatureWeight = 0.0,
                Eigen::Vector3d U = Eigen::Vector3d(-0.01, -0.01, -1.0),
                SpatialSchemeType spatialSchemeType = SpatialSchemeType::UPWIND,
                TimeSchemeType timeSchemeType = TimeSchemeType::FORWARD_EULER)
        : GRID_SIZE(gridSize),
        dt(timeStep),
        STEPS(maxSteps),
        REINIT_INTERVAL(reinitInterval),
        NARROW_BAND_UPDATE_INTERVAL(narrowBandInterval),
        NARROW_BAND_WIDTH(narrowBandWidth),
        CURVATURE_WEIGHT(curvatureWeight),
        U(U) {

        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        }
        loadMesh(filename);
        generateGrid();
       
        switch (spatialSchemeType) {
            case SpatialSchemeType::UPWIND:
                spatialScheme = std::static_pointer_cast<SpatialScheme>(std::make_shared<UpwindScheme>(gridSize));
                break;
            // case SpatialSchemeType::WENO:
                // spatialScheme = std::static_pointer_cast<SpatialScheme>(std::make_shared<WENOScheme>(gridSize));
                // break;
            default:
                spatialScheme = std::static_pointer_cast<SpatialScheme>(std::make_shared<UpwindScheme>(gridSize));
                break;
        }
        
        switch (timeSchemeType) {
            case TimeSchemeType::FORWARD_EULER:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<ForwardEulerScheme>(dt));
                break;
            case TimeSchemeType::BACKWARD_EULER:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<BackwardEulerScheme>(dt));
                break;
            case TimeSchemeType::CRANK_NICOLSON:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<CrankNicolsonScheme>(dt));
                break;
            case TimeSchemeType::RUNGE_KUTTA_3:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<RungeKutta3Scheme>(dt));
                break;
            default:
                timeScheme = std::static_pointer_cast<TimeScheme>(std::make_shared<ForwardEulerScheme>(dt));
                break;
        }
    }
    ~LevelSetMethod() = default;
    
    CGAL::Bbox_3 calculateBoundingBox() const;
    bool extractSurfaceMeshCGAL(const std::string& filename);
    void loadMesh(const std::string& filename);
    bool evolve();
    void reinitialize();

private:
    Eigen::Vector3d U;
    const int GRID_SIZE;
    double GRID_SPACING;
    const double dt;
    const int STEPS;
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

class BackwardEulerScheme : public TimeScheme {
public:
    BackwardEulerScheme(double timeStep) : TimeScheme(timeStep) {}
    
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        // Simple fixed-point iteration for implicit scheme
        // phi^(n+1) = phi^n + dt * L(phi^(n+1))
        const int MAX_ITERATIONS = 10;
        const double TOLERANCE = 1e-6;
        
        Eigen::VectorXd phi_new = phi;
        Eigen::VectorXd phi_prev;
        
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            phi_prev = phi_new;
            phi_new = phi + dt * L(phi_prev);
            
            // Check convergence with safeguard against division by small numbers
            double phi_new_norm = phi_new.norm();
            double diff_norm = (phi_new - phi_prev).norm();
            
            // Use absolute and relative error for more robust convergence check
            if (diff_norm < TOLERANCE || (phi_new_norm > 1e-10 && diff_norm / phi_new_norm < TOLERANCE)) {
                break;
            }
        }
        
        return phi_new;
    }
};

class CrankNicolsonScheme : public TimeScheme {
public:
    CrankNicolsonScheme(double timeStep) : TimeScheme(timeStep) {}
    
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        // Crank-Nicolson scheme: phi^(n+1) = phi^n + dt/2 * (L(phi^n) + L(phi^(n+1)))
        const int MAX_ITERATIONS = 10;
        const double TOLERANCE = 1e-6;
        
        Eigen::VectorXd L_phi_n = L(phi);
        Eigen::VectorXd phi_new = phi + dt * L_phi_n; // Initial guess using explicit Euler
        Eigen::VectorXd phi_prev;
        
        for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
            phi_prev = phi_new;
            Eigen::VectorXd L_phi_new = L(phi_prev);
            phi_new = phi + (dt/2.0) * (L_phi_n + L_phi_new);
            
            // Check convergence with safeguard against division by small numbers
            double phi_new_norm = phi_new.norm();
            double diff_norm = (phi_new - phi_prev).norm();
            
            // Use absolute and relative error for more robust convergence check
            if (diff_norm < TOLERANCE || (phi_new_norm > 1e-10 && diff_norm / phi_new_norm < TOLERANCE)) {
                break;
            }
        }
        
        return phi_new;
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

#endif // LEVEL_SET_METHOD_HPP