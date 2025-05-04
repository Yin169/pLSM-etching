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
                SpatialSchemeType spatialSchemeType = SpatialSchemeType::UPWIND,
                TimeSchemeType timeSchemeType = TimeSchemeType::FORWARD_EULER)
        : GRID_SIZE(gridSize),
        dt(timeStep),
        STEPS(maxSteps),
        REINIT_INTERVAL(reinitInterval),
        NARROW_BAND_UPDATE_INTERVAL(narrowBandInterval),
        NARROW_BAND_WIDTH(narrowBandWidth) {

        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        }
        loadMesh(filename);
        generateGrid();
       
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
    const int GRID_SIZE;
    double GRID_SPACING;
    const double dt;
    const int STEPS;
    const int REINIT_INTERVAL;
    const int NARROW_BAND_UPDATE_INTERVAL;
    const double NARROW_BAND_WIDTH;
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
    void updateNarrowBand();
    void generateGrid();
    Eigen::VectorXd initializeSignedDistanceField();
    bool isOnBoundary(int idx) const;
    int getIndex(int x, int y, int z) const;
};


class SpatialScheme{
    public:
        SpatialScheme(double gridSize): GRID_SIZE(gridSize) {};
        virtual ~SpatialScheme() = default;
    
        inline int getIndex(int x, int y, int z) const {
            static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
            return x + y * GRID_SIZE + z * GRID_SIZE_SQ;
        }
        
        virtual void SpatialSch(int idx, const Eigen::VectorXd& phi, double spacing, double& dx, double& dy, double& dz) = 0;
    
    protected:
        const int GRID_SIZE;       
};

class UpwindScheme : public SpatialScheme {
    public:
        UpwindScheme(double gridSize) : SpatialScheme(gridSize) {}
        
        void SpatialSch(int idx, const Eigen::VectorXd& phi, double spacing, double& dx, double& dy, double& dz) override {
            int x = idx % GRID_SIZE;
            int y = (idx / GRID_SIZE) % GRID_SIZE;
            int z = idx / (GRID_SIZE * GRID_SIZE);

            dx = computeUpwindDerivative(phi, spacing, x, y, z, 0);
            dy = computeUpwindDerivative(phi, spacing, x, y, z, 1);
            dz = computeUpwindDerivative(phi, spacing, x, y, z, 2);
        }

        private:
        double computeUpwindDerivative(const Eigen::VectorXd& phi, double spacing, int x, int y, int z, int direction) const {
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
            
            double backward_derivative = computUpwind(v[0], v[1], v[2], true, spacing);
            double forward_derivative = computUpwind(v[1], v[2], v[0], false, spacing);
            
            return std::max(backward_derivative, 0.0) + std::min(forward_derivative, 0.0);
        }

        double computUpwind(double v0, double v1, double v2,bool forward, double h) const {
            return (v1 - v0)/h; 
        }
};

class WENOScheme : public SpatialScheme {
    public:
        WENOScheme(double gridSize) : SpatialScheme(gridSize) {}
        
        void SpatialSch(int idx, const Eigen::VectorXd& phi, double spacing, double& dx, double& dy, double& dz) override {
            int x = idx % GRID_SIZE;
            int y = (idx / GRID_SIZE) % GRID_SIZE;
            int z = idx / (GRID_SIZE * GRID_SIZE);
            
            dx = computeWENODerivative(phi, spacing, x, y, z, 0);
            dy = computeWENODerivative(phi, spacing, x, y, z, 1);
            dz = computeWENODerivative(phi, spacing, x, y, z, 2);
        }
        
    private:
        double computeWENODerivative(const Eigen::VectorXd& phi, double spacing, int x, int y, int z, int direction) const {
            std::vector<int> stencil;
            if (direction == 0) {
                stencil = {
                    getIndex(x-2, y, z),
                    getIndex(x-1, y, z),
                    getIndex(x, y, z),
                    getIndex(x+1, y, z),
                    getIndex(x+2, y, z),
                };
            } else if (direction == 1) {
                stencil = {
                    getIndex(x, y-2, z),
                    getIndex(x, y-1, z),
                    getIndex(x, y, z),
                    getIndex(x, y+1, z),
                    getIndex(x, y+2, z),
                };
            } else {
                stencil = {
                    getIndex(x, y, z-2),
                    getIndex(x, y, z-1),
                    getIndex(x, y, z),
                    getIndex(x, y, z+1),
                    getIndex(x, y, z+2),
                };
            }
            
            std::vector<double> v(7);
            for (int i = 0; i < 5; i++) {
                v[i] = phi[stencil[i]];
            }
            
            return computeWENO3(v[0], v[1], v[2], v[3], v[4], true, spacing);
        }
        
        double computeWENO3(double v0, double v1, double v2, double v3, double v4, double v5, double v6, 
                            bool forward, double h) const {
            const double eps = 1e-6;
            
            double beta0 = 13.0/12.0 * std::pow(v0 - 2.0*v1 + v2, 2) + 
                          1.0/4.0 * std::pow(v0 - 4.0*v1 + v2, 2);
            
            double beta1 = 13.0/12.0 * std::pow(v1 - 2.0*v2 + v3, 2) + 
                          1.0/4.0 * std::pow(v1 - v3, 2);
            
            double beta2 = 13.0/12.0 * std::pow(v2 - 2.0*v3 + v4, 2) + 
                          1.0/4.0 * std::pow(v2 - 4.0*v3 + v4, 2);
            
            double alpha0 = 0.1 / std::pow(eps + beta0, 2);
            double alpha1 = 0.6 / std::pow(eps + beta1, 2);
            double alpha2 = 0.3 / std::pow(eps + beta2, 2);
            
            double sum_alpha = alpha0 + alpha1 + alpha2;
            double w0 = alpha0 / sum_alpha;
            double w1 = alpha1 / sum_alpha;
            double w2 = alpha2 / sum_alpha;
            
            double q0 = (2.0*v0 - 7.0*v1 + 11.0*v2) / 6.0;
            double q1 = (-v1 + 5.0*v2 + 2.0*v3) / 6.0;
            double q2 = (2.0*v2 + 5.0*v3 - v4) / 6.0;
            
            double derivative = (w0 * q0 + w1 * q1 + w2 * q2) / h;
            
            return derivative;
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
        Eigen::VectorXd phi2 = phi + dt * (- k1 + 2 * k2);
        
        Eigen::VectorXd k3 = L(phi2);
        return phi + dt * (k1 + 4 * k2 + k3) / 6;
    }
};

#endif // LEVEL_SET_METHOD_HPP