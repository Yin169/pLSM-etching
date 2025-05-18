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
// #include <CGAL/IO/polygon_mesh_io.h> // Already included via PMP

#include <CGAL/Surface_mesh_default_triangulation_3.h>
// #include <CGAL/Complex_2_in_triangulation_3.h> // Already included
// #include <CGAL/make_surface_mesh.h> // Already included
// #include <CGAL/Surface_mesh.h> // Already included

#include <eigen3/Eigen/Dense>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mutex> // For materialPropertiesMutex if needed, though current usage might not require shared_mutex
#include <shared_mutex> // Keep if complex concurrent read/write patterns for materialProperties are planned
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <atomic> // For progress counters
#include <execution> // For std::sort(std::execution::par, ...)

namespace PMP = CGAL::Polygon_mesh_processing;

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;

// Forward declarations
class SpatialScheme;
class UpwindScheme;
class WENOScheme;
class TimeScheme;
class ForwardEulerScheme;
class RungeKutta3Scheme;

// Enum for spatial scheme types
enum class SpatialSchemeType {
    UPWIND,
    // ENO, // ENO was declared as a comment but not used. Removed for clarity unless planned.
    WENO
};

// Enum for time scheme types
enum class TimeSchemeType {
    FORWARD_EULER,
    RUNGE_KUTTA_3
};

struct DerivativeOperator{
    double dxN; // Negative part of x-derivative
    double dyN; // Negative part of y-derivative
    double dzN; // Negative part of z-derivative
    double dxP; // Positive part of x-derivative
    double dyP; // Positive part of y-derivative
    double dzP; // Positive part of z-derivative
};

class SpatialScheme{
    public:
        SpatialScheme(int gridSize): GRID_SIZE(gridSize) {}; // Changed double to int for GRID_SIZE
        virtual ~SpatialScheme() = default;
    
        // Helper to get 1D index from 3D coordinates
        inline int getIndex(int x, int y, int z) const {
            // Clamp coordinates to be within grid boundaries to prevent out-of-bounds access
            // This is a common strategy for stencil operations near boundaries.
            // Alternatively, ensure calling code handles boundary conditions appropriately.
            x = std::max(0, std::min(x, GRID_SIZE - 1));
            y = std::max(0, std::min(y, GRID_SIZE - 1));
            z = std::max(0, std::min(z, GRID_SIZE - 1));
            return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
        }
        
        // Pure virtual function to compute spatial derivatives
        virtual void SpatialSch(int idx, const Eigen::VectorXd& phi, double spacing, DerivativeOperator& Dop) = 0;
    
    protected:
        const int GRID_SIZE; // GRID_SIZE should be an integer       
};

class UpwindScheme : public SpatialScheme {
    public:
        UpwindScheme(int gridSize) : SpatialScheme(gridSize) {} // Changed double to int
        
        void SpatialSch(int idx, const Eigen::VectorXd& phi, double spacing, DerivativeOperator& Dop) override {
            const int x = idx % GRID_SIZE;
            const int y = (idx / GRID_SIZE) % GRID_SIZE;
            const int z = idx / (GRID_SIZE * GRID_SIZE);

            // For upwind, dxN is typically (phi_i - phi_i-1)/h and dxP is (phi_i+1 - phi_i)/h
            // The computeUpwindDerivativeN/P methods seem to mix these based on velocity sign,
            // which is then handled in the advection term calculation.
            // Let's simplify: dxN = backward diff, dxP = forward diff.
            
            Dop.dxN = (phi[getIndex(x, y, z)] - phi[getIndex(x-1, y, z)]) / spacing;
            Dop.dxP = (phi[getIndex(x+1, y, z)] - phi[getIndex(x, y, z)]) / spacing;

            Dop.dyN = (phi[getIndex(x, y, z)] - phi[getIndex(x, y-1, z)]) / spacing;
            Dop.dyP = (phi[getIndex(x, y+1, z)] - phi[getIndex(x, y, z)]) / spacing;

            Dop.dzN = (phi[getIndex(x, y, z)] - phi[getIndex(x, y, z-1)]) / spacing;
            Dop.dzP = (phi[getIndex(x, y, z+1)] - phi[getIndex(x, y, z)]) / spacing;
        }
};

class WENOScheme : public SpatialScheme {
public:
    WENOScheme(int gridSize) : SpatialScheme(gridSize) {} // Changed double to int
    
    void SpatialSch(int idx, const Eigen::VectorXd& phi, double spacing, DerivativeOperator& Dop) override {
        const int x = idx % GRID_SIZE;
        const int y = (idx / GRID_SIZE) % GRID_SIZE;
        const int z = idx / (GRID_SIZE * GRID_SIZE);

        // WENO computes left-biased (dxN) and right-biased (dxP) derivatives
        Dop.dxN = computeWENO(phi, x, y, z, 0, true, spacing);  // true for left-biased (negative direction)
        Dop.dxP = computeWENO(phi, x, y, z, 0, false, spacing); // false for right-biased (positive direction)

        Dop.dyN = computeWENO(phi, x, y, z, 1, true, spacing);
        Dop.dyP = computeWENO(phi, x, y, z, 1, false, spacing);

        Dop.dzN = computeWENO(phi, x, y, z, 2, true, spacing);
        Dop.dzP = computeWENO(phi, x, y, z, 2, false, spacing);
    }

protected:
    // Gets the 6-point stencil v_i-2, v_i-1, v_i, v_i+1, v_i+2, v_i+3 for point i
    // The point 'idx' (x,y,z) is considered v_i in this context for derivative calculation.
    std::vector<double> getWideStencil(const Eigen::VectorXd& phi, int x, int y, int z, int direction) const {
        std::vector<double> v(6);
        if (direction == 0) { // X-direction
            for(int i = 0; i < 6; ++i) v[i] = phi[getIndex(x + i - 2, y, z)];
        } else if (direction == 1) { // Y-direction
            for(int i = 0; i < 6; ++i) v[i] = phi[getIndex(x, y + i - 2, z)];
        } else { // Z-direction
            for(int i = 0; i < 6; ++i) v[i] = phi[getIndex(x, y, z + i - 2)];
        }
        return v;
    }
    
    // Computes WENO derivative. 'isLeftBiased' true for D^- (uses v[0] to v[4]), false for D^+ (uses v[1] to v[5])
    // This is a common 5th order WENO-JS formulation.
    double computeWENO(const Eigen::VectorXd& phi_vec, int x, int y, int z, int direction, bool isLeftBiased, double h) const {
        std::vector<double> v = getWideStencil(phi_vec, x, y, z, direction); // v_i-2, v_i-1, v_i, v_i+1, v_i+2, v_i+3
        
        // Coefficients for 5th order WENO (Jiang and Shu, 1996)
        // For D_i^- (left-biased derivative at point i, using stencil points i-2, i-1, i, i+1, i+2)
        // For D_i^+ (right-biased derivative at point i, using stencil points i-1, i, i+1, i+2, i+3)

        // Constants for ideal weights
        constexpr double c0 = 1.0/10.0, c1 = 6.0/10.0, c2 = 3.0/10.0; // For D_i^- (alpha_0, alpha_1, alpha_2)
        constexpr double c0_p = 3.0/10.0, c1_p = 6.0/10.0, c2_p = 1.0/10.0; // For D_i^+ (alpha_0, alpha_1, alpha_2)

        constexpr double epsilon = 1e-6; // Small value to avoid division by zero and maintain stability

        double p0, p1, p2; // Candidate polynomial reconstructions
        double beta0, beta1, beta2; // Smoothness indicators

        if (isLeftBiased) { // Computes D_i^-, uses v[0]..v[4] which are phi_{i-2}..phi_{i+2}
                            // Stencil S0: {phi_{i-2}, phi_{i-1}, phi_i} -> v[0], v[1], v[2]
                            // Stencil S1: {phi_{i-1}, phi_i, phi_{i+1}} -> v[1], v[2], v[3]
                            // Stencil S2: {phi_i, phi_{i+1}, phi_{i+2}} -> v[2], v[3], v[4]
            p0 = (2.0*v[0] - 7.0*v[1] + 11.0*v[2]) / 6.0;
            p1 = (-v[1] + 5.0*v[2] + 2.0*v[3]) / 6.0;
            p2 = (2.0*v[2] + 5.0*v[3] - v[4]) / 6.0;

            beta0 = (13.0/12.0) * std::pow(v[0] - 2.0*v[1] + v[2], 2) + (1.0/4.0) * std::pow(v[0] - 4.0*v[1] + 3.0*v[2], 2);
            beta1 = (13.0/12.0) * std::pow(v[1] - 2.0*v[2] + v[3], 2) + (1.0/4.0) * std::pow(v[1] - v[3], 2); // Original (v[1] - v[3])^2
            beta2 = (13.0/12.0) * std::pow(v[2] - 2.0*v[3] + v[4], 2) + (1.0/4.0) * std::pow(3.0*v[2] - 4.0*v[3] + v[4], 2);
        } else { // Computes D_i^+, uses v[1]..v[5] which are phi_{i-1}..phi_{i+3}
                 // Stencil S0: {phi_{i-1}, phi_i, phi_{i+1}} -> v[1], v[2], v[3]
                 // Stencil S1: {phi_i, phi_{i+1}, phi_{i+2}} -> v[2], v[3], v[4]
                 // Stencil S2: {phi_{i+1}, phi_{i+2}, phi_{i+3}} -> v[3], v[4], v[5]
            p0 = (-v[1] + 5.0*v[2] + 2.0*v[3]) / 6.0; // Note: This is p1 from left-biased
            p1 = (2.0*v[2] + 5.0*v[3] - v[4]) / 6.0;  // Note: This is p2 from left-biased
            p2 = (11.0*v[3] - 7.0*v[4] + 2.0*v[5]) / 6.0;

            // Smoothness indicators for D_i^+ (shifted stencil)
            beta0 = (13.0/12.0) * std::pow(v[1] - 2.0*v[2] + v[3], 2) + (1.0/4.0) * std::pow(v[1] - 4.0*v[2] + 3.0*v[3], 2);
            beta1 = (13.0/12.0) * std::pow(v[2] - 2.0*v[3] + v[4], 2) + (1.0/4.0) * std::pow(v[2] - v[4], 2);
            beta2 = (13.0/12.0) * std::pow(v[3] - 2.0*v[4] + v[5], 2) + (1.0/4.0) * std::pow(3.0*v[3] - 4.0*v[4] + v[5], 2);
        }

        double alpha0_tilde = (isLeftBiased ? c0 : c2_p) / std::pow(epsilon + beta0, 2); // c2_p = 0.1 for right-biased
        double alpha1_tilde = (isLeftBiased ? c1 : c1_p) / std::pow(epsilon + beta1, 2); // c1 = 0.6
        double alpha2_tilde = (isLeftBiased ? c2 : c0_p) / std::pow(epsilon + beta2, 2); // c0_p = 0.3 for right-biased
        
        // The ideal weights for D_i^+ are d_0 = 0.3, d_1 = 0.6, d_2 = 0.1
        // The provided code used 0.1, 0.6, 0.3 for both. Corrected for D_i^+
        // For D_i^- (isLeftBiased=true): c0=0.1, c1=0.6, c2=0.3
        // For D_i^+ (isLeftBiased=false): c0=0.3, c1=0.6, c2=0.1 (if mapping p0,p1,p2 to these weights)
        // The original code's beta and q values for forward=false (isLeftBiased=false)
        // seemed to map to ideal weights 0.3, 0.6, 0.1 for its q0, q1, q2.
        // Let's use the standard Jiang-Shu weights:
        // D^- (isLeftBiased=true): d_0=0.1 (for p0), d_1=0.6 (for p1), d_2=0.3 (for p2)
        // D^+ (isLeftBiased=false): d_0=0.3 (for p0), d_1=0.6 (for p1), d_2=0.1 (for p2)
        // The variable names c0, c1, c2 are used for D^-. For D^+, they are c2, c1, c0 effectively.
        
        if (!isLeftBiased) { // Adjust ideal weights for D_i^+
             alpha0_tilde = (3.0/10.0) / std::pow(epsilon + beta0, 2);
             alpha1_tilde = (6.0/10.0) / std::pow(epsilon + beta1, 2);
             alpha2_tilde = (1.0/10.0) / std::pow(epsilon + beta2, 2);
        }


        double sum_alpha_tilde = alpha0_tilde + alpha1_tilde + alpha2_tilde;
        double w0 = alpha0_tilde / sum_alpha_tilde;
        double w1 = alpha1_tilde / sum_alpha_tilde;
        double w2 = alpha2_tilde / sum_alpha_tilde;
            
        return (w0 * p0 + w1 * p1 + w2 * p2) / h;
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

class RungeKutta3Scheme : public TimeScheme { // SSP-RK3
public:
    RungeKutta3Scheme(double timeStep) : TimeScheme(timeStep) {}
    
    Eigen::VectorXd advance(const Eigen::VectorXd& phi_n, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        // Standard SSP-RK3 scheme
        Eigen::VectorXd phi_1 = phi_n + dt * L(phi_n);
        Eigen::VectorXd phi_2 = (3.0/4.0) * phi_n + (1.0/4.0) * phi_1 + (1.0/4.0) * dt * L(phi_1);
        Eigen::VectorXd phi_n_plus_1 = (1.0/3.0) * phi_n + (2.0/3.0) * phi_2 + (2.0/3.0) * dt * L(phi_2);
        return phi_n_plus_1;
    }
};


class LevelSetMethod {
public:
    // Constructor
    LevelSetMethod(
                const std::string& meshFile,
                const std::string& orgFile, // Original mesh for material mapping
                const std::string& materialCsvFile,
                int gridSize = 400, 
                double timeStep = 0.01, 
                int maxSteps = 80, 
                int reinitInterval = 5,
                int narrowBandInterval = 100,
                double narrowBandWidth = 10.0, // In terms of grid cells
                int numThreads = -1, // -1 for default OpenMP threads
                double curvatureWeight = 0.0,
                SpatialSchemeType spatialSchemeType = SpatialSchemeType::UPWIND,
                TimeSchemeType timeSchemeType = TimeSchemeType::FORWARD_EULER)
        : GRID_SIZE(gridSize),
        dt(timeStep),
        STEPS(maxSteps),
        REINIT_INTERVAL(reinitInterval),
        NARROW_BAND_UPDATE_INTERVAL(narrowBandInterval),
        NARROW_BAND_WIDTH_CELLS(narrowBandWidth), // Store as cells
        CURVATURE_WEIGHT(curvatureWeight){

        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        } else {
            // omp_get_max_threads() could be used if specific default behavior is desired
        }
        
        loadMesh(meshFile); // Load the evolving mesh
        generateGrid();     // Generate grid based on loaded mesh
        loadMaterialInfo(materialCsvFile, orgFile); // Load material properties
        
        // Initialize spatial scheme
        switch (spatialSchemeType) {
            case SpatialSchemeType::UPWIND:
                spatialScheme = std::make_shared<UpwindScheme>(GRID_SIZE);
                break;
            case SpatialSchemeType::WENO:
                spatialScheme = std::make_shared<WENOScheme>(GRID_SIZE);
                break;
            default: // Fallback to Upwind
                std::cerr << "Warning: Unknown spatial scheme type. Defaulting to UPWIND." << std::endl;
                spatialScheme = std::make_shared<UpwindScheme>(GRID_SIZE);
                break;
        }
        
        // Initialize time scheme
        switch (timeSchemeType) {
            case TimeSchemeType::FORWARD_EULER:
                timeScheme = std::make_shared<ForwardEulerScheme>(dt);
                break;
            case TimeSchemeType::RUNGE_KUTTA_3:
                timeScheme = std::make_shared<RungeKutta3Scheme>(dt);
                break;
            default: // Fallback to Forward Euler
                std::cerr << "Warning: Unknown time scheme type. Defaulting to FORWARD_EULER." << std::endl;
                timeScheme = std::make_shared<ForwardEulerScheme>(dt);
                break;
        }
    }
    virtual ~LevelSetMethod() = default; // Virtual destructor for base class
    
    // Public interface
    CGAL::Bbox_3 calculateBoundingBox() const;
    bool extractSurfaceMeshCGAL(const std::string& filename);
    void loadMesh(const std::string& filename); // For the evolving surface
    virtual bool evolve(); // Made virtual for overriding
    void reinitialize();
    
    // Material properties setup
    void setMaterialProperties(const std::string& material, double etchRatio, double lateralRatio){
        // std::unique_lock<std::shared_mutex> lock(materialPropertiesMutex); // If writing concurrently
        materialProperties[material] = {etchRatio, lateralRatio, material};
    }

protected: // Changed to protected for access by derived classes
    // Grid and simulation parameters
    const int GRID_SIZE;
    double GRID_SPACING = 0.0; // Initialized in generateGrid
    const double dt; // Time step
    const int STEPS; // Total number of evolution steps
    const int REINIT_INTERVAL; // Frequency of reinitialization
    const int NARROW_BAND_UPDATE_INTERVAL; // Frequency of narrow band update
    const double NARROW_BAND_WIDTH_CELLS; // Width of the narrow band in grid cells
    const double CURVATURE_WEIGHT;
    double BOX_SIZE = -1.0; // Physical size of the simulation box, set in generateGrid
    
    // Grid origin (min corner of the bounding box)
    Point_3 gridOrigin{0.0, 0.0, 0.0}; // Initialized in generateGrid
   
    // Scheme pointers
    std::shared_ptr<SpatialScheme> spatialScheme;
    std::shared_ptr<TimeScheme> timeScheme;

    // CGAL and Eigen data structures
    Mesh mesh; // The evolving surface mesh
    std::unique_ptr<AABB_tree> tree; // AABB tree for distance queries to 'mesh'
    std::vector<Point_3> gridPoints; // Coordinates of grid points
    Eigen::VectorXd phi; // Signed distance function values at grid points
    std::vector<int> narrowBandIndices; // Indices of grid points within the narrow band
    
    // Material related members
    struct MaterialProperties {
        double etchRatio;
        double lateralRatio;
        std::string name; // Optional: for debugging or identification
    };
    
    std::unordered_map<std::string, MaterialProperties> materialProperties;
    // mutable std::shared_mutex materialPropertiesMutex; // For thread-safe access if needed
    std::vector<std::string> gridCellMaterials; // Material type for each grid cell
    
    // Core simulation methods
    void loadMaterialInfo(const std::string& csvFilename, const std::string& orgMeshFilename);
    std::string getMaterialAtPoint(int gridIndex) const; // Get material for a grid cell

    double computeMeanCurvature(int gridIndex, const Eigen::VectorXd& currentPhi);
    void updateNarrowBand();
    void generateGrid();
    Eigen::VectorXd initializeSignedDistanceField();
    
    // Utility methods
    bool isOnBoundary(int gridIndex, int boundaryThickness = 1) const; // Added thickness parameter
    
    // Optimized getIndex, ensures it's always inlined and handles boundary clamping internally
    // This version is for internal use where x,y,z are already validated or are fine with clamping.
    inline int getIndexUnsafe(int x, int y, int z) const {
        return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
    }

    // Safe version of getIndex, clamps coordinates to be within grid boundaries.
    // Useful for stencil operations that might read near boundaries.
    inline int getIndexSafe(int x, int y, int z) const {
        x = std::max(0, std::min(x, GRID_SIZE - 1));
        y = std::max(0, std::min(y, GRID_SIZE - 1));
        z = std::max(0, std::min(z, GRID_SIZE - 1));
        return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
    }
};

#endif // LEVEL_SET_METHOD_HPP
