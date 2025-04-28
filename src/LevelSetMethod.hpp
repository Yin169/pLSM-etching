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
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <functional>

namespace PMP = CGAL::Polygon_mesh_processing;


typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<Kernel, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> AABB_tree;


class LevelSetMethod {
public:

    LevelSetMethod(int gridSize = 400, double boxSize = 1400.0, 
                  double timeStep = 0.01, int maxSteps = 80, 
                  int reinitInterval = 5)
        : GRID_SIZE(gridSize), 
          BOX_SIZE(boxSize),
          GRID_SPACING(boxSize / (gridSize - 1)),
          dt(timeStep),
          STEPS(maxSteps),
          REINIT_INTERVAL(reinitInterval) {
        
        generateGrid();
    }


	bool extractSurfaceMeshCGAL(const std::string& filename);

    bool loadMesh(const std::string& filename) {
        try {
            if (!PMP::IO::read_polygon_mesh(filename, mesh) || is_empty(mesh) || !CGAL::is_closed(mesh) || !is_triangle_mesh(mesh)) {
                std::cerr << "Error: Could not open file " << filename << std::endl;
                return false;
            }
            
            tree = std::make_unique<AABB_tree>(faces(mesh).first, faces(mesh).second, mesh);
            // tree->build();
            tree->accelerate_distance_queries(); 

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading mesh: " << e.what() << std::endl;
            return false;
        }
    }


    bool evolve() {
        try {
            // Initialize the signed distance field
            phi = initializeSignedDistanceField();
            
            // Main evolution loop
            for (int step = 0; step < STEPS; ++step) {
                // Create a copy of the current level set
                Eigen::VectorXd newPhi = phi;
                
                for (size_t i = 0; i < grid.size(); ++i) {
                    if (isOnBoundary(i)) continue; // Skip boundary points
                    
                    // Get grid indices
                    int idx = i;
                    int x = idx % GRID_SIZE;
                    int y = (idx / GRID_SIZE) % GRID_SIZE;
                    int z = idx / (GRID_SIZE * GRID_SIZE);

                    if (std::abs(phi[idx]) >= 1) continue;
                    
                    // Calculate spatial derivatives using central differences
                    double dx_forward = (phi[getIndex(x+1, y, z)] - phi[idx]) / GRID_SPACING;
                    double dx_backward = (phi[idx] - phi[getIndex(x-1, y, z)]) / GRID_SPACING;
                    double dy_forward = (phi[getIndex(x, y+1, z)] - phi[idx]) / GRID_SPACING;
                    double dy_backward = (phi[idx] - phi[getIndex(x, y-1, z)]) / GRID_SPACING;
                    double dz_forward = (phi[getIndex(x, y, z+1)] - phi[idx]) / GRID_SPACING;
                    double dz_backward = (phi[idx] - phi[getIndex(x, y, z-1)]) / GRID_SPACING;
                    
                    // Calculate gradient magnitude using upwind scheme
                    double dx = (dx_forward > 0) ? std::max(dx_backward, 0.0) : std::min(dx_forward, 0.0);
                    double dy = (dy_forward > 0) ? std::max(dy_backward, 0.0) : std::min(dy_forward, 0.0);
                    double dz = (dz_forward > 0) ? std::max(dz_backward, 0.0) : std::min(dz_forward, 0.0);
                    
                    double gradMag = std::sqrt(dx*dx + dy*dy + dz*dz);
                    
                    // Calculate curvature term
                    double dxx = (phi[getIndex(x+1, y, z)] - 2*phi[idx] + phi[getIndex(x-1, y, z)]) / (GRID_SPACING*GRID_SPACING);
                    double dyy = (phi[getIndex(x, y+1, z)] - 2*phi[idx] + phi[getIndex(x, y-1, z)]) / (GRID_SPACING*GRID_SPACING);
                    double dzz = (phi[getIndex(x, y, z+1)] - 2*phi[idx] + phi[getIndex(x, y, z-1)]) / (GRID_SPACING*GRID_SPACING);
                    double dxy = (phi[getIndex(x+1, y+1, z)] - phi[getIndex(x+1, y-1, z)] - phi[getIndex(x-1, y+1, z)] + phi[getIndex(x-1, y-1, z)]) / (4*GRID_SPACING*GRID_SPACING);
                    double dxz = (phi[getIndex(x+1, y, z+1)] - phi[getIndex(x+1, y, z-1)] - phi[getIndex(x-1, y, z+1)] + phi[getIndex(x-1, y, z-1)]) / (4*GRID_SPACING*GRID_SPACING);
                    double dyz = (phi[getIndex(x, y+1, z+1)] - phi[getIndex(x, y+1, z-1)] - phi[getIndex(x, y-1, z+1)] + phi[getIndex(x, y-1, z-1)]) / (4*GRID_SPACING*GRID_SPACING);
                    
                    // Mean curvature calculation
                    double curvature = 0.0;
                    if (gradMag > 1e-10) {
                        curvature = (dxx*(dy*dy + dz*dz) + dyy*(dx*dx + dz*dz) + dzz*(dx*dx + dy*dy) 
                                    - 2*dxy*dx*dy - 2*dxz*dx*dz - 2*dyz*dy*dz) / (gradMag*gradMag*gradMag);
                    }
                    
                    // Calculate extension speed F based on the first equation
                    // Assuming gravity direction is (0, 0, -1) and sigma = 0.5
                    double nx = dx / (gradMag + 1e-10);
                    double ny = dy / (gradMag + 1e-10);
                    double nz = dz / (gradMag + 1e-10);
                    
                    // Gravity direction (unit vector pointing downward)
                    double gx = 0.0;
                    double gy = 0.0;
                    double gz = -1.0;
                    
                    // Calculate theta (angle between normal and gravity direction)
                    double dotProduct = nx*gx + ny*gy + nz*gz;
                    double theta = std::acos(std::min(std::max(dotProduct, -1.0), 1.0));
                    
                    // Calculate extension speed F
                    double sigma = 1.0; // Parameter controlling angular spread
                    double F = std::exp(-theta/(2*sigma*sigma));
                    
                    // Curvature coefficient (epsilon)
                    double epsilon = 0.1;
                    
                    // Update level set function using the level set equation
                    newPhi[idx] = phi[idx] - dt * (F * gradMag - epsilon * curvature * gradMag);
                }
                
                phi = newPhi;
                
                // Reinitialization to maintain signed distance property
                if (step % REINIT_INTERVAL == 0 && step > 0) {
                    reinitialize();
                }

                if (step % 10 == 0) {
                    std::cout << "Step " << step << " completed." << std::endl;
                }
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error during evolution: " << e.what() << std::endl;
            return false;
        }
    }

    void reinitialize() {
        // Create a temporary copy of phi
        Eigen::VectorXd tempPhi = phi;
        
        // Number of iterations for reinitialization
        const int REINIT_STEPS = 10;
        const double dtau = dt; // Time step for reinitialization
        
        // Perform reinitialization iterations
        for (int step = 0; step < REINIT_STEPS; ++step) {
            #pragma omp parallel for
            for (size_t i = 0; i < grid.size(); ++i) {
                if (isOnBoundary(i)) continue;
                
                int idx = i;
                int x = idx % GRID_SIZE;
                int y = (idx / GRID_SIZE) % GRID_SIZE;
                int z = idx / (GRID_SIZE * GRID_SIZE);
                
                double dx = (tempPhi[getIndex(x+1, y, z)] - tempPhi[getIndex(x-1, y, z)]) / (2*GRID_SPACING);
                double dy = (tempPhi[getIndex(x, y+1, z)] - tempPhi[getIndex(x, y-1, z)]) / (2*GRID_SPACING);
                double dz = (tempPhi[getIndex(x, y, z+1)] - tempPhi[getIndex(x, y, z-1)]) / (2*GRID_SPACING);
                
                double gradMag = std::sqrt(dx*dx + dy*dy + dz*dz);
               
                // Sign function
                double sign = tempPhi[idx] / std::sqrt(tempPhi[idx]*tempPhi[idx] + gradMag*gradMag*GRID_SPACING*GRID_SPACING); 

                // Update equation for reinitialization
                tempPhi[idx] = tempPhi[idx] - dtau * sign * (gradMag - 1.0);
            }
        }
        
        // Update phi with reinitialized values
        phi = tempPhi;
    }


    bool saveResult(const std::string& filename) {
        try {
            std::ofstream output(filename);
            if (!output.is_open()) {
                std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
                return false;
            }
            
            output << "x,y,z,value" << std::endl;
            
            // Write the SDF values with coordinates
            for (size_t i = 0; i < grid.size(); ++i) {
                output << grid[i].x() << "," << grid[i].y() << "," << grid[i].z() << "," << phi[i] << std::endl;
            }
            
            output.close();
            std::cout << "Results saved to " << filename << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error saving results: " << e.what() << std::endl;
            return false;
        }
    }

private:
    // Configuration parameters
    const int GRID_SIZE;
    const double BOX_SIZE;
    const double GRID_SPACING;
    const double dt;
    const int STEPS;
    const int REINIT_INTERVAL;
    
    // Data structures
    Mesh mesh;
    std::unique_ptr<AABB_tree> tree;
    std::vector<Point_3> grid;
    Eigen::VectorXd phi;
    

    void generateGrid() {
        grid.clear();
        grid.reserve(GRID_SIZE * GRID_SIZE * GRID_SIZE);
        
        double halfBox = BOX_SIZE / 2.0;
        for (int x = 0; x < GRID_SIZE; ++x) {
            for (int y = 0; y < GRID_SIZE; ++y) {
                for (int z = 0; z < GRID_SIZE; ++z) {
                    double px = -halfBox + x * GRID_SPACING;
                    double py = -halfBox + y * GRID_SPACING;
                    double pz = -halfBox + z * GRID_SPACING;
                    grid.emplace_back(px, py, pz);
                }
            }
        }
    }
    

    Eigen::VectorXd initializeSignedDistanceField() {
        if (!tree) {
            throw std::runtime_error("AABB tree not initialized. Load a mesh first.");
        }
        
        Eigen::VectorXd sdf(grid.size());
        CGAL::Side_of_triangle_mesh<Mesh, Kernel> inside(mesh);

        #pragma omp parallel for
        for (size_t i = 0; i < grid.size(); ++i) {
            // Compute squared distance to the mesh
            // auto closest = tree->closest_point_and_primitive(grid[i]);
            // double sq_dist = CGAL::sqrt(CGAL::squared_distance(grid[i], closest.first));
            double sq_dist = 1.0;
            
            CGAL::Bounded_side res = inside(grid[i]);

            if (res == CGAL::ON_BOUNDED_SIDE){
                sdf[i] = -sq_dist;
            } else if (res == CGAL::ON_BOUNDARY){
                sdf[i] = 0.0;
            } else{ 
                sdf[i] = sq_dist;
            }
            
        }
        
        return sdf;
    }
    

    bool isOnBoundary(int idx) const {
        int x = idx % GRID_SIZE;
        int y = (idx / GRID_SIZE) % GRID_SIZE;
        int z = idx / (GRID_SIZE * GRID_SIZE);
        
        return x == 0 || x == GRID_SIZE - 1 || 
               y == 0 || y == GRID_SIZE - 1 || 
               z == 0 || z == GRID_SIZE - 1;
    }
    

    int getIndex(int x, int y, int z) const {
        // Boundary check
        x = std::max(0, std::min(x, GRID_SIZE - 1));
        y = std::max(0, std::min(y, GRID_SIZE - 1));
        z = std::max(0, std::min(z, GRID_SIZE - 1));
        
        return x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
    }
    
    

};


bool LevelSetMethod::extractSurfaceMeshCGAL(const std::string& filename) {
    try {
        if (phi.size() != grid.size()) {
            throw std::runtime_error("Level set function not initialized.");
        }
 
        typedef CGAL::Surface_mesh_default_triangulation_3 Tr;
        typedef CGAL::Complex_2_in_triangulation_3<Tr> C2t3;
        typedef Tr::Geom_traits GT;
        typedef GT::Sphere_3 Sphere_3;
        typedef GT::FT FT;
        typedef std::function<FT(typename GT::Point_3)> Function;
        typedef CGAL::Implicit_surface_3<GT, Function> Surface_3;
    
        // Define the implicit function for the zero level set
        class LevelSetImplicitFunction {
        private:
            const std::vector<Point_3>& grid;
            const Eigen::VectorXd& phi;
            const int GRID_SIZE;
            const double GRID_SPACING;
            const double gridOriginX, gridOriginY, gridOriginZ;
            
        public:
            LevelSetImplicitFunction(const std::vector<Point_3>& grid, const Eigen::VectorXd& phi, 
                                    int gridSize, double gridSpacing)
                : grid(grid), phi(phi), GRID_SIZE(gridSize), GRID_SPACING(gridSpacing),
                  gridOriginX(grid[0].x()), gridOriginY(grid[0].y()), gridOriginZ(grid[0].z()) {
            }
                
            FT operator()(const Point_3& p) const {
                // Fast grid-based lookup instead of linear search
                // Calculate grid indices based on point position
                int x = std::round((p.x() - gridOriginX) / GRID_SPACING);
                int y = std::round((p.y() - gridOriginY) / GRID_SPACING);
                int z = std::round((p.z() - gridOriginZ) / GRID_SPACING);
                
                // Clamp to grid boundaries
                x = std::max(0, std::min(x, GRID_SIZE - 1));
                y = std::max(0, std::min(y, GRID_SIZE - 1));
                z = std::max(0, std::min(z, GRID_SIZE - 1));
                
                // Calculate grid index
                int idx = x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                
                // Bounds check
                if (idx >= 0 && idx < static_cast<int>(phi.size())) {
                    return phi[idx];
                }
                
                // Fallback to trilinear interpolation for points outside the grid
                // This provides smoother results than nearest neighbor
                // Find the cell containing the point
                x = std::floor((p.x() - gridOriginX) / GRID_SPACING);
                y = std::floor((p.y() - gridOriginY) / GRID_SPACING);
                z = std::floor((p.z() - gridOriginZ) / GRID_SPACING);
                
                // Clamp to valid range for interpolation
                x = std::max(0, std::min(x, GRID_SIZE - 2));
                y = std::max(0, std::min(y, GRID_SIZE - 2));
                z = std::max(0, std::min(z, GRID_SIZE - 2));
                
                // Calculate fractional position within cell
                double fx = (p.x() - (gridOriginX + x * GRID_SPACING)) / GRID_SPACING;
                double fy = (p.y() - (gridOriginY + y * GRID_SPACING)) / GRID_SPACING;
                double fz = (p.z() - (gridOriginZ + z * GRID_SPACING)) / GRID_SPACING;
                
                // Get the eight corners of the cell
                int idx000 = x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                int idx001 = x + y * GRID_SIZE + (z+1) * GRID_SIZE * GRID_SIZE;
                int idx010 = x + (y+1) * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                int idx011 = x + (y+1) * GRID_SIZE + (z+1) * GRID_SIZE * GRID_SIZE;
                int idx100 = (x+1) + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                int idx101 = (x+1) + y * GRID_SIZE + (z+1) * GRID_SIZE * GRID_SIZE;
                int idx110 = (x+1) + (y+1) * GRID_SIZE + z * GRID_SIZE * GRID_SIZE;
                int idx111 = (x+1) + (y+1) * GRID_SIZE + (z+1) * GRID_SIZE * GRID_SIZE;
                
                // Perform trilinear interpolation
                double v000 = phi[idx000];
                double v001 = phi[idx001];
                double v010 = phi[idx010];
                double v011 = phi[idx011];
                double v100 = phi[idx100];
                double v101 = phi[idx101];
                double v110 = phi[idx110];
                double v111 = phi[idx111];
                
                // Interpolate along x
                double v00 = v000 * (1 - fx) + v100 * fx;
                double v01 = v001 * (1 - fx) + v101 * fx;
                double v10 = v010 * (1 - fx) + v110 * fx;
                double v11 = v011 * (1 - fx) + v111 * fx;
                
                // Interpolate along y
                double v0 = v00 * (1 - fy) + v10 * fy;
                double v1 = v01 * (1 - fy) + v11 * fy;
                
                // Interpolate along z
                return v0 * (1 - fz) + v1 * fz;
            }
        };

        // Create the implicit function with grid parameters
        LevelSetImplicitFunction implicitFunction(grid, phi, GRID_SIZE, GRID_SPACING);

        // Wrap the implicit function with the corrected type
        Function function = [&implicitFunction](const GT::Point_3& p) {
            return implicitFunction(Point_3(p.x(), p.y(), p.z()));
        };
        
        Tr tr;
        C2t3 c2t3(tr);
        
        // Adjust bounding sphere to better match your data
        double boundingSphereRadius = BOX_SIZE;
        Surface_3 surface(function, Sphere_3(CGAL::ORIGIN, boundingSphereRadius*boundingSphereRadius), 1e-5);
        
        // Adjust mesh criteria for better performance/quality tradeoff
        typedef CGAL::Surface_mesh_default_criteria_3<Tr> Criteria;
        Criteria criteria(30.0, GRID_SPACING * 2.0, GRID_SPACING * 2.0);
        
        // Define the mesh data structure
        typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
        Surface_mesh surface_mesh;
        
        std::cout << "Starting surface mesh generation..." << std::endl;
        // Generate the surface mesh
        CGAL::make_surface_mesh(c2t3, surface, criteria, CGAL::Non_manifold_tag());
        std::cout << "Surface mesh generation completed." << std::endl;
        
        // Convert the complex to a surface mesh
        CGAL::facets_in_complex_2_to_triangle_mesh(c2t3, surface_mesh);
        
        // Rotate the mesh 90 degrees around the X-axis before saving
        for (auto v : surface_mesh.vertices()) {
            Point_3 p = surface_mesh.point(v);
            surface_mesh.point(v) = Point_3(p.z(), p.y(), p.x());
        }
        
        // Save the surface mesh to a file
        if (!CGAL::IO::write_polygon_mesh(filename, surface_mesh, CGAL::parameters::stream_precision(17))) {
            throw std::runtime_error("Failed to write surface mesh to file.");
        }
        
        std::cout << "Surface mesh extracted and saved to " << filename << std::endl;
        std::cout << "Surface mesh has " << surface_mesh.number_of_vertices() << " vertices and " 
                  << surface_mesh.number_of_faces() << " faces." << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error extracting surface mesh: " << e.what() << std::endl;
        return false;
    }
}

#endif