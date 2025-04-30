#ifndef OPENVDB_LEVEL_SET_HPP
#define OPENVDB_LEVEL_SET_HPP

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/math/Operators.h>
#include <openvdb/tools/LevelSetAdvect.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;

namespace PMP = CGAL::Polygon_mesh_processing;

class OpenVDBLevelSet {
public:
    OpenVDBLevelSet(double voxelSize = 1.0, 
                   double timeStep = 0.1, 
                   int maxSteps = 100,
                   double narrowBandWidth = 3.0)
        : mVoxelSize(voxelSize),
          mTimeStep(timeStep),
          mMaxSteps(maxSteps),
          mNarrowBandWidth(narrowBandWidth) {
        // Initialize OpenVDB
        openvdb::initialize();
    }

    ~OpenVDBLevelSet() {
        // Clean up OpenVDB
    }

    bool loadMesh(const std::string& filename) {
        try {
            // Load mesh using CGAL
            if (!CGAL::IO::read_polygon_mesh(filename, mMesh) || 
                is_empty(mMesh) || 
                !CGAL::is_closed(mMesh) || 
                !is_triangle_mesh(mMesh)) {
                std::cerr << "Error: Could not open file " << filename << std::endl;
                return false;
            }

            // Convert CGAL mesh to OpenVDB level set
            std::vector<openvdb::Vec3s> points;
            std::vector<openvdb::Vec3I> triangles;

            // Extract vertices
            for (auto v : mMesh.vertices()) {
                Point_3 p = mMesh.point(v);
                points.emplace_back(p.x(), p.y(), p.z());
            }

            // Extract triangles
            for (auto f : mMesh.faces()) {
                auto vertices = mMesh.vertices_around_face(mMesh.halfedge(f));
                auto it = vertices.begin();
                int v0 = static_cast<int>(*it++); 
                int v1 = static_cast<int>(*it++); 
                int v2 = static_cast<int>(*it);
                triangles.emplace_back(v0, v1, v2);
            }

            // Create a transform with the specified voxel size
            openvdb::math::Transform::Ptr transform = 
                openvdb::math::Transform::createLinearTransform(mVoxelSize);

            // Convert mesh to level set
            mGrid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
                *transform, points, triangles, mNarrowBandWidth);

            // Name the grid for identification
            mGrid->setName("LevelSet");

            std::cout << "Mesh loaded and converted to level set" << std::endl;
            std::cout << "Grid dimensions: " << mGrid->evalActiveVoxelDim() << std::endl;
            std::cout << "Number of active voxels: " << mGrid->activeVoxelCount() << std::endl;

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading mesh: " << e.what() << std::endl;
            return false;
        }
    }

    bool evolve() {
        try {
            if (!mGrid) {
                throw std::runtime_error("Level set grid not initialized. Load a mesh first.");
            }

            // Create a level set filter for evolution
            openvdb::tools::LevelSetFilter<openvdb::FloatGrid> filter(*mGrid);

            // Main evolution loop
            for (int step = 0; step < mMaxSteps; ++step) {
                // Custom speed function for etching simulation
                auto etchingSpeedFunc = [&](const openvdb::Coord& xyz, 
                                          const openvdb::FloatGrid& grid) -> float {
                    // Get the gradient at this location
                    auto accessor = grid.getConstAccessor();
                    // Use the correct gradient calculation method for OpenVDB v12.0
                    // In v12.0, Gradient::result requires a transform map as the first parameter
                    openvdb::Vec3f gradient;
                    // Calculate gradient manually using central differencing
                    gradient[0] = (accessor.getValue(xyz.offsetBy(1, 0, 0)) - accessor.getValue(xyz.offsetBy(-1, 0, 0))) * 0.5f/mVoxelSize;
                    gradient[1] = (accessor.getValue(xyz.offsetBy(0, 1, 0)) - accessor.getValue(xyz.offsetBy(0, -1, 0))) * 0.5f/mVoxelSize;
                    gradient[2] = (accessor.getValue(xyz.offsetBy(0, 0, 1)) - accessor.getValue(xyz.offsetBy(0, 0, -1))) * 0.5f/mVoxelSize;
                    gradient.normalize();

                    // Gravity direction (unit vector pointing downward)
                    openvdb::Vec3f gravity(0.0f, 0.0f, 1.0f);

                    // Calculate angle between normal and gravity direction
                    float dotProduct = gradient.dot(gravity);
                    float theta = std::acos(std::max(std::min(dotProduct, 1.0f), -1.0f));

                    // Calculate etching speed based on angle
                    float sigma = 14.0f; // Parameter controlling angular spread
                    return dotProduct * std::exp(-theta/(2.0f*sigma*sigma));
                };

                // Apply advection with custom speed function
                // In OpenVDB v12.0, we need to use different approach for advection
                // Use the grid's transform to convert world space to index space
                auto transform = mGrid->transformPtr();
                
                // Apply level set evolution directly
                for (auto iter = mGrid->beginValueOn(); iter; ++iter) {
                    const openvdb::Coord xyz = iter.getCoord();
                    const float speed = etchingSpeedFunc(xyz, *mGrid);
                    iter.setValue(iter.getValue() + speed * mTimeStep);
                }

                // Apply mean curvature flow for regularization (optional)
                filter.meanCurvature(); // In OpenVDB v12.0, this doesn't take a time step parameter

                if (step % 10 == 0) {
                    std::cout << "Step " << step << " completed. " 
                              << "Active voxels: " << mGrid->activeVoxelCount() << std::endl;
                }
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error during evolution: " << e.what() << std::endl;
            return false;
        }
    }

    bool extractSurfaceMesh(const std::string& filename) {
        try {
            if (!mGrid) {
                throw std::runtime_error("Level set grid not initialized.");
            }

            // Convert level set to mesh
            std::vector<openvdb::Vec3s> points;
            std::vector<openvdb::Vec3I> triangles;
            std::vector<openvdb::Vec4I> quads;

            // Extract the zero level set as a mesh
            double isovalue = 0.0;
            openvdb::tools::volumeToMesh(*mGrid, points, triangles, quads, isovalue);

            // Convert to CGAL mesh for output
            Mesh outputMesh;
            std::vector<Mesh::Vertex_index> vertexIndices;

            // Add vertices
            for (const auto& p : points) {
                vertexIndices.push_back(outputMesh.add_vertex(Point_3(p.x(), p.y(), p.z())));
            }

            // Add triangular faces
            for (const auto& tri : triangles) {
                outputMesh.add_face(vertexIndices[tri[0]], 
                                   vertexIndices[tri[1]], 
                                   vertexIndices[tri[2]]);
            }

            // Add quad faces (triangulate them)
            for (const auto& quad : quads) {
                outputMesh.add_face(vertexIndices[quad[0]], 
                                   vertexIndices[quad[1]], 
                                   vertexIndices[quad[2]]);
                outputMesh.add_face(vertexIndices[quad[0]], 
                                   vertexIndices[quad[2]], 
                                   vertexIndices[quad[3]]);
            }

            // Save the mesh - use the correct namespace in CGAL 6.0
            if (!CGAL::IO::write_polygon_mesh(filename, outputMesh, 
                                           CGAL::parameters::stream_precision(17))) {
                throw std::runtime_error("Failed to write surface mesh to file.");
            }

            std::cout << "Surface mesh extracted and saved to " << filename << std::endl;
            std::cout << "Surface mesh has " << outputMesh.number_of_vertices() << " vertices and " 
                      << outputMesh.number_of_faces() << " faces." << std::endl;

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error extracting surface mesh: " << e.what() << std::endl;
            return false;
        }
    }

private:
    // Configuration parameters
    double mVoxelSize;
    double mTimeStep;
    int mMaxSteps;
    double mNarrowBandWidth;
    
    // Data structures
    Mesh mMesh;
    openvdb::FloatGrid::Ptr mGrid;
};

#endif // OPENVDB_LEVEL_SET_HPP