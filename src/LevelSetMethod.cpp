#include "LevelSetMethod.hpp"
#include <stdexcept>    // For std::runtime_error, std::out_of_range
#include <iostream>     // For std::cerr, std::cout, std::flush
#include <execution>    // For std::sort(std::execution::par, ...)
#include <algorithm>    // For std::sort, std::min, std::max
#include <vector>
#include <string>
#include <fstream>      // For file operations
#include <atomic>       // For atomic progress counters

// CGAL includes for surface extraction
#include <CGAL/Surface_mesh_default_triangulation_3.h>
#include <CGAL/Complex_2_in_triangulation_3.h>
#include <CGAL/Implicit_surface_3.h>
#include <CGAL/make_surface_mesh.h>
#include <CGAL/Surface_mesh_default_criteria_3.h>
#include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
#include <CGAL/IO/polygon_mesh_io.h>


CGAL::Bbox_3 LevelSetMethod::calculateBoundingBox() const {
    if (mesh.is_empty()) {
        throw std::runtime_error("Mesh is empty - cannot calculate bounding box.");
    }
    return PMP::bbox(mesh);
}

void LevelSetMethod::loadMesh(const std::string& filename) {
    if (!PMP::IO::read_polygon_mesh(filename, mesh) || mesh.is_empty() || !CGAL::is_triangle_mesh(mesh)) {
        std::cerr << "Error: Could not read valid triangle mesh from file: " << filename << std::endl;
        // Consider throwing an exception for critical failures
        throw std::runtime_error("Failed to load mesh: " + filename);
    }
    if (!CGAL::is_closed(mesh)) {
         std::cerr << "Warning: Mesh " << filename << " is not closed. SDF computation might be affected." << std::endl;
    }
        
    tree = std::make_unique<AABB_tree>(faces(mesh).begin(), faces(mesh).end(), mesh); // Use .begin() and .end()
    tree->accelerate_distance_queries(); 
    std::cout << "Mesh loaded successfully from: " << filename << std::endl;
}

double LevelSetMethod::computeMeanCurvature(int idx, const Eigen::VectorXd& currentPhi) {
    // Boundary check: return 0 curvature for points at or near the physical boundary of the grid.
    // A thicker boundary layer (e.g., 2-3 cells) might be needed depending on stencil size.
    if (isOnBoundary(idx, 3)) { // Use a boundary thickness of 3 for curvature calculation
        return 0.0; 
    }
    
    // Grid dimensions (assuming GRID_SIZE is the same in all dimensions)
    // static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE; // Already available via this->GRID_SIZE

    // Central differences for first and second derivatives
    // Using inv_spacing for slight performance gain (multiply vs divide)
    const double inv_h = 1.0 / GRID_SPACING;
    const double inv_h_sq = inv_h * inv_h;
    const double inv_2h = 0.5 * inv_h;
    const double inv_4h_sq = 0.25 * inv_h_sq; // For mixed derivatives

    // Current point's coordinates (not strictly needed if using getIndexSafe)
    // const int ix = idx % GRID_SIZE;
    // const int iy = (idx / GRID_SIZE) % GRID_SIZE;
    // const int iz = idx / (GRID_SIZE * GRID_SIZE);

    // Phi values at stencil points (using safe indexing)
    const double phi_c = currentPhi[idx];
    const double phi_xp = currentPhi[getIndexSafe((idx % GRID_SIZE) + 1, (idx / GRID_SIZE) % GRID_SIZE, idx / (GRID_SIZE*GRID_SIZE))];
    const double phi_xn = currentPhi[getIndexSafe((idx % GRID_SIZE) - 1, (idx / GRID_SIZE) % GRID_SIZE, idx / (GRID_SIZE*GRID_SIZE))];
    const double phi_yp = currentPhi[getIndexSafe((idx % GRID_SIZE), ((idx / GRID_SIZE) % GRID_SIZE) + 1, idx / (GRID_SIZE*GRID_SIZE))];
    const double phi_yn = currentPhi[getIndexSafe((idx % GRID_SIZE), ((idx / GRID_SIZE) % GRID_SIZE) - 1, idx / (GRID_SIZE*GRID_SIZE))];
    const double phi_zp = currentPhi[getIndexSafe((idx % GRID_SIZE), (idx / GRID_SIZE) % GRID_SIZE, (idx / (GRID_SIZE*GRID_SIZE)) + 1)];
    const double phi_zn = currentPhi[getIndexSafe((idx % GRID_SIZE), (idx / GRID_SIZE) % GRID_SIZE, (idx / (GRID_SIZE*GRID_SIZE)) - 1)];

    // First derivatives (central differences)
    const double phix = (phi_xp - phi_xn) * inv_2h;
    const double phiy = (phi_yp - phi_yn) * inv_2h;
    const double phiz = (phi_zp - phi_zn) * inv_2h;

    // Second derivatives (central differences)
    const double phixx = (phi_xp - 2.0 * phi_c + phi_xn) * inv_h_sq;
    const double phiyy = (phi_yp - 2.0 * phi_c + phi_yn) * inv_h_sq;
    const double phizz = (phi_zp - 2.0 * phi_c + phi_zn) * inv_h_sq;

    // Mixed derivatives
    const double phixy = (currentPhi[getIndexSafe(((idx % GRID_SIZE) + 1), (((idx / GRID_SIZE) % GRID_SIZE) + 1), (idx / (GRID_SIZE*GRID_SIZE)))] -
                         currentPhi[getIndexSafe(((idx % GRID_SIZE) + 1), (((idx / GRID_SIZE) % GRID_SIZE) - 1), (idx / (GRID_SIZE*GRID_SIZE)))] -
                         currentPhi[getIndexSafe(((idx % GRID_SIZE) - 1), (((idx / GRID_SIZE) % GRID_SIZE) + 1), (idx / (GRID_SIZE*GRID_SIZE)))] +
                         currentPhi[getIndexSafe(((idx % GRID_SIZE) - 1), (((idx / GRID_SIZE) % GRID_SIZE) - 1), (idx / (GRID_SIZE*GRID_SIZE)))]) * inv_4h_sq;

    const double phixz = (currentPhi[getIndexSafe(((idx % GRID_SIZE) + 1), ((idx / GRID_SIZE) % GRID_SIZE), ((idx / (GRID_SIZE*GRID_SIZE)) + 1))] -
                         currentPhi[getIndexSafe(((idx % GRID_SIZE) + 1), ((idx / GRID_SIZE) % GRID_SIZE), ((idx / (GRID_SIZE*GRID_SIZE)) - 1))] -
                         currentPhi[getIndexSafe(((idx % GRID_SIZE) - 1), ((idx / GRID_SIZE) % GRID_SIZE), ((idx / (GRID_SIZE*GRID_SIZE)) + 1))] +
                         currentPhi[getIndexSafe(((idx % GRID_SIZE) - 1), ((idx / GRID_SIZE) % GRID_SIZE), ((idx / (GRID_SIZE*GRID_SIZE)) - 1))]) * inv_4h_sq;

    const double phiyz = (currentPhi[getIndexSafe((idx % GRID_SIZE), (((idx / GRID_SIZE) % GRID_SIZE) + 1), ((idx / (GRID_SIZE*GRID_SIZE)) + 1))] -
                         currentPhi[getIndexSafe((idx % GRID_SIZE), (((idx / GRID_SIZE) % GRID_SIZE) + 1), ((idx / (GRID_SIZE*GRID_SIZE)) - 1))] -
                         currentPhi[getIndexSafe((idx % GRID_SIZE), (((idx / GRID_SIZE) % GRID_SIZE) - 1), ((idx / (GRID_SIZE*GRID_SIZE)) + 1))] +
                         currentPhi[getIndexSafe((idx % GRID_SIZE), (((idx / GRID_SIZE) % GRID_SIZE) - 1), ((idx / (GRID_SIZE*GRID_SIZE)) - 1))]) * inv_4h_sq;

    // Gradient magnitude squared
    const double grad_phi_sq = phix * phix + phiy * phiy + phiz * phiz;
    constexpr double epsilon_grad = 1e-12; // Small epsilon to prevent division by zero

    if (grad_phi_sq < epsilon_grad) {
        return 0.0; // Flat region, curvature is ill-defined or zero
    }

    // Mean curvature formula (divergence of the normalized gradient vector)
    double numerator = phixx * (phiy * phiy + phiz * phiz) +
                       phiyy * (phix * phix + phiz * phiz) +
                       phizz * (phix * phix + phiy * phiy) -
                       2.0 * (phixy * phix * phiy +
                              phixz * phix * phiz +
                              phiyz * phiy * phiz);
    
    double denominator = 2.0 * std::pow(grad_phi_sq, 1.5); // 2 * |grad(phi)|^3
    
    double curvature = numerator / denominator;

    // Clamp curvature to prevent numerical instability (optional, but often good)
    const double max_abs_curvature = 1.0 / GRID_SPACING; // Max curvature related to grid size
    return std::max(-max_abs_curvature, std::min(curvature, max_abs_curvature));
}


bool LevelSetMethod::evolve() {
    try {
        phi = initializeSignedDistanceField();
        updateNarrowBand();
        
        // newPhi is not needed here if using timeScheme correctly
        // Eigen::VectorXd newPhi = phi; // Not needed if phi is updated by timeScheme
        
        const int progressInterval = std::max(1, STEPS / 20); // Show progress roughly 20 times
        
        // Pre-computation outside the loop if narrowBand doesn't change frequently
        // or recompute inside if NARROW_BAND_UPDATE_INTERVAL is small.
        // The current logic recomputes these when narrow band is updated.
        std::vector<Eigen::Vector3d> materialEtchRatesLocal(narrowBandIndices.size());
        std::vector<bool> isBoundaryLocal(narrowBandIndices.size());

        auto precomputeNarrowBandData = [&]() {
            materialEtchRatesLocal.resize(narrowBandIndices.size());
            isBoundaryLocal.resize(narrowBandIndices.size());
            #pragma omp parallel for schedule(dynamic, 1024) // Consider guided or static for more uniform work
            for (size_t k = 0; k < narrowBandIndices.size(); ++k) {
                const int idx = narrowBandIndices[k];
                std::string material = getMaterialAtPoint(idx);
                
                Eigen::Vector3d etching_rates_vec;
                const auto it = materialProperties.find(material);
                if (it != materialProperties.end()) { 
                    const auto& props = it->second;  
                    const double lateral_etch = props.lateralRatio * props.etchRatio; 
                    etching_rates_vec << -lateral_etch, -lateral_etch, -props.etchRatio; 
                } else {
                    // std::cerr << "Warning: Material '" << material << "' not found for point " << idx << ". Using zero etch rate." << std::endl;
                    etching_rates_vec.setZero();  
                }
                materialEtchRatesLocal[k] = etching_rates_vec;
                isBoundaryLocal[k] = isOnBoundary(idx);
            }
        };

        precomputeNarrowBandData(); // Initial pre-computation
        
        constexpr double epsilon_grad_mag = 1e-10; // For normalizing gradient
        const bool use_curvature_term = CURVATURE_WEIGHT > 0.0;
        
        // Main evolution loop
        for (int step = 0; step < STEPS; ++step) {
            if (step % progressInterval == 0) {
                std::cout << "Evolution step " << step << "/" << STEPS 
                          << " (Narrow band size: " << narrowBandIndices.size() << ")" << std::endl;
            }
            
            // Define level set operator (RHS of d(phi)/dt = L(phi))
            // This lambda captures necessary variables.
            auto levelSetOperator = 
                [this, &materialEtchRatesLocal, &isBoundaryLocal, epsilon_grad_mag, use_curvature_term]
                (const Eigen::VectorXd& currentPhi) -> Eigen::VectorXd {
                
                Eigen::VectorXd L_phi = Eigen::VectorXd::Zero(currentPhi.size());
                
                #pragma omp parallel for schedule(dynamic, 1024) // Consider guided or static
                for (size_t k = 0; k < narrowBandIndices.size(); ++k) {
                    const int idx = narrowBandIndices[k];
                    
                    if (isBoundaryLocal[k]) { // Skip points on the physical boundary of the grid
                        continue;
                    }
                    
                    const Eigen::Vector3d& V_material = materialEtchRatesLocal[k]; // Etching velocity vector
                    
                    DerivativeOperator Dop; // To store spatial derivatives
                    spatialScheme->SpatialSch(idx, currentPhi, GRID_SPACING, Dop);
                    
                    // Advection term: -V_material . grad(phi)
                    // Using upwinded derivatives based on sign of V_material components
                    double advection_term = 0.0;
                    advection_term += (V_material.x() > 0 ? V_material.x() * Dop.dxN : V_material.x() * Dop.dxP);
                    advection_term += (V_material.y() > 0 ? V_material.y() * Dop.dyN : V_material.y() * Dop.dyP);
                    advection_term += (V_material.z() > 0 ? V_material.z() * Dop.dzN : V_material.z() * Dop.dzP);
                    
                    // Curvature term: -alpha * K * |grad(phi)|
                    double curvature_flow_term = 0.0;
                    if (use_curvature_term) {
                        double mean_k = computeMeanCurvature(idx, currentPhi);
                        
                        // Gradient magnitude |grad(phi)| using central differences for consistency with curvature
                        double gx_c = 0.5 * (Dop.dxP + Dop.dxN); // Approx central from upwind parts
                        double gy_c = 0.5 * (Dop.dyP + Dop.dyN);
                        double gz_c = 0.5 * (Dop.dzP + Dop.dzN);
                        double grad_mag = std::sqrt(gx_c*gx_c + gy_c*gy_c + gz_c*gz_c + epsilon_grad_mag);
                        
                        curvature_flow_term = CURVATURE_WEIGHT * mean_k * grad_mag;
                    }
                    
                    L_phi[idx] = -advection_term - curvature_flow_term; // d(phi)/dt = -V.grad(phi) - alpha*K*|grad(phi)|
                }
                return L_phi;
            };
            
            // Advance phi in time using the chosen time integration scheme
            phi = timeScheme->advance(phi, levelSetOperator);
            
            // Periodic reinitialization and narrow band update
            if ((step + 1) % REINIT_INTERVAL == 0 && step < STEPS -1) { // Avoid reinit on last step if not needed
                std::cout << "Reinitializing SDF at step " << step << std::endl;
                reinitialize(); // Reinitialize phi to be a signed distance function
            }

            if ((step + 1) % NARROW_BAND_UPDATE_INTERVAL == 0 && step < STEPS -1) {
                std::cout << "Updating narrow band at step " << step << std::endl;
                updateNarrowBand();
                precomputeNarrowBandData(); // Recompute cached data for the new narrow band
            }
        }
        
        std::cout << "Evolution completed successfully after " << STEPS << " steps." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during evolution: " << e.what() << std::endl;
        return false;
    }
}

void LevelSetMethod::reinitialize() {
    Eigen::VectorXd phi_current = phi; // Work on a copy
    
    const int REINIT_PSEUDO_STEPS = 10; // Number of iterations for reinitialization
    // Pseudo time step for reinitialization, dtau ~ GRID_SPACING
    const double dtau = 0.5 * GRID_SPACING; // Should satisfy CFL: dtau < h
    constexpr double epsilon_reinit = 1e-12; // For |grad(phi)| in denominator and S(phi_0)
    
    // Temporary storage for phi at each pseudo-step if needed, or update in place carefully
    Eigen::VectorXd phi_next_reinit = phi_current;

    for (int step = 0; step < REINIT_PSEUDO_STEPS; ++step) {
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t k = 0; k < narrowBandIndices.size(); ++k) {
            const int idx = narrowBandIndices[k];

            if (isOnBoundary(idx, 2)) { // Avoid reinitializing too close to boundary
                phi_next_reinit[idx] = phi_current[idx]; // Keep original value
                continue;
            }

            // Original phi value (at t=0 of reinitialization) to determine sign
            const double phi0_at_idx = phi[idx]; // Use the phi from before reinit iterations started
            const double sign_phi0 = phi0_at_idx / std::sqrt(phi0_at_idx * phi0_at_idx + GRID_SPACING * GRID_SPACING * epsilon_reinit); // Smoothed sign
            // const double sign_phi0 = (phi0_at_idx > 0) ? 1.0 : ((phi0_at_idx < 0) ? -1.0 : 0.0);


            // Godunov scheme for |grad(phi)| for Hamilton-Jacobi equation S(phi_0)(|grad(phi)| - 1) = 0
            // Derivatives are taken from phi_current (previous pseudo-time step)
            // D^-x, D^+x etc.
            double dx_neg = (phi_current[idx] - phi_current[getIndexSafe((idx % GRID_SIZE) - 1, (idx / GRID_SIZE) % GRID_SIZE, idx / (GRID_SIZE*GRID_SIZE))]) / GRID_SPACING;
            double dx_pos = (phi_current[getIndexSafe((idx % GRID_SIZE) + 1, (idx / GRID_SIZE) % GRID_SIZE, idx / (GRID_SIZE*GRID_SIZE))] - phi_current[idx]) / GRID_SPACING;
            double dy_neg = (phi_current[idx] - phi_current[getIndexSafe((idx % GRID_SIZE), ((idx / GRID_SIZE) % GRID_SIZE) - 1, idx / (GRID_SIZE*GRID_SIZE))]) / GRID_SPACING;
            double dy_pos = (phi_current[getIndexSafe((idx % GRID_SIZE), ((idx / GRID_SIZE) % GRID_SIZE) + 1, idx / (GRID_SIZE*GRID_SIZE))] - phi_current[idx]) / GRID_SPACING;
            double dz_neg = (phi_current[idx] - phi_current[getIndexSafe((idx % GRID_SIZE), (idx / GRID_SIZE) % GRID_SIZE, (idx / (GRID_SIZE*GRID_SIZE)) - 1)]) / GRID_SPACING;
            double dz_pos = (phi_current[getIndexSafe((idx % GRID_SIZE), (idx / GRID_SIZE) % GRID_SIZE, (idx / (GRID_SIZE*GRID_SIZE)) + 1)] - phi_current[idx]) / GRID_SPACING;

            double grad_phi_godunov_sq = 0.0;
            if (sign_phi0 > 0) { // Outward pointing normal: use backward differences for positive parts, forward for negative
                grad_phi_godunov_sq += std::pow(std::max(dx_neg, 0.0), 2) + std::pow(std::min(dx_pos, 0.0), 2);
                grad_phi_godunov_sq += std::pow(std::max(dy_neg, 0.0), 2) + std::pow(std::min(dy_pos, 0.0), 2);
                grad_phi_godunov_sq += std::pow(std::max(dz_neg, 0.0), 2) + std::pow(std::min(dz_pos, 0.0), 2);
            } else if (sign_phi0 < 0) { // Inward pointing normal
                grad_phi_godunov_sq += std::pow(std::min(dx_neg, 0.0), 2) + std::pow(std::max(dx_pos, 0.0), 2);
                grad_phi_godunov_sq += std::pow(std::min(dy_neg, 0.0), 2) + std::pow(std::max(dy_pos, 0.0), 2);
                grad_phi_godunov_sq += std::pow(std::min(dz_neg, 0.0), 2) + std::pow(std::max(dz_pos, 0.0), 2);
            }
            // If sign_phi0 is 0, grad_phi_godunov_sq remains 0, so d_phi/d_tau = 0.

            double grad_phi_mag = std::sqrt(grad_phi_godunov_sq + epsilon_reinit); // Add epsilon for stability
            
            phi_next_reinit[idx] = phi_current[idx] - dtau * sign_phi0 * (grad_phi_mag - 1.0);
        }
        phi_current.swap(phi_next_reinit); // Update for next iteration
    }
    phi = phi_current; // Assign reinitialized SDF back
    std::cout << "Reinitialization complete." << std::endl;
}


void LevelSetMethod::updateNarrowBand() {
    narrowBandIndices.clear();
    // Estimate narrow band size: surface area ~ (GRID_SIZE)^2, width ~ NARROW_BAND_WIDTH_CELLS
    // This is a rough estimate.
    size_t estimated_size = static_cast<size_t>(6 * GRID_SIZE * GRID_SIZE * NARROW_BAND_WIDTH_CELLS / GRID_SIZE); 
    if (estimated_size == 0) estimated_size = gridPoints.size() / 10; // Fallback if too small
    narrowBandIndices.reserve(std::min(gridPoints.size(), estimated_size)); 
    
    const double narrow_band_distance_threshold = NARROW_BAND_WIDTH_CELLS * GRID_SPACING;
    
    // Using thread-local vectors to collect indices, then merge.
    std::vector<std::vector<int>> thread_local_bands(omp_get_max_threads());
    for(auto& vec : thread_local_bands) { // Pre-reserve in thread-local vectors
        vec.reserve(estimated_size / omp_get_max_threads() + 1);
    }

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 8192) // Dynamic with reasonably large chunks
        for (size_t i = 0; i < gridPoints.size(); ++i) {
            // Check if point is within the narrow band distance and not too close to physical boundary
            if (std::abs(phi[i]) <= narrow_band_distance_threshold && !isOnBoundary(i, 1)) {
                thread_local_bands[thread_id].push_back(i);
            }
        }
    }
    
    // Merge thread-local vectors
    size_t total_narrow_band_size = 0;
    for (const auto& local_band : thread_local_bands) {
        total_narrow_band_size += local_band.size();
    }
    narrowBandIndices.reserve(total_narrow_band_size); // Ensure enough capacity
    for (const auto& local_band : thread_local_bands) {
        narrowBandIndices.insert(narrowBandIndices.end(), local_band.begin(), local_band.end());
    }
    
    // Optional: Sort and unique if duplicate indices could occur (not expected with current omp for)
    // std::sort(std::execution::par_unseq, narrowBandIndices.begin(), narrowBandIndices.end());
    // narrowBandIndices.erase(std::unique(narrowBandIndices.begin(), narrowBandIndices.end()), narrowBandIndices.end());
    // For most operations, sorting is not strictly necessary unless a specific order is assumed later.
    // If sorting is beneficial, parallel sort can be used for large narrow bands.
    // std::sort(std::execution::par, narrowBandIndices.begin(), narrowBandIndices.end());


    std::cout << "Narrow band updated. Size: " << narrowBandIndices.size() 
              << " (" << (narrowBandIndices.empty() ? 0.0 : (narrowBandIndices.size() * 100.0 / gridPoints.size())) << "% of grid)" << std::endl;
    if (narrowBandIndices.empty() && !gridPoints.empty()) {
        std::cerr << "Warning: Narrow band is empty. This might indicate an issue with SDF or parameters." << std::endl;
    }
}

void LevelSetMethod::generateGrid() {
    if (mesh.is_empty()) {
        throw std::runtime_error("Mesh not loaded - cannot generate grid.");
    }
    
    CGAL::Bbox_3 bbox = calculateBoundingBox();
    // Add padding (e.g., 5-10% of max dimension or a few NARROW_BAND_WIDTH_CELLS)
    // This ensures the evolving surface has space and narrow band doesn't hit the boundary too soon.
    double max_extent = std::max({bbox.xmax()-bbox.xmin(), bbox.ymax()-bbox.ymin(), bbox.zmax()-bbox.zmin()});
    if (max_extent == 0) max_extent = 1.0; // Handle degenerate mesh bbox
    double padding = std::max(0.1 * max_extent, 2.0 * NARROW_BAND_WIDTH_CELLS * (max_extent / GRID_SIZE) );


    Point_3 min_pt(bbox.xmin() - padding, bbox.ymin() - padding, bbox.zmin() - padding);
    Point_3 max_pt(bbox.xmax() + padding, bbox.ymax() + padding, bbox.zmax() + padding);
    
    gridOrigin = min_pt; // Store the origin of the grid

    // Calculate grid spacing based on the largest dimension of the padded box
    double sim_box_x = max_pt.x() - min_pt.x();
    double sim_box_y = max_pt.y() - min_pt.y();
    double sim_box_z = max_pt.z() - min_pt.z();
    
    BOX_SIZE = std::max({sim_box_x, sim_box_y, sim_box_z}); // Physical size of the cubic simulation domain
    GRID_SPACING = BOX_SIZE / (GRID_SIZE - 1); // Physical distance between grid points
    
    gridPoints.clear();
    gridPoints.resize(GRID_SIZE * GRID_SIZE * GRID_SIZE); // Pre-allocate
    
    #pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < GRID_SIZE; ++z) {
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                double px = min_pt.x() + x * GRID_SPACING;
                double py = min_pt.y() + y * GRID_SPACING;
                double pz = min_pt.z() + z * GRID_SPACING;
                gridPoints[getIndexUnsafe(x,y,z)] = Point_3(px, py, pz);
            }
        }
    }
    std::cout << "Grid generated. Size: " << GRID_SIZE << "x" << GRID_SIZE << "x" << GRID_SIZE 
              << ", Spacing: " << GRID_SPACING << ", Origin: " << gridOrigin << std::endl;
}


Eigen::VectorXd LevelSetMethod::initializeSignedDistanceField() {
    if (!tree || mesh.is_empty()) {
        throw std::runtime_error("AABB tree not initialized or mesh empty. Load a mesh first.");
    }
    
    std::cout << "Initializing signed distance field..." << std::endl;
    
    const size_t num_grid_points = gridPoints.size();
    Eigen::VectorXd sdf(num_grid_points);
    
    // CGAL's Side_of_triangle_mesh for inside/outside queries
    // It's generally thread-safe for queries after construction.
    CGAL::Side_of_triangle_mesh<Mesh, Kernel> inside_tester(mesh);
    
    std::atomic<size_t> progress_counter(0);
    const size_t report_interval = std::max(size_t(1), num_grid_points / 100); // Report every 1%

    #pragma omp parallel for schedule(dynamic, 2048) // Dynamic with a good chunk size
    for (size_t i = 0; i < num_grid_points; ++i) {
        const Point_3& point = gridPoints[i];
        
        // CGAL AABB tree query for distance
        // Using `sqrt` on squared_distance is fine. `tree->distance(point)` could also be used.
        double dist = CGAL::sqrt(tree->squared_distance(point));
        
        // Determine sign
        CGAL::Bounded_side side = inside_tester(point);
        double sign = 0.0;
        if (side == CGAL::ON_BOUNDED_SIDE) { // Inside
            sign = -1.0;
        } else if (side == CGAL::ON_UNBOUNDED_SIDE) { // Outside
            sign = 1.0;
        } else { // On boundary
            sign = 0.0; 
            dist = 0.0; // Ensure distance is zero if on boundary
        }
        
        sdf[i] = sign * dist;
        
        // Progress reporting (atomic operation, so keep it somewhat infrequent)
        if (report_interval > 0 && (progress_counter.fetch_add(1, std::memory_order_relaxed) + 1) % report_interval == 0) {
             std::cout << "\rSDF Initialization: " 
                       << (progress_counter.load(std::memory_order_relaxed) * 100 / num_grid_points) << "%" << std::flush;
        }
    }
    
    std::cout << "\rSDF Initialization: 100% Complete.                 " << std::endl;
    return sdf;
}

// Check if a grid point (by 1D index) is on the physical boundary of the simulation grid
bool LevelSetMethod::isOnBoundary(int idx, int boundaryThickness) const {
    // boundaryThickness: how many layers of cells from the edge are considered "boundary"
    if (boundaryThickness < 1) boundaryThickness = 1;

    const int x = idx % GRID_SIZE;
    const int y = (idx / GRID_SIZE) % GRID_SIZE;
    const int z = idx / (GRID_SIZE * GRID_SIZE); // Integer division
    
    return (x < boundaryThickness || x >= GRID_SIZE - boundaryThickness ||
            y < boundaryThickness || y >= GRID_SIZE - boundaryThickness ||
            z < boundaryThickness || z >= GRID_SIZE - boundaryThickness);
}


// Implicit function for CGAL surface mesher
class LevelSetImplicitFunctionForCGAL {
public:
    LevelSetImplicitFunctionForCGAL(const std::vector<Point_3>& grid_pts, 
                                    const Eigen::VectorXd& phi_sdf, 
                                    int N, double h, Point_3 origin)
        : grid_points_ref(grid_pts), phi_ref(phi_sdf), 
          GRID_SIZE_N(N), GRID_SPACING_H(h), grid_origin_pt(origin) {
        // Precompute inverse spacing for performance
        inv_GRID_SPACING_H = (GRID_SPACING_H > 1e-9) ? 1.0 / GRID_SPACING_H : 0.0;
    }

    // Operator called by CGAL's surface mesher
    // It needs to return the SDF value at point p
    double operator()(const Point_3& p) const {
        // Transform point p to grid coordinates (i, j, k)
        // These can be fractional
        double fx = (p.x() - grid_origin_pt.x()) * inv_GRID_SPACING_H;
        double fy = (p.y() - grid_origin_pt.y()) * inv_GRID_SPACING_H;
        double fz = (p.z() - grid_origin_pt.z()) * inv_GRID_SPACING_H;

        // Get integer indices of the cell containing p (bottom-left-back corner)
        int i0 = static_cast<int>(std::floor(fx));
        int j0 = static_cast<int>(std::floor(fy));
        int k0 = static_cast<int>(std::floor(fz));

        // Check if the point is outside the grid bounds for interpolation
        // If so, extrapolate (e.g., return a large positive value or value of nearest boundary cell)
        if (i0 < 0 || i0 >= GRID_SIZE_N - 1 ||
            j0 < 0 || j0 >= GRID_SIZE_N - 1 ||
            k0 < 0 || k0 >= GRID_SIZE_N - 1) {
            
            // Simplistic: clamp to nearest grid point's value
            int ci = std::max(0, std::min(i0, GRID_SIZE_N - 1));
            int cj = std::max(0, std::min(j0, GRID_SIZE_N - 1));
            int ck = std::max(0, std::min(k0, GRID_SIZE_N - 1));
            return phi_ref[ck * GRID_SIZE_N * GRID_SIZE_N + cj * GRID_SIZE_N + ci]; // Access using 1D index
        }

        // Fractional parts for trilinear interpolation
        double u = fx - i0;
        double v = fy - j0;
        double w = fz - k0;

        // Indices of the 8 corners of the cell
        // (i0,j0,k0), (i0+1,j0,k0), ..., (i0+1,j0+1,k0+1)
        // Using a helper for 1D index conversion
        auto get_phi = [&](int i, int j, int k) {
            return phi_ref[k * GRID_SIZE_N * GRID_SIZE_N + j * GRID_SIZE_N + i];
        };

        // Trilinear interpolation
        double val = (1-u)*(1-v)*(1-w) * get_phi(i0,   j0,   k0) +
                     u*(1-v)*(1-w)     * get_phi(i0+1, j0,   k0) +
                     (1-u)*v*(1-w)     * get_phi(i0,   j0+1, k0) +
                     (1-u)*(1-v)*w     * get_phi(i0,   j0,   k0+1) +
                     u*v*(1-w)         * get_phi(i0+1, j0+1, k0) +
                     u*(1-v)*w         * get_phi(i0+1, j0,   k0+1) +
                     (1-u)*v*w         * get_phi(i0,   j0+1, k0+1) +
                     u*v*w             * get_phi(i0+1, j0+1, k0+1);
        return val;
    }

private:
    const std::vector<Point_3>& grid_points_ref; // Reference to grid points (not directly used if origin, N, H known)
    const Eigen::VectorXd& phi_ref;       // Reference to SDF values
    const int GRID_SIZE_N;                // Number of grid points in one dimension
    const double GRID_SPACING_H;          // Grid spacing
    const Point_3 grid_origin_pt;         // Physical coordinate of grid point (0,0,0)
    double inv_GRID_SPACING_H;            // Precomputed 1.0 / GRID_SPACING_H
};

bool LevelSetMethod::extractSurfaceMeshCGAL(const std::string& filename) {
    try {
        if (phi.size() != gridPoints.size()) {
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
        LevelSetImplicitFunction implicitFunction(gridPoints, phi, GRID_SIZE, GRID_SPACING);

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



void LevelSetMethod::loadMaterialInfo(const std::string& csvFilename, const std::string& orgMeshFilename) {
    std::cout << "Loading material information from CSV: " << csvFilename 
              << " and original mesh: " << orgMeshFilename << std::endl;
    
    std::unordered_map<int, std::string> face_to_material_map; // Maps face index in orgMesh to material name
    std::ifstream csvFile(csvFilename);
    if (!csvFile.is_open()) {
        throw std::runtime_error("Failed to open material CSV file: " + csvFilename);
    }
    
    std::string line;
    int line_num = 0;
    while (std::getline(csvFile, line)) {
        line_num++;
        if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments
        
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> segments;
        while(std::getline(ss, segment, ',')) {
           segments.push_back(segment);
        }
        if (segments.size() >= 2) {
            try {
                int faceIdx = std::stoi(segments[0]);
                std::string materialName = segments[1];
                // Trim whitespace from materialName if any
                materialName.erase(0, materialName.find_first_not_of(" \t\n\r\f\v"));
                materialName.erase(materialName.find_last_not_of(" \t\n\r\f\v") + 1);
                face_to_material_map[faceIdx] = materialName;
            } catch (const std::invalid_argument& ia) {
                std::cerr << "Warning: Invalid number format for face index in CSV line " << line_num << ": " << segments[0] << std::endl;
            } catch (const std::out_of_range& oor) {
                std::cerr << "Warning: Face index out of range in CSV line " << line_num << ": " << segments[0] << std::endl;
            }
        } else {
            std::cerr << "Warning: Malformed CSV line " << line_num << ": " << line << std::endl;
        }
    }
    csvFile.close();
    std::cout << "Loaded " << face_to_material_map.size() << " face-material mappings from CSV." << std::endl;

    Mesh org_mesh; // The original mesh defining material regions
    if (!PMP::IO::read_polygon_mesh(orgMeshFilename, org_mesh) || org_mesh.is_empty() || !CGAL::is_triangle_mesh(org_mesh)) {
        throw std::runtime_error("Failed to read original mesh for material info: " + orgMeshFilename);
    }
    
    // AABB tree for the original mesh
    AABB_tree org_mesh_tree(faces(org_mesh).begin(), faces(org_mesh).end(), org_mesh);
    org_mesh_tree.accelerate_distance_queries();
    
    gridCellMaterials.resize(gridPoints.size());
    const std::string default_material_name = "default"; // Fallback material

    std::atomic<size_t> unmapped_points(0);

    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < gridPoints.size(); ++i) {
        const Point_3& pt = gridPoints[i];
        // Find the closest face on the original mesh to this grid point
        auto closest_primitive = org_mesh_tree.closest_point_and_primitive(pt);
        Mesh::Face_index closest_face_idx = closest_primitive.second; // This is a face_descriptor
        
        // The .id() method for face_descriptor might not be standard or could be tricky.
        // CGAL face_descriptors are typically iterators or handles.
        // To get a unique integer ID, one common way is to iterate and assign them if not built-in.
        // However, if PMP::IO::read_polygon_mesh preserves original face indices from some formats,
        // or if the CSV indices directly correspond to an iteration order, that's how it's matched.
        // Assuming the CSV face indices are 0-based and correspond to iteration order of faces(org_mesh).
        // This part is CRITICAL and depends on how face indices in CSV are defined.
        // A robust way is to map face_descriptor to an int ID if CGAL doesn't provide one directly.
        // For now, let's assume face_descriptor can be cast or converted to an int that matches CSV.
        // This is a common source of error if indexing schemes don't match.
        // A safer approach if `closest_face_idx.id()` is not available or reliable:
        // Iterate through all faces of org_mesh once, store their descriptors in a map to int index.
        // int face_id = -1; // Placeholder
        // For simplicity, assuming `closest_face_idx` can be used with `face_to_material_map`
        // This requires `Mesh::Face_index` to be usable as a key or convertible to one.
        // If `Mesh::Face_index` is an iterator, you might need `std::distance(faces(org_mesh).begin(), closest_face_idx)`.
        
        // Let's assume `closest_face_idx` gives an index compatible with the CSV.
        // This is a BIG assumption. A common way is that the CSV refers to faces by their
        // order in the mesh file (e.g., 0th face, 1st face, etc.).
        // If `CGAL::SM_Face_index` is the type, it has an `idx()` method.
        int face_id_for_lookup = static_cast<int>(closest_face_idx.idx());


        auto mat_it = face_to_material_map.find(face_id_for_lookup);
        if (mat_it != face_to_material_map.end()) {
            gridCellMaterials[i] = mat_it->second;
        } else {
            gridCellMaterials[i] = default_material_name;
            // Only report unmapped points once to avoid console spam
            if (unmapped_points.fetch_add(1, std::memory_order_relaxed) == 0 && face_id_for_lookup != -1) {
                 // std::cerr << "Warning: Material not found for face index " << face_id_for_lookup 
                 //           << " (closest to grid point " << i << "). Using default." << std::endl;
            }
        }
    }
    if (unmapped_points.load() > 0) {
        std::cout << "Info: " << unmapped_points.load() << " grid points were mapped to default material due to missing face index in CSV or other mapping issues." << std::endl;
    }
    std::cout << "Material information loaded for grid cells." << std::endl;
}

std::string LevelSetMethod::getMaterialAtPoint(int gridIndex) const {
    if (gridIndex >= 0 && gridIndex < static_cast<int>(gridCellMaterials.size())) {
        return gridCellMaterials[gridIndex];
    }
    // std::cerr << "Warning: gridIndex " << gridIndex << " out of bounds for gridCellMaterials. Returning default." << std::endl;
    return "default"; // Fallback material
}
