#ifndef __TIME_SCHEME_HPP__
#define __TIME_SCHEME_HPP__

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

class TimeScheme {
public:
    TimeScheme(double timeStep, double GRID_SPACING) : dt(timeStep), dx(GRID_SPACING) {}
    virtual ~TimeScheme() = default;
    
    virtual Eigen::VectorXd advance(
        const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
        const Eigen::VectorXd& phi, 
        const Eigen::VectorXd& Ux,
        const Eigen::VectorXd& Uy,
        const Eigen::VectorXd& Uz,
        int gridSize) = 0;
    virtual Eigen::SparseMatrix<double> GenMatrixA(
        const Eigen::VectorXd& phi,
        const Eigen::VectorXd& Ux, 
        const Eigen::VectorXd& Uy, 
        const Eigen::VectorXd& Uz,
        double spacing,
        int gridSize) = 0;
    
protected:
    const double dt;
    const double dx;
};

class BackwardEulerScheme : public TimeScheme {
public:
    BackwardEulerScheme(double timeStep, double GRID_SPACING = 1.0) 
        : TimeScheme(timeStep, GRID_SPACING) {}
    
    Eigen::SparseMatrix<double> GenMatrixA(const Eigen::VectorXd& phi, 
        const Eigen::VectorXd& Ux, 
        const Eigen::VectorXd& Uy, 
        const Eigen::VectorXd& Uz,
        double spacing,
        int gridSize) override {
       
            const int n = phi.size();
        
            // Create the system matrix for implicit time stepping
            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletList;
            tripletList.reserve(7 * n); // Each row has at most 7 non-zero entries (center + 6 neighbors)
            
            // Build the linear system (I - dt*L)phi^{n+1} = phi^n
            // Use thread-local storage to avoid critical section
            const int num_threads = omp_get_max_threads();
            std::vector<std::vector<T>> thread_triplets(num_threads);
            
            #pragma omp parallel
            {
                const int thread_id = omp_get_thread_num();
                thread_triplets[thread_id].reserve(7 * n / num_threads);
                
                #pragma omp for nowait
                for (int idx = 0; idx < n; idx++) {
                    int x = idx % gridSize;
                    int y = (idx / gridSize) % gridSize;
                    int z = idx / (gridSize * gridSize);
                    
                    // Check if this is a boundary point
                    bool isBoundary = (x == 0 || x == gridSize-1 || 
                                      y == 0 || y == gridSize-1 || 
                                      z == 0 || z == gridSize-1);
                    
                    if (isBoundary) {
                        // For boundary points, use identity equation (phi^{n+1} = phi^n)
                        thread_triplets[thread_id].emplace_back(idx, idx, 1.0);
                    } else {
                        // Diagonal term: 1 + dt*regularization
                        double diagTerm = 1.0;
    
                        // Add connections to neighboring cells based on velocity direction
                        // X direction
                        if (Ux(idx) <= 0) { // Flow from right to left
                            diagTerm -= dt * Ux(idx) / spacing; 
                            thread_triplets[thread_id].emplace_back(idx, idx+1, dt * Ux(idx) / spacing);
                        } else if (Ux(idx) > 0) { // Flow from left to right
                            diagTerm += dt * Ux(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx-1, -dt * Ux(idx) / spacing);
                        }
                        
                        // Y direction
                        if (Uy(idx) <= 0) { // Flow from top to bottom
                            diagTerm -= dt * Uy(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx+gridSize, dt * Uy(idx) / spacing);
                        } else if (Uy(idx) > 0) { // Flow from bottom to top
                            diagTerm += dt * Uy(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx-gridSize, -dt * Uy(idx) / spacing);
                        }
                        
                        // Z direction
                        if (Uz(idx) <= 0) { // Flow from front to back
                            diagTerm -= dt * Uz(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx+gridSize*gridSize, dt * Uz(idx) / spacing);
                        } else if (Uz(idx) > 0) { // Flow from back to front
                            diagTerm += dt * Uz(idx) / spacing;
                            thread_triplets[thread_id].emplace_back(idx, idx-gridSize*gridSize, -dt * Uz(idx) / spacing);
                        }
    
                        thread_triplets[thread_id].emplace_back(idx, idx, diagTerm);
                    }
                }
            }
            
            // Merge thread-local triplet lists
            size_t total_triplets = 0;
            for (const auto& thread_list : thread_triplets) {
                total_triplets += thread_list.size();
            }
            tripletList.reserve(total_triplets);
            
            for (const auto& thread_list : thread_triplets) {
                tripletList.insert(tripletList.end(), thread_list.begin(), thread_list.end());
            }
            
            // Create sparse matrix from triplets
            Eigen::SparseMatrix<double> A(n, n);
            A.setFromTriplets(tripletList.begin(), tripletList.end());
            return A;
    }
    

    Eigen::VectorXd advance(
        const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
        const Eigen::VectorXd& phi, 
        const Eigen::VectorXd& Ux,
        const Eigen::VectorXd& Uy,
        const Eigen::VectorXd& Uz,
        int gridSize 
    ) override {
        
        Eigen::VectorXd b = phi;
        Eigen::VectorXd phi_next;
        
        phi_next = solveStandard(A, b);
        
        return phi_next;
    }
    
private:
    
    // Standard solver method
    Eigen::VectorXd solveStandard(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, const Eigen::VectorXd& b) {
        // Use BiCGSTAB solver which is more robust for non-symmetric matrices
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
        
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
        Eigen::VectorXd x = solver.solve(b);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed with error: " << solver.error() << std::endl;
            std::cerr << "Iterations: " << solver.iterations() << ", estimated error: " << solver.error() << std::endl;
            throw std::runtime_error("Solver failed");
        }
        
        return x;
    }

};


class implicitCN : public TimeScheme {
public:
    implicitCN(double timeStep, double GRID_SPACING = 1.0) 
        : TimeScheme(timeStep, GRID_SPACING) {}

    Eigen::SparseMatrix<double> GenMatrixA(const Eigen::VectorXd& phi,
            const Eigen::VectorXd& Ux,
            const Eigen::VectorXd& Uy,
            const Eigen::VectorXd& Uz,
            double spacing,
            int gridSize) override {
    
        const int n = phi.size();
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(7 * n);
    
        const int num_threads = omp_get_max_threads();
        std::vector<std::vector<T>> thread_triplets(num_threads);
    
        const double epsilon = 1e-6;
    
        #pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
            thread_triplets[thread_id].reserve(7 * n / num_threads);
    
            #pragma omp for nowait
            for (int idx = 0; idx < n; idx++) {
                int x = idx % gridSize;
                int y = (idx / gridSize) % gridSize;
                int z = idx / (gridSize * gridSize);
    
                bool isBoundary = (x <= 1 || x >= gridSize - 3 ||
                                   y <= 1 || y >= gridSize - 3 ||
                                   z <= 1 || z >= gridSize - 3);
    
                if (isBoundary) {
                    thread_triplets[thread_id].emplace_back(idx, idx, 1.0 );
                    continue;
                }
    
                double diagTerm = 1.0;
    
                // --- X-direction ---
                int idxL = idx - 1;
                int idxR = idx + 1;
                if (x > 0 && x < gridSize - 1) {
                    double aL = 0.5 * (Ux(idxL) + Ux(idx));
                    double aR = 0.5 * (Ux(idx) + Ux(idxR));
    
                    double fluxL = dt * 0.5 * (aL + std::abs(aL) + epsilon) / (2.0 * spacing);
                    double fluxR = dt * 0.5 * (aR - std::abs(aR) - epsilon) / (2.0 * spacing);
    
                    thread_triplets[thread_id].emplace_back(idx, idxL, -fluxL);
                    thread_triplets[thread_id].emplace_back(idx, idxR, fluxR);
                    diagTerm += fluxL - fluxR;
                }
    
                // --- Y-direction ---
                int idxB = idx - gridSize;
                int idxT = idx + gridSize;
                if (y > 0 && y < gridSize - 1) {
                    double aB = 0.5 * (Uy(idxB) + Uy(idx));
                    double aT = 0.5 * (Uy(idx) + Uy(idxT));
    
                    double fluxB = dt * 0.5 * (aB + std::abs(aB) + epsilon) / (2.0 * spacing);
                    double fluxT = dt * 0.5 * (aT - std::abs(aT) - epsilon) / (2.0 * spacing);
    
                    thread_triplets[thread_id].emplace_back(idx, idxB, -fluxB);
                    thread_triplets[thread_id].emplace_back(idx, idxT, fluxT);
                    diagTerm += fluxB - fluxT;
                }
    
                // --- Z-direction ---
                int idxD = idx - gridSize * gridSize;
                int idxU = idx + gridSize * gridSize;
                if (z > 0 && z < gridSize - 1) {
                    double aD = 0.5 * (Uz(idxD) + Uz(idx));
                    double aU = 0.5 * (Uz(idx) + Uz(idxU));
    
                    double fluxD = dt * 0.5 * (aD + std::abs(aD) + epsilon) / (2.0 * spacing);
                    double fluxU = dt * 0.5 * (aU - std::abs(aU) - epsilon) / (2.0 * spacing);
    
                    thread_triplets[thread_id].emplace_back(idx, idxD, -fluxD);
                    thread_triplets[thread_id].emplace_back(idx, idxU, fluxU);
                    diagTerm += fluxD - fluxU;
                }
    
                thread_triplets[thread_id].emplace_back(idx, idx, diagTerm);
            }
        }
    
        for (const auto& thread_list : thread_triplets) {
            tripletList.insert(tripletList.end(), thread_list.begin(), thread_list.end());
        }
    
        Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
        A.setFromTriplets(tripletList.begin(), tripletList.end());
        return A;
    }

    Eigen::VectorXd advance(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
                            const Eigen::VectorXd& phi_n, 
                            const Eigen::VectorXd& Ux,
                            const Eigen::VectorXd& Uy,
                            const Eigen::VectorXd& Uz,
                            int gridSize) override {
        const int n = phi_n.size();
        double spacing = dx; 
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n);

        #pragma omp parallel for
        for (int idx = 0; idx < n; idx++) {
            int x = idx % gridSize;
            int y = (idx / gridSize) % gridSize;
            int z = idx / (gridSize * gridSize);

            bool isBoundary = (x <= 1 || x >= gridSize - 3 ||
                               y <= 1 || y >= gridSize - 3 ||
                               z <= 1 || z >= gridSize - 3);

            if (isBoundary) {
                b(idx) = phi_n(idx);  // Dirichlet BC
            } else {
                // Face fluxes
                double fxL = computeRoeMUSCLFlux(phi_n, Ux, idx - 1, gridSize, spacing, 0);
                double fxR = computeRoeMUSCLFlux(phi_n, Ux, idx,     gridSize, spacing, 0);
                double fyL = computeRoeMUSCLFlux(phi_n, Uy, idx - gridSize, gridSize, spacing, 1);
                double fyR = computeRoeMUSCLFlux(phi_n, Uy, idx,            gridSize, spacing, 1);
                double fzL = computeRoeMUSCLFlux(phi_n, Uz, idx - gridSize * gridSize, gridSize, spacing, 2);
                double fzR = computeRoeMUSCLFlux(phi_n, Uz, idx,                     gridSize, spacing, 2);

                b(idx) = phi_n(idx) - (dt / (2.0 * spacing)) * ((fxR - fxL) + (fyR - fyL) + (fzR - fzL));
            }
        }

        Eigen::VectorXd phi_np1 = solveStandard(A, b);
        return phi_np1;
    }

private:
    double computeRoeMUSCLFlux(const Eigen::VectorXd& phi, const Eigen::VectorXd& U,
                               int idx, int gridSize, double spacing, int direction) {
        const int n = phi.size();
        int stride = (direction == 0) ? 1 : (direction == 1) ? gridSize : gridSize * gridSize;

        int idx_m1 = idx - stride;
        int idx_p1 = idx + stride;
        int idx_p2 = idx + 2 * stride;

        // Spatially aware boundary check
        int x = idx % gridSize;
        int y = (idx / gridSize) % gridSize;
        int z = idx / (gridSize * gridSize);
        if (x <= 1 || x >= gridSize - 3 ||
            y <= 1 || y >= gridSize - 3 ||
            z <= 1 || z >= gridSize - 3 )
            return 0.0;

        // MUSCL reconstruction
        double slope_L = minmod((phi[idx] - phi[idx_m1]) / spacing,
                                (phi[idx_p1] - phi[idx]) / spacing);
        double slope_R = minmod((phi[idx_p1] - phi[idx]) / spacing,
                                (phi[idx_p2] - phi[idx_p1]) / spacing);

        double phi_L = phi[idx] + 0.5 * slope_L * spacing;
        double phi_R = phi[idx_p1] - 0.5 * slope_R * spacing;

        double u_avg = 0.5 * (U[idx] + U[idx_p1]);
        double flux = 0.5 * (u_avg * (phi_L + phi_R) - std::abs(u_avg) * (phi_R - phi_L));
        return flux;
    }


    double minmod(double a, double b) {
        if (a * b <= 0) return 0;
        return std::abs(a) < std::abs(b) ? a : b;
    }

    Eigen::VectorXd solveStandard(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, 
                                  const Eigen::VectorXd& b) {
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver;
        solver.setMaxIterations(1000);
        solver.setTolerance(1e-8);
        
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Matrix decomposition failed" << std::endl;
            throw std::runtime_error("Decomposition failed");
        }
        
        Eigen::VectorXd x = solver.solve(b);
        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed. Iterations: " << solver.iterations() 
                      << ", Error: " << solver.error() << std::endl;
            throw std::runtime_error("Solver failed");
        }
        
        return x;
    }
};



// TVD Runge-Kutta 3rd order scheme with WENO3 spatial discretization
class TVDRK3WENO3Scheme : public TimeScheme {
public:
    TVDRK3WENO3Scheme(double timeStep, double gridSpacing = 1.0)
        : TimeScheme(timeStep, gridSpacing), eps(1e-6) {}

    Eigen::SparseMatrix<double> GenMatrixA(
        const Eigen::VectorXd& phi,
        const Eigen::VectorXd& Ux,
        const Eigen::VectorXd& Uy,
        const Eigen::VectorXd& Uz,
        double spacing,
        int gridSize) override {
        return Eigen::SparseMatrix<double>(phi.size(), phi.size());
    }

    Eigen::VectorXd advance(
        const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
        const Eigen::VectorXd& phi,
        const Eigen::VectorXd& Ux,
        const Eigen::VectorXd& Uy,
        const Eigen::VectorXd& Uz,
        int gridSize) override {

        // TVD-RK3 time integration
        // Stage 1: phi^(1) = phi^n + dt * L(phi^n)
        Eigen::VectorXd L_phi = computeWENO3LevelSetRHS(phi, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi1 = phi + dt * L_phi;

        // Stage 2: phi^(2) = 3/4 * phi^n + 1/4 * (phi^(1) + dt * L(phi^(1)))
        Eigen::VectorXd L_phi1 = computeWENO3LevelSetRHS(phi1, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi2 = 0.75 * phi + 0.25 * (phi1 + dt * L_phi1);

        // Stage 3: phi^(n+1) = 1/3 * phi^n + 2/3 * (phi^(2) + dt * L(phi^(2)))
        Eigen::VectorXd L_phi2 = computeWENO3LevelSetRHS(phi2, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi_next = (1.0 / 3.0) * phi + (2.0 / 3.0) * (phi2 + dt * L_phi2);

        return phi_next;
    }

private:
    const double eps; // Small parameter for WENO smoothness indicators

    // Compute the right-hand side for level set equation using WENO3
    Eigen::VectorXd computeWENO3LevelSetRHS(const Eigen::VectorXd& phi,
        const Eigen::VectorXd& Ux,
        const Eigen::VectorXd& Uy,
        const Eigen::VectorXd& Uz,
        int gridSize) {
        const int n = phi.size();
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);

        #pragma omp parallel for
        for (int idx = 0; idx < n; idx++) {
            int x = idx % gridSize;
            int y = (idx / gridSize) % gridSize;
            int z = idx / (gridSize * gridSize);

            // Apply boundary conditions (zero flux at boundaries)
            bool isBoundary = (x <= 1 || x >= gridSize - 2 ||
                y <= 1 || y >= gridSize - 2 ||
                z <= 1 || z >= gridSize - 2);

            if (isBoundary) {
                rhs(idx) = 0.0;
                continue;
            }

            // Compute WENO3 fluxes in each direction
            double flux_x = computeWENO3Flux(phi, Ux(idx), idx, gridSize, 0);
            double flux_y = computeWENO3Flux(phi, Uy(idx), idx, gridSize, 1);
            double flux_z = computeWENO3Flux(phi, Uz(idx), idx, gridSize, 2);

            double flux_xm = computeWENO3Flux(phi, Ux(idx - 1), idx - 1, gridSize, 0);
            double flux_ym = computeWENO3Flux(phi, Uy(idx - gridSize), idx - gridSize, gridSize, 1);
            double flux_zm = computeWENO3Flux(phi, Uz(idx - gridSize * gridSize), idx - gridSize * gridSize, gridSize, 2);

            // Level set equation: phi_t + u · ∇phi = 0
            rhs(idx) = -(flux_x - flux_xm + flux_y - flux_ym + flux_z - flux_zm) / dx;
        }

        return rhs;
    }

    // Compute WENO3 flux in specified direction
    double computeWENO3Flux(const Eigen::VectorXd& phi, double velocity,
        int idx, int gridSize, int direction) {
        // Get stencil indices based on direction
        std::vector<int> stencil = getStencil(idx, gridSize, direction);

        if (stencil.empty()) return 0.0; // Boundary handling

        // Extract values from stencil
        std::vector<double> values(stencil.size());
        for (size_t i = 0; i < stencil.size(); ++i) {
            values[i] = phi(stencil[i]);
        }

        // Compute WENO3 reconstruction
        double flux_pos, flux_neg;
        computeWENO3Reconstruction(values, flux_pos, flux_neg);

        double alpha = std::abs(velocity);
        double flux = 0.5 * (velocity * (flux_pos + flux_neg) - alpha * (flux_pos - flux_neg));

        return flux;
    }

    // Get stencil indices for WENO3 reconstruction
    std::vector<int> getStencil(int idx, int gridSize, int direction) {
        std::vector<int> stencil;
        int stride = (direction == 0) ? 1 :
            (direction == 1) ? gridSize :
            gridSize * gridSize;

        // Get 3D coordinates
        int x = idx % gridSize;
        int y = (idx / gridSize) % gridSize;
        int z = idx / (gridSize * gridSize);

        // Check if we have enough points for WENO3 stencil
        bool canConstruct = true;
        if (direction == 0 && (x < 2 || x > gridSize - 3)) canConstruct = false;
        if (direction == 1 && (y < 2 || y > gridSize - 3)) canConstruct = false;
        if (direction == 2 && (z < 2 || z > gridSize - 3)) canConstruct = false;

        if (!canConstruct) return stencil; // Return empty stencil

        // Build 5-point stencil: [i-2, i-1, i, i+1, i+2]
        for (int offset = -2; offset <= 2; ++offset) {
            stencil.push_back(idx + offset * stride);
        }

        return stencil;
    }

    // WENO3 reconstruction
    void computeWENO3Reconstruction(const std::vector<double>& values,
        double& flux_pos, double& flux_neg) {
        if (values.size() != 5) {
            flux_pos = flux_neg = 0.0;
            return;
        }

        // Extract stencil values: u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}
        double um2 = values[0], um1 = values[1], u0 = values[2], up1 = values[3], up2 = values[4];

        // WENO3 reconstruction for positive flux (upwind from left)
        // Stencils: S0 = {i-1, i}, S1 = {i, i+1}
        double q0_pos = (1.0 / 3.0) * um1 + (5.0 / 6.0) * u0 - (1.0 / 6.0) * up1; // ENO2 on S0
        double q1_pos = (-1.0 / 6.0) * um2 + (5.0 / 6.0) * um1 + (1.0 / 3.0) * u0; // ENO2 on S1

        // Smoothness indicators
        double beta0_pos = (13.0 / 12.0) * std::pow(um1 - 2.0 * u0 + up1, 2) + 0.25 * std::pow(um1 - up1, 2);
        double beta1_pos = (13.0 / 12.0) * std::pow(um2 - 2.0 * um1 + u0, 2) + 0.25 * std::pow(um2 - 4.0 * um1 + 3.0 * u0, 2);

        // Linear weights
        double d0_pos = 2.0 / 3.0;
        double d1_pos = 1.0 / 3.0;

        // Nonlinear weights
        double alpha0_pos = d0_pos / std::pow(eps + beta0_pos, 2);
        double alpha1_pos = d1_pos / std::pow(eps + beta1_pos, 2);
        double sum_alpha_pos = alpha0_pos + alpha1_pos;

        double w0_pos = alpha0_pos / sum_alpha_pos;
        double w1_pos = alpha1_pos / sum_alpha_pos;

        flux_pos = w0_pos * q0_pos + w1_pos * q1_pos;

        // WENO3 reconstruction for negative flux (upwind from right)
        // Stencils: S0 = {i, i+1}, S1 = {i+1, i+2}
        double q0_neg = (1.0 / 3.0) * u0 + (5.0 / 6.0) * up1 - (1.0 / 6.0) * up2; // ENO2 on S0
        double q1_neg = (-1.0 / 6.0) * um1 + (5.0 / 6.0) * u0 + (1.0 / 3.0) * up1; // ENO2 on S1

        // Smoothness indicators
        double beta0_neg = (13.0 / 12.0) * std::pow(u0 - 2.0 * up1 + up2, 2) + 0.25 * std::pow(u0 - up2, 2);
        double beta1_neg = (13.0 / 12.0) * std::pow(um1 - 2.0 * u0 + up1, 2) + 0.25 * std::pow(3.0 * um1 - 4.0 * u0 + up1, 2);

        // Linear weights
        double d0_neg = 2.0 / 3.0;
        double d1_neg = 1.0 / 3.0;

        // Nonlinear weights
        double alpha0_neg = d0_neg / std::pow(eps + beta0_neg, 2);
        double alpha1_neg = d1_neg / std::pow(eps + beta1_neg, 2);
        double sum_alpha_neg = alpha0_neg + alpha1_neg;

        double w0_neg = alpha0_neg / sum_alpha_neg;
        double w1_neg = alpha1_neg / sum_alpha_neg;

        flux_neg = w0_neg * q0_neg + w1_neg * q1_neg;
    }
};

#endif