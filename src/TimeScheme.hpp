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

class TVDRK3WENO3Scheme : public TimeScheme {
public:
    TVDRK3WENO3Scheme(double timeStep, double gridSpacing = 1.0)
        : TimeScheme(timeStep, gridSpacing), eps(1e-6) {}
    
    Eigen::SparseMatrix<double> GenMatrixA(
        const Eigen::VectorXd&, const Eigen::VectorXd&,
        const Eigen::VectorXd&, const Eigen::VectorXd&,
        double, int) override {
        return Eigen::SparseMatrix<double>(0, 0); // Not used for explicit scheme
    }
    
    Eigen::VectorXd advance(
        const Eigen::SparseMatrix<double, Eigen::RowMajor>&,
        const Eigen::VectorXd& phi,
        const Eigen::VectorXd& Ux,
        const Eigen::VectorXd& Uy,
        const Eigen::VectorXd& Uz,
        int gridSize) override {
        
        // Stage 1: u^(1) = u^n + dt * L(u^n)
        Eigen::VectorXd L1 = computeRHS(phi, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi1 = phi + dt * L1;
        
        // Stage 2: u^(2) = 3/4 * u^n + 1/4 * u^(1) + 1/4 * dt * L(u^(1))
        Eigen::VectorXd L2 = computeRHS(phi1, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi2 = 0.75 * phi + 0.25 * phi1 + 0.25 * dt * L2;
        
        // Stage 3: u^(n+1) = 1/3 * u^n + 2/3 * u^(2) + 2/3 * dt * L(u^(2))
        Eigen::VectorXd L3 = computeRHS(phi2, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi_next = (1.0/3.0) * phi + (2.0/3.0) * phi2 + (2.0/3.0) * dt * L3;
        
        return phi_next;
    }

private:
    const double eps;
    
    Eigen::VectorXd computeRHS(const Eigen::VectorXd& phi,
                              const Eigen::VectorXd& Ux,
                              const Eigen::VectorXd& Uy,
                              const Eigen::VectorXd& Uz,
                              int N) {
        int n = phi.size();
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n);
        
        #pragma omp parallel for
        for (int idx = 0; idx < n; ++idx) {
            int x = idx % N;
            int y = (idx / N) % N;
            int z = idx / (N * N);
            
            // Apply boundary conditions (zero gradient or other appropriate BC)
            if (x < 2 || x >= N - 2 || y < 2 || y >= N - 2 || z < 2 || z >= N - 2) {
                rhs(idx) = 0.0;
                continue;
            }
            
            // Compute flux differences in each direction
            // dF/dx term
            double flux_x_plus = WENO3Flux(phi, Ux, idx, N, 0);
            double flux_x_minus = WENO3Flux(phi, Ux, idx-1, N, 0);
            
            // dG/dy term  
            double flux_y_plus = WENO3Flux(phi, Uy, idx, N, 1);
            double flux_y_minus = WENO3Flux(phi, Uy, idx - N, N, 1);
            
            // dH/dz term
            double flux_z_plus = WENO3Flux(phi, Uz, idx , N, 2);
            double flux_z_minus = WENO3Flux(phi, Uz, idx - N*N, N, 2);
            
            // Conservative form: -d/dx(u*phi) - d/dy(v*phi) - d/dz(w*phi)
            rhs(idx) = -(flux_x_plus - flux_x_minus) / dx
                      -(flux_y_plus - flux_y_minus) / dx  
                      -(flux_z_plus - flux_z_minus) / dx;
        }
        return rhs;
    }
    
    // Compute numerical flux at interface using WENO3
    double WENO3Flux(const Eigen::VectorXd& phi,
                     const Eigen::VectorXd& velocity,
                     int idx, int N, int dir) {
        
        int stride = (dir == 0) ? 1 : (dir == 1) ? N : N * N;
        
        // Check bounds for WENO3 stencil
        int test_idx = idx + stride;
        if (test_idx - 2*stride < 0 || test_idx + 2*stride >= phi.size()) {
            return 0.0;
        }
        
        double v_face = 0.5 * (velocity(idx) + velocity(idx + stride));

        // Reconstruct left and right states
        double phi_l = WENO3Reconstruct(
                phi(idx - stride),
                phi(idx),
                phi(idx + stride),
                -1
            );
        
        double phi_r = WENO3Reconstruct(
                phi(idx),
                phi(idx + stride),
                phi(idx + 2*stride),
                1
            );
        
        return 0.5*(v_face*(phi_l + phi_r) - std::abs(v_face)*(phi_r - phi_l));
    }
    
    double WENO3Reconstruct(double f_m1, double f_0, double f_p1, int side) {
        // Candidate reconstructions
        double u0, u1;
        if (side <= 0) {
            u0 = (f_0 + f_p1) / 2.0;        // Stencil {f_0, f_p1}
            u1 = (3.0 * f_0 - f_m1) / 2.0;  // Stencil {f_m1, f_0}
        } else {
            u0 = (3.0 * f_0 - f_p1) / 2.0;  // Stencil {f_0, f_p1}
            u1 = (f_m1 + f_0) / 2.0;        // Stencil {f_m1, f_0}
        }
        
        // Smoothness indicators
        double beta0 = (f_p1 - f_0) * (f_p1 - f_0);
        double beta1 = (f_0 - f_m1) * (f_0 - f_m1);
        
        double d0 = 2.0 / 3.0;
        double d1 = 1.0 / 3.0;
        
        // Nonlinear weights
        const double eps = 1e-6;
        double alpha0 = d0 / ((eps + beta0) * (eps + beta0));
        double alpha1 = d1 / ((eps + beta1) * (eps + beta1));
        double sum_alpha = alpha0 + alpha1;
        double w0 = alpha0 / sum_alpha;
        double w1 = alpha1 / sum_alpha;
        
        // Final reconstructed value
        return w0 * u0 + w1 * u1;
    }
};

#endif