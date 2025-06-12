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

    inline int getIndex(int x, int y, int z, int GRID_SIZE) const {
        static const int GRID_SIZE_SQ = GRID_SIZE * GRID_SIZE;
        return x + y * GRID_SIZE + z * GRID_SIZE_SQ;
    }
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

class TVDRK3WENO5Scheme : public TimeScheme {
public:
    TVDRK3WENO5Scheme(double timeStep, double gridSpacing = 1.0)
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

        Eigen::VectorXd L1 = computeRHS(phi, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi1 = phi + dt * L1;

        Eigen::VectorXd L2 = computeRHS(phi1, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi2 = 0.75 * phi + 0.25 * phi1 + 0.25 * dt * L2;

        Eigen::VectorXd L3 = computeRHS(phi2, Ux, Uy, Uz, gridSize);
        Eigen::VectorXd phi_next = (1.0 / 3.0) * phi + (2.0 / 3.0) * phi2 + (2.0 / 3.0) * dt * L3;

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

            if (x < 3 || x >= N - 3 || y < 3 || y >= N - 3 || z < 3 || z >= N - 3) {
                rhs(idx) = 0.0;
                continue;
            }

            double flux_x_plus = WENO5Flux(phi, Ux, idx, N, 0);
            double flux_x_minus = WENO5Flux(phi, Ux, idx - 1, N, 0);

            double flux_y_plus = WENO5Flux(phi, Uy, idx, N, 1);
            double flux_y_minus = WENO5Flux(phi, Uy, idx - N, N, 1);

            double flux_z_plus = WENO5Flux(phi, Uz, idx, N, 2);
            double flux_z_minus = WENO5Flux(phi, Uz, idx - N * N, N, 2);

            rhs(idx) = -(flux_x_plus - flux_x_minus) / dx
                     - (flux_y_plus - flux_y_minus) / dx
                     - (flux_z_plus - flux_z_minus) / dx + 0.1 * computeMeanCurvature(idx, phi, N);
        }
        return rhs;
    }

    double WENO5Flux(const Eigen::VectorXd& phi,
                     const Eigen::VectorXd& velocity,
                     int idx, int N, int dir) {

        int stride = (dir == 0) ? 1 : (dir == 1) ? N : N * N;

        // Check bounds for 5-point stencil
        if (idx - 2 * stride < 0 || idx + 3 * stride >= phi.size()) {
            return 0.0;
        }

        double v_face = 0.5 * (velocity(idx) + velocity(idx + stride));

        // Reconstruct left and right states at interface i+1/2
        double phi_l = WENO5Reconstruct(
            phi(idx - 2 * stride), phi(idx - stride), phi(idx),
            phi(idx + stride), phi(idx + 2 * stride), 1);  // Left state (f_{i+1/2}^-)

        double phi_r = WENO5Reconstruct(
            phi(idx + 3 * stride), phi(idx + 2 * stride), phi(idx + stride),
            phi(idx), phi(idx - stride), -1);  // Right state (f_{i+1/2}^+)

        return 0.5 * (v_face * (phi_l + phi_r) - std::abs(v_face) * (phi_r - phi_l));
        // return v_face > 0.0 ? v_face * phi_l : v_face * phi_r;
    }

    double WENO5Reconstruct(double f_m2, double f_m1, double f_0,
                            double f_p1, double f_p2, int side) {
        // Standard optimal weights for WENO5
        double d[3] = {0.1 , 0.6, 0.3};
        double IS[3], alpha[3], w[3], q[3];

        if (side > 0) {
            // Reconstruct f_{i+1/2}^- (left state at interface)
            q[0] = (2.0 * f_m2 - 7.0 * f_m1 + 11.0 * f_0) / 6.0;
            q[1] = (-f_m1 + 5.0 * f_0 + 2.0 * f_p1) / 6.0;
            q[2] = (2.0 * f_0 + 5.0 * f_p1 - f_p2) / 6.0;

            // Smoothness indicators (same for both sides)
            IS[0] = 13.0 / 12.0 * std::pow(f_0 - 2.0 * f_m1 + f_m2, 2)
                    + 1.0 / 4.0 * std::pow(f_m2 - 4.0 * f_m1 + 3.0 * f_0, 2);
            IS[1] = 13.0 / 12.0 * std::pow(f_m1 - 2.0 * f_0 + f_p1, 2)
                    + 1.0 / 4.0 * std::pow(f_p1 - f_m1, 2);
            IS[2] = 13.0 / 12.0 * std::pow(f_0 - 2.0 * f_p1 + f_p2, 2)
                    + 1.0 / 4.0 * std::pow(3.0 * f_0 - 4.0 * f_p1 + f_p2, 2);
            
        } else {
            // Reconstruct f_{i+1/2}^+ (right state at interface)
            q[0] = (-f_p2 + 5.0 * f_p1 + 2.0 * f_0) / 6.0;
            q[1] = (2.0 * f_p1 + 5.0 * f_0 - f_m1) / 6.0;
            q[2] = (11.0 * f_0 - 7.0 * f_m1 + 2.0 * f_m2) / 6.0;

        // Smoothness indicators (same for both sides)
            IS[0] = 13.0 / 12.0 * std::pow(f_p2 - 2.0 * f_p1 + f_0, 2)
                    + 1.0 / 4.0 * std::pow(3.0 * f_p2 - 4.0 * f_p1 + f_0, 2);
            IS[1] = 13.0 / 12.0 * std::pow(f_p1 - 2.0 * f_0 + f_m1, 2)
                    + 1.0 / 4.0 * std::pow(f_p1 - f_m1, 2);
            IS[2] = 13.0 / 12.0 * std::pow(f_0 - 2.0 * f_m1 + f_m2, 2)
                    + 1.0 / 4.0 * std::pow(3.0 * f_0 - 4.0 * f_m1 + f_m2, 2);
        }

        // Calculate nonlinear weights
        for (int i = 0; i < 3; ++i) {
            alpha[i] = d[i] / std::pow(eps + IS[i], 2);
        }

        double alpha_sum = alpha[0] + alpha[1] + alpha[2];
        for (int i = 0; i < 3; ++i) {
            w[i] = alpha[i] / alpha_sum;
        }

        return w[0] * q[0] + w[1] * q[1] + w[2] * q[2];
    }

    double computeMeanCurvature(int idx, const Eigen::VectorXd& phi, int GRID_SIZE) {
        const int x = idx % GRID_SIZE;
        const int y = (idx / GRID_SIZE) % GRID_SIZE;
        const int z = idx / (GRID_SIZE * GRID_SIZE);
        
        // Check if we're too close to the boundary for accurate curvature calculation
        
        if (x < 3 || x >= GRID_SIZE - 3 || y < 3 || y >= GRID_SIZE - 3 || z < 3 || z >= GRID_SIZE - 3) {
            return 0.0;
        }
        
        // Get indices for central differences
        const int idx_x_plus = getIndex(x+1, y, z, GRID_SIZE);
        const int idx_x_minus = getIndex(x-1, y, z, GRID_SIZE);
        const int idx_y_plus = getIndex(x, y+1, z, GRID_SIZE);
        const int idx_y_minus = getIndex(x, y-1, z, GRID_SIZE);
        const int idx_z_plus = getIndex(x, y, z+1, GRID_SIZE);
        const int idx_z_minus = getIndex(x, y, z-1, GRID_SIZE);
        
        // Mixed derivatives indices
        const int idx_xy_plus = getIndex(x+1, y+1, z, GRID_SIZE);
        const int idx_xy_minus = getIndex(x-1, y-1, z, GRID_SIZE);
        const int idx_xz_plus = getIndex(x+1, y, z+1, GRID_SIZE);
        const int idx_xz_minus = getIndex(x-1, y, z-1, GRID_SIZE);
        const int idx_yz_plus = getIndex(x, y+1, z+1, GRID_SIZE);
        const int idx_yz_minus = getIndex(x, y-1, z-1, GRID_SIZE);
        
        // First derivatives (central differences)
        const double inv_spacing = 1.0 / (2.0 * dx);
        const double phi_x = (phi[idx_x_plus] - phi[idx_x_minus]) * inv_spacing;
        const double phi_y = (phi[idx_y_plus] - phi[idx_y_minus]) * inv_spacing;
        const double phi_z = (phi[idx_z_plus] - phi[idx_z_minus]) * inv_spacing;
        
        // Calculate gradient magnitude with small epsilon to avoid division by zero
        const double epsilon = 1e-10;
        const double grad_phi_squared = phi_x*phi_x + phi_y*phi_y + phi_z*phi_z + epsilon;
        const double grad_phi_magnitude = std::sqrt(grad_phi_squared);
        
        // If gradient is too small, curvature is not well-defined
        if (grad_phi_magnitude < 1e-6) {
            return 0.0;
        }
        
        // Second derivatives (central differences)
        const double inv_spacing_squared = 1.0 / (dx * dx);
        const double phi_xx = (phi[idx_x_plus] - 2.0 * phi[idx] + phi[idx_x_minus]) * inv_spacing_squared;
        const double phi_yy = (phi[idx_y_plus] - 2.0 * phi[idx] + phi[idx_y_minus]) * inv_spacing_squared;
        const double phi_zz = (phi[idx_z_plus] - 2.0 * phi[idx] + phi[idx_z_minus]) * inv_spacing_squared;
        
        // Mixed derivatives (central differences) with more stable calculation
        const double phi_xy = (phi[idx_xy_plus] - phi[idx_x_plus] - phi[idx_y_plus] + phi[idx] +
                              phi[idx] - phi[idx_x_minus] - phi[idx_y_minus] + phi[idx_xy_minus]) * 
                              (0.25 * inv_spacing_squared);
        
        const double phi_xz = (phi[idx_xz_plus] - phi[idx_x_plus] - phi[idx_z_plus] + phi[idx] +
                              phi[idx] - phi[idx_x_minus] - phi[idx_z_minus] + phi[idx_xz_minus]) * 
                              (0.25 * inv_spacing_squared);
        
        const double phi_yz = (phi[idx_yz_plus] - phi[idx_y_plus] - phi[idx_z_plus] + phi[idx] +
                              phi[idx] - phi[idx_y_minus] - phi[idx_z_minus] + phi[idx_yz_minus]) * 
                              (0.25 * inv_spacing_squared);
        
        // Compute mean curvature using the formula:
        // κ = div(∇φ/|∇φ|) = (φxx(φy²+φz²) + φyy(φx²+φz²) + φzz(φx²+φy²) - 2φxyφxφy - 2φxzφxφz - 2φyzφyφz) / |∇φ|³
        const double numerator = phi_xx * (phi_y*phi_y + phi_z*phi_z) +
                                phi_yy * (phi_x*phi_x + phi_z*phi_z) +
                                phi_zz * (phi_x*phi_x + phi_y*phi_y) -
                                2.0 * (phi_xy * phi_x * phi_y +
                                      phi_xz * phi_x * phi_z +
                                      phi_yz * phi_y * phi_z);
        
        // Use grad_phi_magnitude^3 with epsilon to avoid division by very small numbers
        const double grad_phi_cubed = grad_phi_magnitude * grad_phi_squared;
        
        // Limit the curvature to avoid extreme values that can cause instability
        double curvature = numerator / grad_phi_cubed;
        
        return curvature;
    }
    
};

#endif