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
    
    virtual Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                                   const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) = 0;
    
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
        int gridSize) const {
       
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

    // Standard interface for TimeScheme
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        // This is a placeholder implementation that will never be called
        // The actual implementation is in the specialized version below
        return phi;
    }
    

    Eigen::VectorXd advance(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, const Eigen::VectorXd& phi) {
        
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


class implicitDualStep : public TimeScheme {
public:
    implicitDualStep(double timeStep, double GRID_SPACING = 1.0, 
                     double pseudoTimeStep = 1, double dampingFactor = 0.5, 
                     size_t maxInnerIterations = 500, double convergenceTol = 1e-8) 
        : TimeScheme(timeStep, GRID_SPACING), tau(pseudoTimeStep), gamma(dampingFactor), 
          dualTimeStepping(maxInnerIterations), convergenceTolerance(convergenceTol) {}

    Eigen::SparseMatrix<double> GenMatrixA(const Eigen::VectorXd& phi,
                                           const Eigen::VectorXd& Ux,
                                           const Eigen::VectorXd& Uy,
                                           const Eigen::VectorXd& Uz,
                                           double spacing,
                                           int gridSize) const {
        const int n = phi.size();
        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(10 * n); // Increased reserve for 2nd-order terms

        const int num_threads = omp_get_max_threads();
        std::vector<std::vector<T>> thread_triplets(num_threads);

        const double epsilon = 1e-6;

        #pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
            thread_triplets[thread_id].reserve(10 * n / num_threads);

            #pragma omp for nowait
            for (int idx = 0; idx < n; idx++) {
                int x = idx % gridSize;
                int y = (idx / gridSize) % gridSize;
                int z = idx / (gridSize * gridSize);

                bool isBoundary = (x <= 1 || x >= gridSize - 2 ||
                                   y <= 1 || y >= gridSize - 2 ||
                                   z <= 1 || z >= gridSize - 2);

                if (isBoundary) {
                    thread_triplets[thread_id].emplace_back(idx, idx, 1);
                    continue;
                }

                double diagTerm = 1.0 / tau + (1.0 + gamma) / dt;

                // --- X-direction (2nd-order upwind) ---
                if (x > 1 && x < gridSize - 2) {
                    double ux_avg = Ux(idx);
                    if (ux_avg >= 0) {
                        int idxL1 = idx - 1;  // i-1
                        int idxL2 = idx - 2;  // i-2
                        double coef_i = ux_avg * 3.0 / (2.0 * spacing);
                        double coef_im1 = -ux_avg * 4.0 / (2.0 * spacing);
                        double coef_im2 = ux_avg * 1.0 / (2.0 * spacing);
                        thread_triplets[thread_id].emplace_back(idx, idxL2, coef_im2);
                        thread_triplets[thread_id].emplace_back(idx, idxL1, coef_im1);
                        diagTerm += coef_i;
                    } else {
                        int idxR1 = idx + 1;  // i+1
                        int idxR2 = idx + 2;  // i+2
                        double coef_i = -ux_avg * 3.0 / (2.0 * spacing);
                        double coef_ip1 = ux_avg * 4.0 / (2.0 * spacing);
                        double coef_ip2 = -ux_avg * 1.0 / (2.0 * spacing);
                        thread_triplets[thread_id].emplace_back(idx, idxR1, coef_ip1);
                        thread_triplets[thread_id].emplace_back(idx, idxR2, coef_ip2);
                        diagTerm += coef_i;
                    }
                }

                // --- Y-direction (2nd-order upwind) ---
                if (y > 1 && y < gridSize - 2) {
                    double uy_avg = Uy(idx);
                    if (uy_avg >= 0) {
                        int idxB1 = idx - gridSize;      // j-1
                        int idxB2 = idx - 2 * gridSize;  // j-2
                        double coef_j = uy_avg * 3.0 / (2.0 * spacing);
                        double coef_jm1 = -uy_avg * 4.0 / (2.0 * spacing);
                        double coef_jm2 = uy_avg * 1.0 / (2.0 * spacing);
                        thread_triplets[thread_id].emplace_back(idx, idxB2, coef_jm2);
                        thread_triplets[thread_id].emplace_back(idx, idxB1, coef_jm1);
                        diagTerm += coef_j;
                    } else {
                        int idxT1 = idx + gridSize;      // j+1
                        int idxT2 = idx + 2 * gridSize;  // j+2
                        double coef_j = -uy_avg * 3.0 / (2.0 * spacing);
                        double coef_jp1 = uy_avg * 4.0 / (2.0 * spacing);
                        double coef_jp2 = -uy_avg * 1.0 / (2.0 * spacing);
                        thread_triplets[thread_id].emplace_back(idx, idxT1, coef_jp1);
                        thread_triplets[thread_id].emplace_back(idx, idxT2, coef_jp2);
                        diagTerm += coef_j;
                    }
                }

                // --- Z-direction (2nd-order upwind) ---
                if (z > 1 && z < gridSize - 2) {
                    double uz_avg = Uz(idx);
                    if (uz_avg >= 0) {
                        int idxD1 = idx - gridSize * gridSize;      // k-1
                        int idxD2 = idx - 2 * gridSize * gridSize;  // k-2
                        double coef_k = uz_avg * 3.0 / (2.0 * spacing);
                        double coef_km1 = -uz_avg * 4.0 / (2.0 * spacing);
                        double coef_km2 = uz_avg * 1.0 / (2.0 * spacing);
                        thread_triplets[thread_id].emplace_back(idx, idxD2, coef_km2);
                        thread_triplets[thread_id].emplace_back(idx, idxD1, coef_km1);
                        diagTerm += coef_k;
                    } else {
                        int idxU1 = idx + gridSize * gridSize;      // k+1
                        int idxU2 = idx + 2 * gridSize * gridSize;  // k+2
                        double coef_k = -uz_avg * 3.0 / (2.0 * spacing);
                        double coef_kp1 = uz_avg * 4.0 / (2.0 * spacing);
                        double coef_kp2 = -uz_avg * 1.0 / (2.0 * spacing);
                        thread_triplets[thread_id].emplace_back(idx, idxU1, coef_kp1);
                        thread_triplets[thread_id].emplace_back(idx, idxU2, coef_kp2);
                        diagTerm += coef_k;
                    }
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

    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                            const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        return phi;
    }
    
    Eigen::VectorXd advance(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, 
                            const Eigen::VectorXd& phi_n,
                            const Eigen::VectorXd& phi_nm1,
                            const int gridSize
                        ) {
        Eigen::VectorXd phi_m = phi_n;     // Current pseudo-time solution
        Eigen::VectorXd phi_mn = phi_n;    // Previous pseudo-time solution
        
        std::cout << "Starting dual time stepping for physical time step..." << std::endl;
        
        for (size_t t = 0; t < dualTimeStepping; t++) {
            Eigen::VectorXd b = phi_m / tau + (1.0 + gamma) * phi_n / dt + gamma * (phi_n - phi_nm1) / dt;
            #pragma omp parallel for
            for (int idx=0; idx < b.size(); idx++) {
                int x = idx % gridSize;
                int y = (idx / gridSize) % gridSize;
                int z = idx / (gridSize * gridSize);
                bool isBoundary = (x <= 1 || x >= gridSize - 2 ||
                    y <= 1 || y >= gridSize - 2 ||
                    z <= 1 || z >= gridSize - 2);
                if (isBoundary) {
                    b(idx) = phi_n(idx);
                }
            }

            Eigen::VectorXd phi_mp = solveStandard(A, b);
            phi_mn = phi_m;
            phi_m = phi_mp;

            Eigen::VectorXd residual = phi_m - phi_mn;
            double l2_norm = residual.norm();
            double relative_norm = l2_norm / (phi_m.norm() + 1e-12);
            
            if (t % 10 == 0 || t < 5) {
                std::cout << "Pseudo-time iteration " << t << ": L2 norm = " << l2_norm 
                          << ", Relative norm = " << relative_norm << std::endl;
            }
            
            if (l2_norm < convergenceTolerance) {
                std::cout << "Converged at pseudo-time iteration: " << t 
                          << ": L2 norm = " << l2_norm 
                          << ", Relative norm = " << relative_norm << std::endl;
                break;
            }
            
            if (t == dualTimeStepping - 1) {
                std::cout << "Warning: Maximum pseudo-time iterations reached without convergence" << std::endl;
                std::cout << "Final relative norm: " << relative_norm << std::endl;
            }
        }
        
        return phi_m;
    }

    void setPseudoTimeStep(double newTau) { tau = newTau; }
    void setDampingFactor(double newGamma) { gamma = newGamma; }
    void setMaxInnerIterations(size_t newMax) { dualTimeStepping = newMax; }
    void setConvergenceTolerance(double newTol) { convergenceTolerance = newTol; }
    
    double getPseudoTimeStep() const { return tau; }
    double getDampingFactor() const { return gamma; }
    size_t getMaxInnerIterations() const { return dualTimeStepping; }
    double getConvergenceTolerance() const { return convergenceTolerance; }

private:
    double tau;
    double gamma;
    size_t dualTimeStepping;
    double convergenceTolerance;

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
#endif