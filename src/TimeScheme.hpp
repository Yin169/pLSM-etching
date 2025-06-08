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



class implicitLUSGS : public TimeScheme {
public:
    implicitLUSGS(double timeStep, double GRID_SPACING = 1.0, 
                  double pseudoTimeStep = 2, double dampingFactor = 0.5, 
                  size_t maxInnerIterations = 500, double convergenceTol = 1e-10) 
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
    
                bool isBoundary = (x == 0 || x == gridSize - 1 ||
                                   y == 0 || y == gridSize - 1 ||
                                   z == 0 || z == gridSize - 1);
    
                if (isBoundary) {
                    thread_triplets[thread_id].emplace_back(idx, idx, 1.0 / tau + (1.0 + gamma) / dt);
                    continue;
                }
    
                double diagTerm = 1.0 / tau + (1.0 + gamma) / dt;
    
                // --- X-direction ---
                int idxL = idx - 1;
                int idxR = idx + 1;
                if (x > 0 && x < gridSize - 1) {
                    double aL = 0.5 * (Ux(idxL) + Ux(idx));
                    double aR = 0.5 * (Ux(idx) + Ux(idxR));
    
                    double fluxL = 0.5 * (aL + std::abs(aL) + epsilon) / spacing;
                    double fluxR = 0.5 * (aR - std::abs(aR) - epsilon) / spacing;
    
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
    
                    double fluxB = 0.5 * (aB + std::abs(aB) + epsilon) / spacing;
                    double fluxT = 0.5 * (aT - std::abs(aT) - epsilon) / spacing;
    
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
    
                    double fluxD = 0.5 * (aD + std::abs(aD) + epsilon) / spacing;
                    double fluxU = 0.5 * (aU - std::abs(aU) - epsilon) / spacing;
    
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


    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        return phi;
    }
    
    Eigen::VectorXd advance(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, const Eigen::VectorXd& phi_n,
                           const Eigen::VectorXd& phi_nm1) {
        
        Eigen::VectorXd phi_m = phi_n;     // Current pseudo-time solution
        Eigen::VectorXd phi_mn = phi_n;    // Previous pseudo-time solution
        
        std::cout << "Starting dual time stepping for physical time step..." << std::endl;
        
        for (size_t t = 0; t < dualTimeStepping; t++) {
            // Generate RHS for current iteration
            Eigen::VectorXd b = phi_m / tau + (1.0 + gamma) * phi_n / dt + gamma * (phi_n - phi_nm1) / dt;
            
            // Solve linear system
            Eigen::VectorXd phi_mp = solveStandard(A, b);

            // Update solutions
            phi_mn = phi_m;
            phi_m = phi_mp;

            // Check convergence
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
            
            // Prevent infinite loops
            if (t == dualTimeStepping - 1) {
                std::cout << "Warning: Maximum pseudo-time iterations reached without convergence" << std::endl;
                std::cout << "Final relative norm: " << relative_norm << std::endl;
            }
        }
        
        return phi_m;
    }

    // Setter methods for parameters
    void setPseudoTimeStep(double newTau) { tau = newTau; }
    void setDampingFactor(double newGamma) { gamma = newGamma; }
    void setMaxInnerIterations(size_t newMax) { dualTimeStepping = newMax; }
    void setConvergenceTolerance(double newTol) { convergenceTolerance = newTol; }
    
    // Getter methods
    double getPseudoTimeStep() const { return tau; }
    double getDampingFactor() const { return gamma; }
    size_t getMaxInnerIterations() const { return dualTimeStepping; }
    double getConvergenceTolerance() const { return convergenceTolerance; }

private:
    double tau;                    // Pseudo time step
    double gamma;                  // Damping factor for temporal discretization
    size_t dualTimeStepping;      // Maximum number of inner iterations
    double convergenceTolerance;   // Convergence tolerance

    Eigen::VectorXd solveStandard(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, const Eigen::VectorXd& b) {
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