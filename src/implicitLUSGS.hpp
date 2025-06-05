#include "LevelSetMethod.hpp"

class implicitLUSGS : public TimeScheme {
  public:
    implicitLUSGS(double timeStep, double GRID_SPACING = 1.0, size_t dualTimeStepping = 20) : TimeScheme(timeStep, GRID_SPACING) {};
    ~implicitLUSGS() {} = default;

	Eigen::SparseMatrix<double> GenMatrixA(const Eigen::VectorXd& phi, 
		const Eigen::VectorXd& Ux, 
        const Eigen::VectorXd& Uy, 
        const Eigen::VectorXd& Uz,
        double spacing, double tau, double gamma,
        int gridSize) const {
			
			const int n = phi.size();
            typedef Eigen::Triplet<double> T;
            std::vector<T> tripletList;
            tripletList.reserve(7 * n);
            
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
                        double diagTerm = 1.0 / tau +  (1+gamma) / dt;
    
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

	Eigen::VectorXd GenRHS(const Eigen::VectorXd& phi_m, const Eigen::VectorXd& phi_n, const Eigen::VectorXd& phi_mn,
        double spacing, double tau, double gamma) const {
        	return phi_m / tau + (1+gamma)*phi_n / dt + (gamma)*(phi+n - phi_mn) / dt; 
        }

    // Standard interface for TimeScheme
    Eigen::VectorXd advance(const Eigen::VectorXd& phi, 
                           const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& L) override {
        // This is a placeholder implementation that will never be called
        // The actual implementation is in the specialized version below
        return phi;
    }
    

    Eigen::VectorXd advance(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& phi) {
		Eigen::VectorXd phi_n = phi;
        Eigen::VectorXd phi_m = phi;
        Eigen::VectorXd phi_mn = phi;

		for (size_t t = 0; t < dualTimeStepping; t++) {
        	Eigen::VectorXd b = GenRHS(phi_m, phi_n, phi_mn, spacing, tau, gamma);
        	Eigen::VectorXd phi_mp = solveStandard(A, b);

            phi_mn = phi_m;
            phi_m = phi_mp;

			Eigen::VectorXd residual = phi_m - phi_mn;
            double l2_norm = residual.norm();
			if (l2_norm < 1e-8) {
				std::cout << "Converged at iteration: " << t << std::endl;
				break;
			}

		}
        return phi_m;
    }

	private:
		double tau;
		double gamma;
		size_t dualTimeStepping;

    // Standard solver method
    Eigen::VectorXd solveStandard(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) {
        // Use BiCGSTAB solver which is more robust for non-symmetric matrices
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
        
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
}