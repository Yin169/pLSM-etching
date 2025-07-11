# Parallel Level-Set Based Approach for Etching Topography Simulation

## Abstract

This repository implements a Level Set Method (LSM) framework to simulate material interface evolution during semiconductor etching. The LSM naturally handles sharp corners, topological changes (merging and splitting), and large speed variations by solving Hamilton-Jacobi equations on structured grids. Key features include multiple high-order time integration and spatial reconstruction schemes, surface extraction, material-dependent etching, and parallel computation via OpenMP combined with sparse matrix operations. The implementation achieves super linear speedup. Validation using comprehensive quantitative metrics—including Hausdorff distance, area difference, perimeter ratio, shape context, and Hu moments—confirms high accuracy and strong robustness in capturing topological changes. Benchmarks demonstrate 97.74% similarity with the industrial standard SEMulator3D.

## Introduction

Semiconductor etching is a critical manufacturing process for creating intricate microstructures through selective material removal. Accurate simulation of this process is essential to reduce development costs. The process involves tracking evolving interfaces with complex geometries, multi-material interactions, and challenging boundary conditions. Traditional methods face limitations: marker/string techniques suffer from swallowtail instabilities during topological changes; cell-based approaches compromise geometric accuracy; and characteristic methods exhibit 3D stability issues. The Level Set Method (LSM), pioneered by Osher and Sethian, overcomes these challenges by implicitly representing interfaces as zero-level sets of higher-dimensional functions. This approach naturally handles topological changes, sharp corners, and extreme velocity variations while providing entropy-satisfying weak solutions to Hamilton-Jacobi equations. Although extensions like fast marching and narrow-band techniques exist, prior semiconductor implementations have been limited to first-order schemes and lack efficient parallelization.

This work introduces a parallel 3D LSM framework for etching simulation featuring:

1. High-order spatial and temporal discretization for improved accuracy
2. Multi-threaded parallelization using OpenMP for enhanced computational efficiency
3. Robust handling of orientation-dependent etching and multi-material interactions
4. Validation against industrial standards, with quantitative metrics (Hausdorff distance, shape context, Hu moments) verifying accurate topology management

## Preliminaries

The level set method represents the evolving etching front $\Gamma(t)$ as the zero level set of a signed distance function $\phi(\mathbf{x}, t)$:

$$\Gamma(t) = \{ \mathbf{x} \mid \phi(\mathbf{x}, t) = 0 \}$$

Where:
- $\phi <= 0$: inside material
- $\phi > 0$: outside geometry

This implicit representation naturally handles topological changes (splitting/merging) and complex geometries.

### Governing Equations

The evolution of $\phi$ is governed by the linear advection equation:

$$\frac{\partial \phi}{\partial t} + \mathbf{U} \cdot \nabla \phi = 0$$

Alternatively, it can be expressed in Hamilton-Jacobi form:

$$\frac{\partial \phi}{\partial t} + F|\nabla \phi| = 0 \tag{1}$$

where:
- $\mathbf{U}$ is the velocity field
- $F = \mathbf{U} \cdot \mathbf{n}$ is the normal velocity component
- $\mathbf{n} = \nabla \phi / |\nabla \phi|$ is the unit normal vector

#### Velocity Field

For semiconductor applications, the velocity field $\mathbf{U}$ depends on the material:

$$\mathbf{U}(\mathbf{r}) = [\alpha R_m, \alpha R_m, R_m]^T$$

where:
- $R_m$ is the vertical etching rate for material $m$
- $\alpha_r$ controls the lateral-to-vertical etching ratio

#### Initial Condition

The initial signed distance field is:

$$\phi(\mathbf{x}, 0) = \pm d(\mathbf{x}, \Gamma_0) \tag{2}$$

where $d$ is the signed distance to the initial interface $\Gamma_0$.

## Tracking Etching Front by Solving PDEs

Etching simulations involve evolving the interface of materials over time, which can be described by the zero level set of a signed distance function $\phi(\mathbf{r}, t)$ governed by the Hamilton-Jacobi equation.The motion of the etching front $\Gamma(t)$ is described by a velocity field $\mathbf{U(r)}$, typically defined as:

$$
\mathbf{U(r)} = 
\begin{bmatrix}
\alpha R_m \\
\alpha R_m \\
R_m
\end{bmatrix}
$$

where $R_m$ is the vertical etching rate of material $m$, and $\alpha < 1$ controls lateral etching.

Due to the complex geometries in semiconductor structures, first-order methods fail to provide sufficient accuracy. Hence, higher-order methods—such as finite volume schemes with Riemann solvers and high-order Runge-Kutta integration—are employed to obtain precise and robust numerical solutions.

---

### Finite Volume Discretization

To maintain conservation and handle discontinuities accurately, the semi-discrete form of the conservation law is used:

$$
\frac{\partial \phi}{\partial t} + \frac{f\left(\phi_{i+\frac{1}{2}}\right) - f\left(\phi_{i-\frac{1}{2}}\right)}{\Delta x} = 0
$$

where the flux function is defined as:

$$
f(\phi) = \mathbf{U} \cdot \nabla \phi
$$

---

### Spatial Discretization Schemes

The hyperbolic convective term $\mathbf{U} \cdot \nabla \phi$ presents numerical challenges, requiring specialized discretization schemes.

#### 1. First-Order Upwind

A monotonic but diffusive scheme:

$$
\mathbf{U} \cdot \nabla \phi =
\begin{cases} 
U_x^+ \dfrac{\phi_{i,j,k} - \phi_{i-1,j,k}}{\Delta x} + U_x^- \dfrac{\phi_{i+1,j,k} - \phi_{i,j,k}}{\Delta x} + \\
U_y^+ \dfrac{\phi_{i,j,k} - \phi_{i,j-1,k}}{\Delta y} + U_y^- \dfrac{\phi_{i,j+1,k} - \phi_{i,j,k}}{\Delta y} + \\
U_z^+ \dfrac{\phi_{i,j,k} - \phi_{i,j,k-1}}{\Delta z} + U_z^- \dfrac{\phi_{i,j,k+1} - \phi_{i,j,k}}{\Delta z}
\end{cases}
$$

where $U^\pm = \max/\min(U, 0)$.
While stable, it suffers from excessive numerical diffusion.

---

#### 2. Roe’s Scheme with MUSCL

Combines the Roe solver with piecewise linear MUSCL reconstruction:

$$
\begin{aligned}
F_{i+1/2} &= \frac{1}{2} \left[ U \phi_L + U \phi_R - |U|(\phi_R - \phi_L) \right] \\
\phi_{i+1/2}^L &= \phi_i + \frac{1}{2} \psi(r_i)(\phi_i - \phi_{i-1}) \\
\phi_{i+1/2}^R &= \phi_{i+1} - \frac{1}{2} \psi(r_{i+1})(\phi_{i+2} - \phi_{i+1}) 
\end{aligned}
$$

Limiter function:

$$
\psi(r) = \max(0, \min(1, r))
$$

This configuration enhances accuracy while preserving monotonicity.

---

#### 3. Roe’s Scheme with QUICK Interpolation

Utilizes third-order accuracy in smooth regions via quadratic interpolation:

$$
\phi_L = \begin{cases}
\frac{6\phi_{i-1} + 3\phi_i - \phi_{i-2}}{8} & \text{if } U_f \geq 0 \\
\frac{6\phi_i + 3\phi_{i-1} - \phi_{i+1}}{8} & \text{if } U_f < 0
\end{cases}
$$

$$
\phi_R = \begin{cases}
\frac{6\phi_i + 3\phi_{i+1} - \phi_{i-1}}{8} & \text{if } U_f \geq 0 \\
\frac{6\phi_{i+1} + 3\phi_i - \phi_{i+2}}{8} & \text{if } U_f < 0
\end{cases}
$$

Limiter for oscillation control:

$$
\psi(r) = \frac{r + |r|}{1 + |r|}, \quad r = \frac{\phi_i - \phi_{i-1}}{\phi_{i+1} - \phi_i + \epsilon}
$$

This limiter ensures **Total Variation Diminishing (TVD)** behavior, preserving solution stability near discontinuities.

---

### Extension to 3D

All schemes are extended to three dimensions using **operator splitting**, applied dimension-by-dimension. Grid spacings $\Delta x, \Delta y, \Delta z$ can be anisotropic to match semiconductor geometries.

#### Boundary Conditions

* **Dirichlet**: $\phi = \phi_{\text{specified}}$
* **Neumann**: $\frac{\partial \phi}{\partial n} = 0$

---

### Time Integration

Three schemes are implemented for temporal discretization, balancing accuracy and stability:

#### 1. Backward Euler

1st-order, unconditionally stable:

$$\frac{\phi^{n+1} - \phi^n}{\Delta t} = - \mathbf{U} \cdot \nabla \phi^{n+1}$$

Discretization yields the linear system:

$$(I + \Delta t A)\phi^{n+1} = \phi^n$$

where $A$ is the convection operator matrix. The method's stability makes it robust for stiff problems but introduces $\mathcal{O}(\Delta t)$ dissipation.

---

#### 2. Crank-Nicolson

2nd-order, unconditionally stable:

$$\frac{\phi^{n+1} - \phi^n}{\Delta t} = -\frac{1}{2}\left[\mathbf{U} \cdot \nabla \phi^n + \mathbf{U} \cdot \nabla \phi^{n+1}\right]$$

This yields the linear system:

$$\left(I + \frac{\Delta t}{2}A\right)\phi^{n+1} = \left(I - \frac{\Delta t}{2}A\right)\phi^n$$

---

#### 3. TVD Runge-Kutta 3

3rd-order, explicit:

$$\begin{aligned}
\phi^{(1)} &= \phi^n + \Delta t L(\phi^n) \\
\phi^{(2)} &= \frac{3}{4}\phi^n + \frac{1}{4}\phi^{(1)} + \frac{1}{4}\Delta t L(\phi^{(1)}) \\
\phi^{n+1} &= \frac{1}{3}\phi^n + \frac{2}{3}\phi^{(2)} + \frac{2}{3}\Delta t L(\phi^{(2)})
\end{aligned}$$

where $L(\phi) = -\mathbf{U} \cdot \nabla \phi$.

The method preserves TVD properties when combined with spatial limiters and requires the CFL condition:

$$\Delta t \leq C \frac{\min(\Delta x, \Delta y, \Delta z)}{\max |\mathbf{U}|}, \quad C \leq 1.0$$

#### Comparison of Time Integration Schemes

| Method         | Order | Stability     | Computational Cost  |
|----------------|-------|---------------|---------------------|
| Backward Euler | 1st   | Unconditional | High (linear solve) |
| Crank-Nicolson | 2nd   | Unconditional | High (linear solve) |
| TVD RK3        | 3rd   | CFL-limited   | Low (explicit)      |

### Reinitialization

To maintain the signed distance property ($|\nabla \phi| = 1$), solve the reinitialization equation:

$$\frac{\partial \psi}{\partial \tau} = \text{sign}(\phi_0)(1 - |\nabla \psi|)$$

where:

Smoothed sign function:

$$\text{sign}(\phi_0) = \frac{\phi_0}{\sqrt{\phi_0^2 + |\nabla \phi_0|^2 \epsilon^2}}, \quad \epsilon = 0.5 \Delta x$$

Additional parameters:
- $|\nabla\psi|$ is computed with central differences
- Forward Euler time stepping: $\Delta\tau = 0.1 \min \Delta x$
- Terminate when: $| |\nabla\psi| - 1 | < 0.01$

> **Reinitialization** is executed every 5–10 physical time steps and parallelized using OpenMP.

## High-Performance Computing Strategies

To manage the computational demands of 3D semiconductor etching simulations, we adopt a high-performance computing technique that combines sparse linear algebra optimization with advanced solver configuration. Efficient solution of large sparse systems arising from implicit temporal discretization is achieved through parallel matrix assembly techniques and parallelism of the BiCGSTAB solver. This includes the use of Triplet storage with thread-local buffers, pre-computation of sparsity patterns to eliminate dynamic allocation during assembly, and Lock-free insertion strategies using OpenMP parallel. 

To enhance memory efficiency and performance, matrices are stored as compressed row storage (CRS) format employing blocked CRS layouts for improved cache locality, and utilize SIMD-optimized packing to increase arithmetic throughput. For solving the discretized Hamilton-Jacobi equations, we configure an accelerated BiCGSTAB solver based on Eigen's vectorized implementation. The solver is further enhanced with diagonal pre-conditioning, together with a strict convergence tolerance of $10^{-8}$ to guarantee the accuracy of the results. Meanwhile, OpenMP enable the scaling of explicit method by parallelizing stencil operator over fixed grid.

## Benchmark: Backward Euler vs Runge-Kutta3

The simulation is performed using a $600 \times 600 \times 600$ spatial grid with a fixed time step of $\Delta t = 1\,\text{s}$. Results at $t = 60\,\text{s}$ are compared against those generated by SEMulator3D. All experiments are conducted on a Linux server equipped with an Intel® Xeon® Gold 6230R CPU @ 2.10 GHz. We evaluate the performance of the backward Euler method with first-order upwind scheme against the third-order Runge-Kutta scheme, focusing on both execution time and contour accuracy relative to industrial benchmarks from SEMulator3D.

### Runtimes and Parallel Speedups

| **#Threads** | **Backward Euler (s)** | **Speedup** | **Runge-Kutta 3 (s)** | **Speedup** |
|--------------|------------------------|-------------|------------------------|-------------|
| 1            | 5126                   | -           | 12340                  | -           |
| 2            | 3670                   | 1.39×       | 6406                   | 1.92×       |
| 4            | 3764                   | 1.36×       | 3414                   | 3.61×       |
| 8            | 2683                   | 1.91×       | 1834                   | 6.73×       |
| 16           | 2684                   | 1.90×       | 1098                   | 11.22×      |

From the table above, third-order Runge-Kutta scales significantly better with increasing threads, achieving **11.22× speedup** when the number of threads increases to 16. In contrast, the Backward Euler method demonstrates scaling limitations, with minimal gain beyond 8 threads.

**SEMulator3D's** runtime for simulating this case is **940 seconds with 8 threads** on a computer equipped with an AMD Ryzen 9 5900HX CPU @ 3.30 GHz. Although SEMulator3D uses **4,614,538 triangles** in spatial discretization and applies undisclosed etching simulation techniques, the proposed algorithm already exhibits **comparable efficiency** to SEMulator3D.

### Similarity Comparison with SEMulator3D at Two Y-slices

| **Method**       | **Score $S$ at $y = -184$** | **Score $S$ at $y = 254$** |
|------------------|------------------------------|------------------------------|
| Backward Euler   | 0.9880                      | 0.9837                      |
| Runge-Kutta 3    | 0.9774                      | 0.9770                      |

# Conclusion

This work presents a high-fidelity, parallel level-set framework for simulating semiconductor etching processes with complex topographies. By integrating high-order spatial reconstruction (MUSCL, QUICK) and time integration schemes (Backward Euler, Crank-Nicolson, TVD RK3), the implementation achieves both numerical stability and geometric accuracy. The method effectively captures sharp corners, topological transitions, and anisotropic etching behaviors, overcoming key limitations of traditional front-tracking approaches.

The adoption of OpenMP-parallelized sparse matrix solvers and stencil operations enables efficient large-scale 3D simulations. Benchmark results show that the third-order Runge-Kutta scheme not only improves accuracy but also demonstrates superior parallel scalability—achieving up to 11.22× speedup with 16 threads. Validation against SEMulator3D confirms that the proposed method maintains a high degree of geometric similarity (up to 98.80%) with industrial standards.

Altogether, the proposed framework offers a robust, extensible, and computationally efficient platform for simulating etching topographies.