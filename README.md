# Parallel Level-Set BAsed Approach for Etching Topography Simulation

## Abstract
Simulating evolving surfaces during semiconductor etching poses significant challenges for front-tracking methods, particularly in handling sharp corners, topological changes (merging and splitting), and large speed variations. The Level Set Method (LSM) effectively addresses these issues by naturally accommodating such complexities through the solution of Hamilton-Jacobi equations, leveraging techniques from hyperbolic conservation laws.This repository contains the implementation of an LSM framework modeling material interface evolution under etching conditions using partial differential equations solved on structured grids. Key features include multiple high-order time integration schemes and spatial reconstruction scheme, surface extraction, material-dependent etching, and parallel computation via OpenMP combined with sparse matrix operations. This implementation achieves near-linear speedup.Validation using comprehensive quantitative metrics—including Hausdorff distance, area difference, perimeter ratio, shape context, and Hu moments—confirms high accuracy and strong robustness in capturing topological changes. Benchmarks demonstrate 97.74% similarity with the industrial standard SEMulator3D.

## Introduction
Semiconductor etching—a critical manufacturing process for creating intricate microstructures through selective material removal—demands precise simulation to reduce development costs. This process involves tracking evolving interfaces with complex geometries, multi-material interactions, and challenging boundary conditions. Traditional methods face fundamental limitations: marker/string techniques suffer from swallowtail instabilities during topological changes; cell-based approaches compromise geometric accuracy; and characteristic methods exhibit 3D stability issues. The Level Set Method (LSM), pioneered by Osher and Sethian, overcomes these challenges by implicitly representing interfaces as zero-level sets of higher-dimensional functions. This approach naturally handles topological changes, sharp corners, and extreme velocity variations while providing entropy-satisfying weak solutions to Hamilton-Jacobi equations. Although extensions like fast marching and narrow-band techniques exist, prior semiconductor implementations have been limited to first-order schemes and lack efficient parallelization.

This work introduces a parallel 3D LSM framework for etching simulation featuring:

1. High-order (3rd) spatial/temporal discretization for high accuracy
2. Multi-threaded parallelization (OpenMP) for large-scale efficiency
3. Robust handling of orientation-dependent etching and multi-material interactions
4. Validation against industrial standards confirms solution robustness, with quantitative metrics (Hausdorff distance, shape context, Hu moments) verifying accurate topology management.

## Preliminaries

The level set method represents an etching front $\Gamma(t)$ as the zero level set of a higher-dimensional signed distance function $\phi(\mathbf{x}, t)$:

$$
\Gamma(t) = \{ \mathbf{x} \mid \phi(\mathbf{x}, t) = 0 \}
$$

where $\phi$ is defined with negative values inside the material and positive values in etched regions. This implicit representation automatically handles topological changes (splitting/merging) and complex geometries.

### Governing Equations

The evolution of $\phi$ is governed by the linear advection equation:

$$
\frac{\partial \phi}{\partial t} + \mathbf{U} \cdot \nabla \phi = 0
$$

Alternatively, it can be written in Hamilton-Jacobi form:

$$
\frac{\partial \phi}{\partial t} + F|\nabla \phi| = 0 \tag{1}
$$

where:

* $\mathbf{U}$ is the velocity field
* $F = \mathbf{U} \cdot \mathbf{n}$ is the normal velocity component
* $\mathbf{n} = \nabla \phi / |\nabla \phi|$ is the unit normal vector

#### Velocity Field

For semiconductor applications, the velocity field $\mathbf{U}$ is material-dependent:

$$
\mathbf{U(r)} = [\alpha R_m, \alpha R_m, R_m]^T 
$$

where:

* $R\_m$ is the vertical etching rate for material $m$
* $\alpha\_r$ controls the lateral-to-vertical etching ratio

#### Initial Condition

The initial signed distance field is:

$$
\phi(\mathbf{x}, 0) = \pm d(\mathbf{x}, \Gamma_0) \tag{2}
$$

where \$d\$ is the signed distance to the initial interface \$\Gamma\_0\$.

# Numerical Implementation

## Spatial Discretization

The hyperbolic convective term $\mathbf{U} \cdot \nabla \phi$ requires specialized discretization. The following schemes are implemented:

### First-Order Upwind (Finite Difference)

Basic and stable, but suffers from numerical diffusion:

$$
(\mathbf{U} \cdot \nabla \phi)_{ijk} = \sum_{\nu \in \{x,y,z\}} \left[ U_\nu^+ D^{-\nu}\phi + U_\nu^- D^{+\nu}\phi \right]
$$

where:

* $U\_\nu^+ = \max(U\_\nu, 0)$
* $U\_\nu^- = \min(U\_\nu, 0)$
* $D^{\pm\nu}$ are directional difference operators

### Roe's Scheme with MUSCL

A monotonicity-preserving scheme using piecewise linear reconstruction:

$$
\begin{aligned}
\phi_{i+1/2}^L &= \phi_i + \frac{1}{2}\psi(r_i)(\phi_i - \phi_{i-1}) \\
\phi_{i+1/2}^R &= \phi_{i+1} - \frac{1}{2}\psi(r_{i+1})(\phi_{i+2} - \phi_{i+1}) \\
F_{i+1/2} &= \frac{1}{2}\left[U\phi_L + U\phi_R - |U|(\phi_R - \phi_L)\right]
\end{aligned}
$$

Limiter function:
$$
\psi(r) = \max(0, \min(1, r))
$$


### Roe's Scheme with QUICK Scheme

Achieves higher accuracy via quadratic interpolation:

$$
\phi_L =
\begin{cases}
\frac{6\phi_{i-1} + 3\phi_i - \phi_{i-2}}{8} & \text{if } U_f \geq 0 \\
\frac{6\phi_i + 3\phi_{i-1} - \phi_{i+1}}{8} & \text{if } U_f < 0
\end{cases}, \quad
\phi_R =
\begin{cases}
\frac{6\phi_i + 3\phi_{i+1} - \phi_{i-1}}{8} & \text{if } U_f \geq 0 \\
\frac{6\phi_{i+1} + 3\phi_i - \phi_{i+2}}{8} & \text{if } U_f < 0
\end{cases}
$$

Limiter for oscillation control:

$$
\phi(r) = \frac{r + |r|}{1 + |r|}
$$


> **Note:** All schemes extend to 3D via dimension-wise operator splitting.
> Boundary conditions include Dirichlet ($\phi = \phi\_{\text{specified}}$) or Neumann ($\partial\phi/\partial n = 0$).

---

## Time Integration

We implement schemes that balance accuracy and stability:

| Method         | Order | Stability     | Cost                |
| -------------- | ----- | ------------- | ------------------- |
| Backward Euler | 1st   | Unconditional | High (linear solve) |
| Crank-Nicolson | 2nd   | Unconditional | High (linear solve) |
| TVD RK3        | 3rd   | CFL-limited   | Low (explicit)      |

---

### Backward Euler

$$
(I + \Delta t A)\phi^{n+1} = \phi^n
$$

### Crank-Nicolson

$$
\left(I + \frac{\Delta t}{2}A\right)\phi^{n+1} = \left(I - \frac{\Delta t}{2}A\right)\phi^n
$$

### TVD Runge-Kutta 3 (RK3)

$$
\begin{aligned}
\phi^{(1)} &= \phi^n + \Delta t \, L(\phi^n) \\
\phi^{(2)} &= \frac{3}{4}\phi^n + \frac{1}{4}\phi^{(1)} + \frac{1}{4}\Delta t \, L(\phi^{(1)}) \\
\phi^{n+1} &= \frac{1}{3}\phi^n + \frac{2}{3}\phi^{(2)} + \frac{2}{3}\Delta t \, L(\phi^{(2)})
\end{aligned}
$$

where:

$$
L(\phi) = -\mathbf{U} \cdot \nabla \phi
$$

with CFL condition:

$$
\Delta t \leq C \frac{\min(\Delta x, \Delta y, \Delta z)}{\max |\mathbf{U}|}, \quad C \leq 1.0
$$

## Reinitialization

To maintain the signed distance property (\$|\nabla \phi| = 1\$), solve the reinitialization equation:

$$
\frac{\partial\psi}{\partial\tau} = \text{sign}(\phi_0)(1 - |\nabla\psi|)
$$

where:

* Smoothed sign function:

  $$
  \text{sign}(\phi_0) = \frac{\phi_0}{\sqrt{\phi_0^2 + |\nabla\phi_0|^2 \epsilon^2}}, \quad \epsilon = 0.5 \Delta x
  $$
* $|\nabla\psi|$ is computed with central differences
* Forward Euler time stepping: $\Delta\tau = 0.1 \min \Delta x$
* Terminate when: $| |\nabla\psi| - 1 |\_{L^\infty} < 0.01$

> **Reinitialization** is executed every 5–10 physical time steps and parallelized using OpenMP.

# High-Performance Computing Strategies
To manage the computational demands of 3D semiconductor etching simulations, we adopt a high-performance computing technique that combines sparse linear algebra optimization with advanced solver configuration. Efficient solution of large sparse systems arising from implicit temporal discretization is achieved through parallel matrix assembly techniques and parallelism of the BiCGSTAB solver. This includes the use of Triplet storage with thread-local buffers, pre-computation of sparsity patterns to eliminate dynamic allocation during assembly, and Lock-free insertion strategies using OpenMP parallel. To enhance memory efficiency and performance, matrices are stored as compressed row storage (CRS) format employing blocked CRS layouts for improved cache locality, and utilize SIMD-optimized packing to increase arithmetic throughput. For solving the discretized Hamilton-Jacobi equations, we configure an accelerated BiCGSTAB solver based on Eigen's vectorized implementation. The solver is further enhanced with diagonal pre-conditioning, together with a strict convergence tolerance of  $10^{-8}$ to guarantee the accuracy of the results. Meanwhile, OpenMP enable the scaling of explicit method by parallelizing stencil operator over fixed grid.