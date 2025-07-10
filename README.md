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

The level set method represents an etching front \$\Gamma(t)\$ as the zero level set of a higher-dimensional signed distance function \$\phi(\mathbf{x}, t)\$:

$$
\Gamma(t) = \{ \mathbf{x} \mid \phi(\mathbf{x}, t) = 0 \}
$$

where \$\phi\$ is defined with negative values inside the material and positive values in etched regions. This implicit representation automatically handles topological changes (splitting/merging) and complex geometries.

### Governing Equations

The evolution of \$\phi\$ is governed by the linear advection equation:

$$
\frac{\partial \phi}{\partial t} + \mathbf{U} \cdot \nabla \phi = 0
$$

Alternatively, it can be written in Hamilton-Jacobi form:

$$
\frac{\partial \phi}{\partial t} + F|\nabla \phi| = 0 \tag{1}
$$

where:

* \$\mathbf{U}\$ is the velocity field
* \$F = \mathbf{U} \cdot \mathbf{n}\$ is the normal velocity component
* \$\mathbf{n} = \nabla \phi / |\nabla \phi|\$ is the unit normal vector

#### Velocity Field

For semiconductor applications, the velocity field \$\mathbf{U}\$ is material-dependent:

$$
\mathbf{U}(\mathbf{x}) =
\begin{cases}
-\begin{pmatrix} \alpha_r R_m, & \alpha_r R_m, & R_m \end{pmatrix} & \text{if } \mathbf{x} \in \text{Material } m \\
0 & \text{otherwise}
\end{cases}
$$

where:

* \$R\_m\$ is the vertical etching rate for material \$m\$
* \$\alpha\_r\$ controls the lateral-to-vertical etching ratio

#### Initial Condition

The initial signed distance field is:

$$
\phi(\mathbf{x}, 0) = \pm d(\mathbf{x}, \Gamma_0) \tag{2}
$$

where \$d\$ is the signed distance to the initial interface \$\Gamma\_0\$.


### Numerical Implementation

Equations (1) and (2) are solved on a 3D structured grid using finite differences.

Let \$\phi\_{ijk}^{n}\$ denote the value of \$\phi\$ at grid point \$(i, j, k)\$ and timestep \$n\$. A first-order upwind discretization of Eq. (1) is:

$$
\phi_{ijk}^{n+1} = \phi_{ijk}^{n} - \Delta t \left[ \max(F_{ijk}, 0) \nabla^+ + \min(F_{ijk}, 0) \nabla^- \right]
$$

#### Spatial Derivatives

The upwind spatial derivatives are computed as:

$$
\nabla^+ = \sqrt{ \sum_{\nu \in \{x, y, z\}} \left[ \max(D^{-\nu} \phi, 0)^2 + \min(D^{+\nu} \phi, 0)^2 \right] }
$$

$$
\nabla^- = \sqrt{ \sum_{\nu \in \{x, y, z\}} \left[ \max(D^{+\nu} \phi, 0)^2 + \min(D^{-\nu} \phi, 0)^2 \right] }
$$

where \$D^{\pm \nu}\$ are directional difference operators (forward/backward differences along direction \$\nu\$).

#### Time Integration

Time advancement is done using a first-order forward Euler method.

