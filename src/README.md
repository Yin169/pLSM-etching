# EDA Competition - Source Directory

## Overview
This directory contains the core C++ source code for the EDA Competition project. It provides modules for DF-ISE mesh parsing, level set evolution, time integration schemes, and geometry conversion utilities.

## Main Components

### DFISEParser (`DFISEParser.hpp`)
- Parses DF-ISE mesh files
- Extracts vertices, edges, faces, elements, and material mappings
- Supports exporting to OBJ format with material information

### Time Integration Schemes (`TimeScheme.hpp`)
- Defines an abstract base class for time-stepping methods
- Implementations include:
  - **BackwardEulerScheme**: Implicit scheme using BiCGSTAB solver
  - **CrankNicolsonMUSCLScheme**: Explicit Crank-Nicolson with MUSCL reconstruction
  - **TVDRK3RoeQUICKScheme**: Explicit TVD-RK3 with Roe/QUICK flux-limiting

### OBJ to BND Converter (`OBJToBNDConverter.hpp`)
- Converts OBJ mesh files to DF-ISE BND format
- Handles vertex, edge, and face extraction and material mapping

### Level Set Method (`LevelSetMethod.hpp` / `LevelSetMethod.cpp`)
- Implements level set evolution algorithms
- Utilizes velocity fields from parsed DF-ISE data
- Provides grid generation, mesh loading, signed distance initialization, and result export

### Conversion Utilities (`convert.hpp`)
- Provides functions for converting between supported mesh formats

### Main Application Entry (`levelset_main.cpp`)
- Entry point for running level set simulations and mesh processing

## Dependencies
- Eigen3 (linear algebra)
- OpenMP (parallel processing)
- C++17 standard

## Build Instructions
```bash
bash run.sh
```

## File Structure
```
src/
├── DFISEParser.hpp         # DF-ISE file parser
├── TimeScheme.hpp          # Time integration schemes
├── OBJToBNDConverter.hpp   # Geometry converter
├── LevelSetMethod.hpp/.cpp # Level set algorithms
├── convert.hpp             # Conversion utilities
├── levelset_main.cpp       # Main application entry
└── output_helper.h         # Output filename utilities
```

## Level Set Method Flow
```
LevelSetMethod
|
+--> generateGrid()
|
+--> loadMesh()
|    |
|    +--> Create AABB tree
|
+--> evolve()
|    |
|    +--> initializeSignedDistanceField()
|    |    |
|    |    +--> Use tree to check if points are inside mesh
|    |
|    +--> isOnBoundary()
|    +--> getIndex()
|    +--> reinitialize()
|         |
|         +--> isOnBoundary()
|         +--> getIndex()
|
+--> saveResult()
|    |
|    +--> Write results to file
|
+--> extractSurfaceMeshCGAL()
|
+--> Create LevelSetImplicitFunction
     |
     +--> Use phi and grid data
     +--> getIndex() (indirectly)
```