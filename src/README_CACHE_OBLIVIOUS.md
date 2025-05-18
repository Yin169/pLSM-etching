# Cache-Oblivious Level Set Method Implementation

This implementation extends the standard Level Set Method with a cache-oblivious algorithm that improves performance through recursive space-time decomposition. The cache-oblivious approach optimizes memory access patterns without explicit knowledge of cache parameters, leading to better performance across different hardware architectures.

## Key Features

- **Cache-Oblivious Algorithm**: Uses recursive trapezoidal decomposition of the space-time domain to improve cache efficiency
- **Parallel Execution**: Leverages OpenMP for multi-threaded computation
- **Precomputed Data**: Caches material properties and boundary status to reduce redundant calculations
- **Compatible API**: Inherits from the base LevelSetMethod class, making it a drop-in replacement

## How It Works

The cache-oblivious algorithm divides the computation domain recursively:

1. **Space-Time Decomposition**: The algorithm recursively divides the computation domain into trapezoids in space and time
2. **Adaptive Cutting Strategy**: Cuts are made in space or time based on the aspect ratio of the current trapezoid
3. **Base Case Processing**: At the lowest level, processes a single time step for a range of grid points
4. **Parallel Execution**: Uses OpenMP tasks for parallel processing of independent trapezoids

## Usage

To use the cache-oblivious implementation instead of the standard level set method, simply replace:

```cpp
LevelSetMethod levelSet(/* parameters */);
```

with:

```cpp
LevelSetMethodCacheOblivious levelSet(/* parameters */);
```

The constructor parameters and API are identical to the base LevelSetMethod class.

## Compiling the Example

Compile the example program with:

```bash
g++ -std=c++17 -fopenmp -O3 CacheObliviousExample.cpp LevelSetMethodCacheOblivious.cpp LevelSetMethod.cpp -o cache_oblivious_example -I/path/to/eigen -I/path/to/cgal/include
```

Make sure to include the necessary libraries for Eigen and CGAL.

## Running the Example

Run the example with:

```bash
./cache_oblivious_example <mesh_file> <org_file> <material_csv_file>
```

Where:
- `<mesh_file>` is the path to your input mesh file
- `<org_file>` is the path to your organization file
- `<material_csv_file>` is the path to your material properties CSV file

## Performance Considerations

- The cache-oblivious approach generally performs better on larger grid sizes
- The performance improvement is more significant when the computation is memory-bound
- For small problems that fit entirely in cache, the standard implementation might be faster due to lower overhead

## Implementation Details

The implementation consists of two main files:

1. `LevelSetMethodCacheOblivious.hpp`: Class definition that inherits from LevelSetMethod
2. `LevelSetMethodCacheOblivious.cpp`: Implementation of the overridden evolve() method

The key algorithm is in the `processParallelTrapezoid()` and `processTrapezoid()` methods, which implement the recursive space-time decomposition.