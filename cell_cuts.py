import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from readfile import DFISEParser

# Initialize Taichi with CPU for debugging, switch to GPU for production
ti.init(arch=ti.cpu, default_fp=ti.f32)  # Use CPU for debugging, more stable

@ti.data_oriented
class CellCutsMesh:
    def __init__(self, geometry_parser, nx=50, ny=50, nz=50, bounds=None):
        """
        Initialize the cell-cuts mesh generator.
        """
        self.parser = geometry_parser
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        # Get geometry data
        self.vertices = self.parser.get_vertices()
        self.edges = self.parser.get_edges()
        self.faces = self.parser.get_faces()
        
        # Convert geometry data to Taichi fields for faster access
        self.n_vertices = len(self.vertices)
        self.n_edges = len(self.edges)
        self.n_faces = len(self.faces)
        
        # Create Taichi fields for geometry
        self.ti_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.n_vertices)
        self.ti_edges = ti.Vector.field(2, dtype=ti.i32, shape=self.n_edges)
        
        # Transfer geometry data to Taichi fields
        for i in range(self.n_vertices):
            self.ti_vertices[i] = self.vertices[i]
        
        for i in range(self.n_edges):
            # Adjust for 0-based indexing in Taichi
            v1, v2 = self.edges[i]
            self.ti_edges[i] = [v1-1, v2-1]  # Convert from 1-based to 0-based
        
        # Determine bounds if not provided
        if bounds is None:
            self.determine_bounds()
        else:
            self.xmin, self.xmax = bounds[0]
            self.ymin, self.ymax = bounds[1]
            self.zmin, self.zmax = bounds[2]
        
        # Calculate cell sizes
        self.dx = (self.xmax - self.xmin) / nx
        self.dy = (self.ymax - self.ymin) / ny
        self.dz = (self.zmax - self.zmin) / nz
        
        # Initialize Taichi fields
        self.grid_points = ti.Vector.field(3, dtype=ti.f32, shape=(nx+1, ny+1, nz+1))
        self.cell_types = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        
        # Store cut cells in a Taichi field for faster access
        self.cut_cell_count = ti.field(dtype=ti.i32, shape=())
        self.cut_cell_count[None] = 0
        self.max_cut_cells = nx * ny * nz  # Maximum possible number of cut cells
        self.cut_cell_indices = ti.Vector.field(3, dtype=ti.i32, shape=self.max_cut_cells)
        
        # Generate grid points
        self.generate_grid_points()
        
        # Initialize cell data
        self.cell_types.fill(0)  # 0: outside, 1: inside, 2: cut
        
    def determine_bounds(self):
        """Determine the bounds of the mesh based on the geometry."""
        if len(self.vertices) == 0:
            # Default bounds if no vertices
            self.xmin, self.xmax = -1.0, 1.0
            self.ymin, self.ymax = -1.0, 1.0
            self.zmin, self.zmax = -1.0, 1.0
            return
        
        # Get min and max coordinates with a small buffer
        buffer = 0.1
        mins = np.min(self.vertices, axis=0)
        maxs = np.max(self.vertices, axis=0)
        
        range_x = maxs[0] - mins[0]
        range_y = maxs[1] - mins[1]
        range_z = maxs[2] - mins[2]
        
        self.xmin = mins[0] - buffer * range_x
        self.xmax = maxs[0] + buffer * range_x
        self.ymin = mins[1] - buffer * range_y
        self.ymax = maxs[1] + buffer * range_y
        self.zmin = mins[2] - buffer * range_z
        self.zmax = maxs[2] + buffer * range_z
    
    @ti.kernel
    def generate_grid_points(self):
        """Generate the grid points for the mesh using Taichi kernel."""
        for i, j, k in ti.ndrange(self.nx + 1, self.ny + 1, self.nz + 1):
            x = self.xmin + i * self.dx
            y = self.ymin + j * self.dy
            z = self.zmin + k * self.dz
            self.grid_points[i, j, k] = ti.Vector([x, y, z])
    
    # Simplified point-in-geometry check using Taichi
    @ti.func
    def point_in_geometry_fast(self, point):
        """Fast version of point-in-geometry check for Taichi kernels."""
        inside = False
        # Simple bounding box check first (optimization)
        if (self.xmin <= point[0] <= self.xmax and 
            self.ymin <= point[1] <= self.ymax and 
            self.zmin <= point[2] <= self.zmax):
            
            # Cast a ray in the x direction and count intersections
            intersections = 0
            # This is a simplified version - in a real implementation
            # you would need to check against all faces
            for i in range(min(100, self.n_edges)):  # Limit check to avoid too many iterations
                edge = self.ti_edges[i]
                v1 = self.ti_vertices[edge[0]]
                v2 = self.ti_vertices[edge[1]]
                
                # Very simplified intersection test
                if (min(v1[1], v2[1]) <= point[1] <= max(v1[1], v2[1]) and
                    min(v1[2], v2[2]) <= point[2] <= max(v1[2], v2[2]) and
                    max(v1[0], v2[0]) > point[0] and
                    min(v1[0], v2[0]) <= point[0]):
                    intersections += 1
            
            inside = (intersections % 2) == 1
        
        return inside
    
    @ti.func
    def edge_intersects_cell_fast(self, edge_idx, i, j, k):
        """Fast version of edge-intersects-cell check for Taichi kernels."""
        edge = self.ti_edges[edge_idx]
        v1 = self.ti_vertices[edge[0]]
        v2 = self.ti_vertices[edge[1]]
        
        # Get the bounds of the cell
        xmin = self.xmin + i * self.dx
        xmax = self.xmin + (i + 1) * self.dx
        ymin = self.ymin + j * self.dy
        ymax = self.ymin + (j + 1) * self.dy
        zmin = self.zmin + k * self.dz
        zmax = self.zmin + (k + 1) * self.dz
        
        # Check if either vertex is inside the cell
        v1_in_cell = (xmin <= v1[0] <= xmax and 
                      ymin <= v1[1] <= ymax and 
                      zmin <= v1[2] <= zmax)
        
        v2_in_cell = (xmin <= v2[0] <= xmax and 
                      ymin <= v2[1] <= ymax and 
                      zmin <= v2[2] <= zmax)
        
        result = False
        if v1_in_cell or v2_in_cell:
            result = True
        else:
            # Direction vector of the edge
            dir_x = v2[0] - v1[0]
            dir_y = v2[1] - v1[1]
            dir_z = v2[2] - v1[2]
            
            # Check intersection with x-planes
            if dir_x != 0:
                t1 = (xmin - v1[0]) / dir_x
                if 0 <= t1 <= 1:
                    y = v1[1] + t1 * dir_y
                    z = v1[2] + t1 * dir_z
                    if ymin <= y <= ymax and zmin <= z <= zmax:
                        result = True
                
                t2 = (xmax - v1[0]) / dir_x
                if 0 <= t2 <= 1:
                    y = v1[1] + t2 * dir_y
                    z = v1[2] + t2 * dir_z
                    if ymin <= y <= ymax and zmin <= z <= zmax:
                        result = True
            
            # Check intersection with y-planes
            if not result and dir_y != 0:
                t3 = (ymin - v1[1]) / dir_y
                if 0 <= t3 <= 1:
                    x = v1[0] + t3 * dir_x
                    z = v1[2] + t3 * dir_z
                    if xmin <= x <= xmax and zmin <= z <= zmax:
                        result = True
                
                t4 = (ymax - v1[1]) / dir_y
                if 0 <= t4 <= 1:
                    x = v1[0] + t4 * dir_x
                    z = v1[2] + t4 * dir_z
                    if xmin <= x <= xmax and zmin <= z <= zmax:
                        result = True
            
            # Check intersection with z-planes
            if not result and dir_z != 0:
                t5 = (zmin - v1[2]) / dir_z
                if 0 <= t5 <= 1:
                    x = v1[0] + t5 * dir_x
                    y = v1[1] + t5 * dir_y
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        result = True
                
                t6 = (zmax - v1[2]) / dir_z
                if 0 <= t6 <= 1:
                    x = v1[0] + t6 * dir_x
                    y = v1[1] + t6 * dir_y
                    if xmin <= x <= xmax and ymin <= y <= ymax:
                        result = True
        
        return result
    
    @ti.kernel
    def classify_cells_kernel(self):
        """Classify cells using Taichi kernel for better performance."""
        # Reset cut cell count
        self.cut_cell_count[None] = 0
        
        # Process all cells in parallel
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            # Calculate cell center
            center_x = self.xmin + (i + 0.5) * self.dx
            center_y = self.ymin + (j + 0.5) * self.dy
            center_z = self.zmin + (k + 0.5) * self.dz
            center = ti.Vector([center_x, center_y, center_z])
            
            # Check if center is inside geometry
            if self.point_in_geometry_fast(center):
                self.cell_types[i, j, k] = 1  # Inside
            
            # Check if any edge intersects with the cell
            for e in range(self.n_edges):
                if self.edge_intersects_cell_fast(e, i, j, k):
                    self.cell_types[i, j, k] = 2  # Cut
                    
                    # Store cut cell index
                    idx = ti.atomic_add(self.cut_cell_count[None], 1)
                    if idx < self.max_cut_cells:
                        self.cut_cell_indices[idx] = ti.Vector([i, j, k])
                    
                    break
    
    def classify_cells(self):
        """Classify cells as inside, outside, or cut by the geometry."""
        # Use the Taichi kernel for classification
        self.classify_cells_kernel()
        
        # Extract cut cells to Python list
        cut_cells = []
        count = min(self.cut_cell_count[None], self.max_cut_cells)
        for i in range(count):
            idx = self.cut_cell_indices[i]
            cut_cells.append((idx[0], idx[1], idx[2]))
        
        return cut_cells
    
    def generate_mesh(self):
        """Generate the mesh using the cell-cuts method."""
        # Classify cells
        cut_cells = self.classify_cells()
        
        # Convert Taichi fields to NumPy arrays for return
        # Use more efficient approach with ti.to_numpy()
        grid_points_np = self.grid_points.to_numpy()
        cell_types_np = self.cell_types.to_numpy()
        
        # Return the mesh data
        return {
            'grid_points': grid_points_np,
            'cell_types': cell_types_np,
            'cut_cells': cut_cells,
            'bounds': ((self.xmin, self.xmax), (self.ymin, self.ymax), (self.zmin, self.zmax)),
            'dimensions': (self.nx, self.ny, self.nz)
        }
    
    def plot_mesh(self, show_cut_cells=True, show_inside_cells=True, show_geometry=True):
        """
        Plot the mesh with the embedded geometry.
        
        Parameters:
        -----------
        show_cut_cells : bool
            Whether to show cells cut by the geometry
        show_inside_cells : bool
            Whether to show cells inside the geometry
        show_geometry : bool
            Whether to show the original geometry
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the grid points
        if show_inside_cells or show_cut_cells:
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        cell_type = self.cell_types[i, j, k]
                        if (show_inside_cells and cell_type == 1) or \
                           (show_cut_cells and cell_type == 2):
                            # Get the corners of the cell
                            x0, y0, z0 = self.grid_points[i, j, k]
                            x1, y1, z1 = self.grid_points[i+1, j+1, k+1]
                            
                            # Plot the cell as a wireframe box
                            color = 'g' if cell_type == 1 else 'r'
                            alpha = 0.3 if cell_type == 1 else 0.7
                            
                            # Create the wireframe box
                            for edge in [
                                # Bottom face
                                [[x0, y0, z0], [x1, y0, z0]],
                                [[x1, y0, z0], [x1, y1, z0]],
                                [[x1, y1, z0], [x0, y1, z0]],
                                [[x0, y1, z0], [x0, y0, z0]],
                                # Top face
                                [[x0, y0, z1], [x1, y0, z1]],
                                [[x1, y0, z1], [x1, y1, z1]],
                                [[x1, y1, z1], [x0, y1, z1]],
                                [[x0, y1, z1], [x0, y0, z1]],
                                # Connecting edges
                                [[x0, y0, z0], [x0, y0, z1]],
                                [[x1, y0, z0], [x1, y0, z1]],
                                [[x1, y1, z0], [x1, y1, z1]],
                                [[x0, y1, z0], [x0, y1, z1]]
                            ]:
                                ax.plot([edge[0][0], edge[1][0]],
                                        [edge[0][1], edge[1][1]],
                                        [edge[0][2], edge[1][2]],
                                        color=color, alpha=alpha)
        
        # Plot the original geometry
        if show_geometry:
            vertices = self.vertices
            edges = self.edges
            
            # Plot vertices
            ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       color='b', s=10, label='Vertices')
            
            # Plot edges
            for edge in edges:
                if len(edge) == 2:
                    v1, v2 = edge
                    # Check if indices are valid (adjust for 1-based indexing)
                    if 0 <= v1-1 < len(vertices) and 0 <= v2-1 < len(vertices):
                        ax.plot([vertices[v1-1, 0], vertices[v2-1, 0]],
                                [vertices[v1-1, 1], vertices[v2-1, 1]],
                                [vertices[v1-1, 2], vertices[v2-1, 2]],
                                color='b', linewidth=1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Cell-Cuts Mesh with Embedded Geometry')
        
        # Add legend
        if show_inside_cells:
            ax.plot([], [], [], color='g', alpha=0.3, label='Inside Cells')
        if show_cut_cells:
            ax.plot([], [], [], color='r', alpha=0.7, label='Cut Cells')
        if show_geometry:
            ax.plot([], [], [], color='b', linewidth=1, label='Geometry')
        
        ax.legend()
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_with_gui(self, show_cut_cells=True, show_inside_cells=True, show_geometry=True, 
                          show_slice=False, slice_axis='z', slice_pos=0.5):
        """
        Visualize the mesh using Taichi GUI.
        
        Parameters:
        -----------
        show_cut_cells : bool
            Whether to show cells cut by the geometry
        show_inside_cells : bool
            Whether to show cells inside the geometry
        show_geometry : bool
            Whether to show the original geometry
        show_slice : bool
            Whether to show a slice through the mesh
        slice_axis : str
            Axis to slice along ('x', 'y', or 'z')
        slice_pos : float
            Position of the slice along the axis (0.0 to 1.0)
        """
        # Create a Taichi GUI window
        window = ti.ui.Window("Cell-Cuts Mesh Visualization", (1024, 768))
        canvas = window.get_canvas()
        scene = window.get_scene()
        
        # Set up camera
        camera = ti.ui.make_camera()
        camera.position(2.0, 2.0, 2.0)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 1.0, 0.0)
        
        # Normalize coordinates to [-1, 1] for better visualization
        x_range = self.xmax - self.xmin
        y_range = self.ymax - self.ymin
        z_range = self.zmax - self.zmin
        max_range = max(x_range, y_range, z_range)
        
        # Create fields for visualization
        max_edges = 100000  # Maximum number of edges to visualize
        edge_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(max_edges * 2))
        edge_colors = ti.Vector.field(3, dtype=ti.f32, shape=(max_edges * 2))
        
        # Create an atomic counter for edge count
        edge_counter = ti.field(dtype=ti.i32, shape=())
        
        # Calculate slice position in world coordinates
        slice_val = 0.0
        if slice_axis == 'x':
            slice_val = self.xmin + slice_pos * x_range
        elif slice_axis == 'y':
            slice_val = self.ymin + slice_pos * y_range
        elif slice_axis == 'z':
            slice_val = self.zmin + slice_pos * z_range
        
        # Function to normalize coordinates
        @ti.func
        def normalize_coord(x, y, z):
            nx = 2.0 * (x - self.xmin) / max_range - x_range / max_range
            ny = 2.0 * (y - self.ymin) / max_range - y_range / max_range
            nz = 2.0 * (z - self.zmin) / max_range - z_range / max_range
            return ti.Vector([nx, ny, nz])
        
        # Function to check if a cell intersects with the slice plane
        @ti.func
        def cell_intersects_slice(i, j, k):
            x0, y0, z0 = self.grid_points[i, j, k]
            x1, y1, z1 = self.grid_points[i+1, j+1, k+1]
            
            result = False
            if slice_axis == 'x':
                result = x0 <= slice_val <= x1
            elif slice_axis == 'y':
                result = y0 <= slice_val <= y1
            else:  # z-axis
                result = z0 <= slice_val <= z1
            
            return result
        
        # Kernel to prepare visualization data
        @ti.kernel
        def prepare_visualization():
            # Reset edge counter
            edge_counter[None] = 0
            
            # Add cell edges
            if show_inside_cells or show_cut_cells:
                for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
                    cell_type = self.cell_types[i, j, k]
                    
                    # Check if we should show this cell
                    show_cell = False
                    if show_slice:
                        # Only show cells that intersect with the slice plane
                        show_cell = cell_intersects_slice(i, j, k) and (
                            (show_inside_cells and cell_type == 1) or 
                            (show_cut_cells and cell_type == 2)
                        )
                    else:
                        # Show all cells based on type
                        show_cell = (show_inside_cells and cell_type == 1) or \
                                   (show_cut_cells and cell_type == 2)
                    
                    if show_cell:
                        # Get the corners of the cell
                        x0, y0, z0 = self.grid_points[i, j, k]
                        x1, y1, z1 = self.grid_points[i+1, j+1, k+1]
                        
                        # Set color based on cell type
                        color = ti.Vector([0.0, 1.0, 0.0]) if cell_type == 1 else ti.Vector([1.0, 0.0, 0.0])
                        
                        # Add edges for the cell (12 edges for a cube)
                        if edge_counter[None] < max_edges - 12:
                            # Bottom face
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x0, y0, z0)
                            edge_vertices[idx*2+1] = normalize_coord(x1, y0, z0)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x1, y0, z0)
                            edge_vertices[idx*2+1] = normalize_coord(x1, y1, z0)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x1, y1, z0)
                            edge_vertices[idx*2+1] = normalize_coord(x0, y1, z0)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x0, y1, z0)
                            edge_vertices[idx*2+1] = normalize_coord(x0, y0, z0)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            # Top face
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x0, y0, z1)
                            edge_vertices[idx*2+1] = normalize_coord(x1, y0, z1)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x1, y0, z1)
                            edge_vertices[idx*2+1] = normalize_coord(x1, y1, z1)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x1, y1, z1)
                            edge_vertices[idx*2+1] = normalize_coord(x0, y1, z1)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x0, y1, z1)
                            edge_vertices[idx*2+1] = normalize_coord(x0, y0, z1)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            # Connecting edges
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x0, y0, z0)
                            edge_vertices[idx*2+1] = normalize_coord(x0, y0, z1)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x1, y0, z0)
                            edge_vertices[idx*2+1] = normalize_coord(x1, y0, z1)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x1, y1, z0)
                            edge_vertices[idx*2+1] = normalize_coord(x1, y1, z1)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
                            
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = normalize_coord(x0, y1, z0)
                            edge_vertices[idx*2+1] = normalize_coord(x0, y1, z1)
                            edge_colors[idx*2] = color
                            edge_colors[idx*2+1] = color
            
            # Add geometry edges
            if show_geometry and (not show_slice):
                blue_color = ti.Vector([0.0, 0.0, 1.0])
                for e in range(self.n_edges):
                    if edge_counter[None] < max_edges:
                        edge = self.ti_edges[e]
                        v1 = self.ti_vertices[edge[0]]
                        v2 = self.ti_vertices[edge[1]]
                        
                        idx = ti.atomic_add(edge_counter[None], 1)
                        edge_vertices[idx*2] = normalize_coord(v1[0], v1[1], v1[2])
                        edge_vertices[idx*2+1] = normalize_coord(v2[0], v2[1], v2[2])
                        edge_colors[idx*2] = blue_color
                        edge_colors[idx*2+1] = blue_color
            
            # If showing slice, add a slice plane visualization
            if show_slice:
                # Create a slice plane with a grid
                yellow_color = ti.Vector([1.0, 1.0, 0.0])
                grid_size = 10  # Number of grid lines in each direction
                
                if slice_axis == 'x':
                    # Create a YZ plane at the slice position
                    norm_x = normalize_coord(slice_val, 0, 0)[0]
                    
                    # Add horizontal grid lines (Z direction)
                    for i in range(grid_size + 1):
                        y_pos = self.ymin + (i / grid_size) * y_range
                        norm_y_min = normalize_coord(0, y_pos, self.zmin)[1]
                        norm_y_max = normalize_coord(0, y_pos, self.zmax)[1]
                        norm_z_min = normalize_coord(0, 0, self.zmin)[2]
                        norm_z_max = normalize_coord(0, 0, self.zmax)[2]
                        
                        if edge_counter[None] < max_edges:
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = ti.Vector([norm_x, norm_y_min, norm_z_min])
                            edge_vertices[idx*2+1] = ti.Vector([norm_x, norm_y_min, norm_z_max])
                            edge_colors[idx*2] = yellow_color
                            edge_colors[idx*2+1] = yellow_color
                    
                    # Add vertical grid lines (Y direction)
                    for i in range(grid_size + 1):
                        z_pos = self.zmin + (i / grid_size) * z_range
                        norm_z = normalize_coord(0, 0, z_pos)[2]
                        
                        if edge_counter[None] < max_edges:
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = ti.Vector([norm_x, normalize_coord(0, self.ymin, 0)[1], norm_z])
                            edge_vertices[idx*2+1] = ti.Vector([norm_x, normalize_coord(0, self.ymax, 0)[1], norm_z])
                            edge_colors[idx*2] = yellow_color
                            edge_colors[idx*2+1] = yellow_color
                
                elif slice_axis == 'y':
                    # Create an XZ plane at the slice position
                    norm_y = normalize_coord(0, slice_val, 0)[1]
                    
                    # Add horizontal grid lines (Z direction)
                    for i in range(grid_size + 1):
                        x_pos = self.xmin + (i / grid_size) * x_range
                        norm_x = normalize_coord(x_pos, 0, 0)[0]
                        
                        if edge_counter[None] < max_edges:
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = ti.Vector([norm_x, norm_y, normalize_coord(0, 0, self.zmin)[2]])
                            edge_vertices[idx*2+1] = ti.Vector([norm_x, norm_y, normalize_coord(0, 0, self.zmax)[2]])
                            edge_colors[idx*2] = yellow_color
                            edge_colors[idx*2+1] = yellow_color
                    
                    # Add vertical grid lines (X direction)
                    for i in range(grid_size + 1):
                        z_pos = self.zmin + (i / grid_size) * z_range
                        norm_z = normalize_coord(0, 0, z_pos)[2]
                        
                        if edge_counter[None] < max_edges:
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = ti.Vector([normalize_coord(self.xmin, 0, 0)[0], norm_y, norm_z])
                            edge_vertices[idx*2+1] = ti.Vector([normalize_coord(self.xmax, 0, 0)[0], norm_y, norm_z])
                            edge_colors[idx*2] = yellow_color
                            edge_colors[idx*2+1] = yellow_color
                
                else:  # z-axis
                    # Create an XY plane at the slice position
                    norm_z = normalize_coord(0, 0, slice_val)[2]
                    
                    # Add horizontal grid lines (Y direction)
                    for i in range(grid_size + 1):
                        x_pos = self.xmin + (i / grid_size) * x_range
                        norm_x = normalize_coord(x_pos, 0, 0)[0]
                        
                        if edge_counter[None] < max_edges:
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = ti.Vector([norm_x, normalize_coord(0, self.ymin, 0)[1], norm_z])
                            edge_vertices[idx*2+1] = ti.Vector([norm_x, normalize_coord(0, self.ymax, 0)[1], norm_z])
                            edge_colors[idx*2] = yellow_color
                            edge_colors[idx*2+1] = yellow_color
                    
                    # Add vertical grid lines (X direction)
                    for i in range(grid_size + 1):
                        y_pos = self.ymin + (i / grid_size) * y_range
                        norm_y = normalize_coord(0, y_pos, 0)[1]
                        
                        if edge_counter[None] < max_edges:
                            idx = ti.atomic_add(edge_counter[None], 1)
                            edge_vertices[idx*2] = ti.Vector([normalize_coord(self.xmin, 0, 0)[0], norm_y, norm_z])
                            edge_vertices[idx*2+1] = ti.Vector([normalize_coord(self.xmax, 0, 0)[0], norm_y, norm_z])
                            edge_colors[idx*2] = yellow_color
                            edge_colors[idx*2+1] = yellow_color
        
        # Variables to track slice position and axis
        current_slice_pos = slice_pos
        current_slice_axis = slice_axis
        slice_mode = show_slice
        
        # Main rendering loop
        while window.running:
            # Update camera
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
            scene.set_camera(camera)
            
            # Handle keyboard input for slice control
            if window.get_event(ti.ui.PRESS):
                if window.event.key == 'x':
                    current_slice_axis = 'x'
                    slice_mode = True
                elif window.event.key == 'y':
                    current_slice_axis = 'y'
                    slice_mode = True
                elif window.event.key == 'z':
                    current_slice_axis = 'z'
                    slice_mode = True
                elif window.event.key == 's':
                    slice_mode = not slice_mode
            
            # Handle slice position adjustment with arrow keys
            if slice_mode:
                if window.is_pressed(ti.ui.RIGHT):
                    current_slice_pos = min(1.0, current_slice_pos + 0.01)
                if window.is_pressed(ti.ui.LEFT):
                    current_slice_pos = max(0.0, current_slice_pos - 0.01)
            
            # Update slice values if changed
            if slice_axis != current_slice_axis or slice_pos != current_slice_pos or show_slice != slice_mode:
                slice_axis = current_slice_axis
                slice_pos = current_slice_pos
                show_slice = slice_mode
                
                # Recalculate slice position in world coordinates
                if slice_axis == 'x':
                    slice_val = self.xmin + slice_pos * x_range
                elif slice_axis == 'y':
                    slice_val = self.ymin + slice_pos * y_range
                elif slice_axis == 'z':
                    slice_val = self.zmin + slice_pos * z_range
            
            # Clear scene
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
            
            # Prepare visualization data
            prepare_visualization()
            
            # Draw lines
            scene.lines(edge_vertices, per_vertex_color=edge_colors, width=1.0)
            
            # Display slice information
            if slice_mode:
                window.GUI.begin("Slice Controls", 0.02, 0.02, 0.3, 0.2)
                window.GUI.text(f"Slice Axis: {slice_axis.upper()}")
                window.GUI.text(f"Slice Position: {slice_pos:.2f}")
                window.GUI.text("Press X/Y/Z to change axis")
                window.GUI.text("Press LEFT/RIGHT to adjust position")
                window.GUI.text("Press S to toggle slice mode")
                window.GUI.end()
            
            # Render scene
            canvas.scene(scene)
            
            # Show window
            window.show()