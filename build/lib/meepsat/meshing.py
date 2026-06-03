from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import numpy as np
import matplotlib.pyplot as plt
import math

import meep as mp
from scipy.spatial import Delaunay


# Constants for mesh generation
"""
Module-level constants for triangular mesh generation.
MIN_POINTS_FOR_TRIANGULATION : int
    Minimum number of points required to perform a valid triangulation.
    Set to 3, which is the minimum number of vertices needed to form
    a triangle (simplex in 2D space).
INTERIOR_POINTS_TARGET : int
    Target number of interior points to sample from the valid region
    during triangular mesh generation. Set to 50, which balances mesh
    quality and computational efficiency. Interior points are randomly
    downsampled from the valid region to avoid excessive mesh density
    while maintaining adequate representation of the geometry.
"""
global MIN_POINTS_FOR_TRIANGULATION
global INTERIOR_POINTS_TARGET
MIN_POINTS_FOR_TRIANGULATION = 3
INTERIOR_POINTS_TARGET = 50

def _create_triangular_mesh(
                            epsilon_array, 
                            epsilon_val,
                            grid_size_sx, 
                            grid_size_sy, 
                            resolution,
                            filter_option = "min",
                            plot= False,
                            figname= 'triangular_mesh.png'
                            ):
    """Create triangular mesh from absorber boundary."""
    from scipy import ndimage
    from scipy.spatial import ConvexHull
    from matplotlib.tri import Triangulation
    
    if filter_option == "min":
        binary_array = (epsilon_array > 1.0).astype(int)
    elif filter_option == "max":
        binary_array = (epsilon_array < 1.0).astype(int)
    elif filter_option == "equal":
        binary_array = (epsilon_array == 1.0).astype(int)
        
    boundary = ndimage.binary_erosion(binary_array) ^ binary_array
    
    indices = np.where(boundary)
    absorber_points = np.column_stack((indices[1] / resolution, indices[0] / resolution))
    
    if len(absorber_points) < MIN_POINTS_FOR_TRIANGULATION:
        return None
    
    hull = ConvexHull(absorber_points)
    boundary_points = absorber_points[hull.vertices]
    
    interior_mask = binary_array > 0
    interior_indices = np.where(interior_mask)
    step = max(1, len(interior_indices[0]) // INTERIOR_POINTS_TARGET)
    interior_points = np.column_stack((interior_indices[1][::step] / resolution,
                                    interior_indices[0][::step] / resolution))
    
    all_points = np.vstack([boundary_points, interior_points])
    tri = Triangulation(all_points[:, 0], all_points[:, 1])
    
    if plot:
        _visualize_triangular_mesh(tri, grid_size_sx, grid_size_sy, output_file=figname)

    return tri

def _visualize_triangular_mesh(tri, grid_size_sx, grid_size_sy, output_file='triangular_mesh.png'):
    """Visualize and save triangular mesh."""
    tri_x_shifted = tri.x - grid_size_sx/2
    tri_y_shifted = tri.y - grid_size_sy/2
    
    plt.figure(figsize=(12, 10))
    plt.triplot(tri_x_shifted, tri_y_shifted, tri.triangles, 
                linewidth=0.8, color='navy', alpha=0.8)
    plt.title("Triangular Mesh in Absorber", fontsize=14, fontweight='bold')
    plt.xlabel("X (mm)", fontsize=12)
    plt.ylabel("Y (mm)", fontsize=12)
    plt.xlim(-grid_size_sx/2, grid_size_sx/2)
    plt.ylim(-grid_size_sy/2, grid_size_sy/2)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def convert_triangles_to_prisms(
                                tri, 
                                gridx_size_mm, 
                                gridy_size_mm,
                                material, 
                                thickness=1.0):
    """
    Convert 2D triangulation to 3D MEEP Prism objects.
    
    Parameters:
    -----------
    tri : Triangulation object
        2D triangulation from matplotlib (in pixel coordinates)
    material_eps : float
        Material permittivity value
    thickness : float
        Height/thickness of the prisms in z-direction
        TODO: NEED TO COMEBACK FOR THIS PARAMETER WHILE DOING 3D SIMS

    Returns:
    --------
    geometries : list
        List of mp.Prism objects
    """
    geometries = []
    
    # Get points and triangles from Triangulation object
    points = np.column_stack((tri.x, tri.y))  # In pixel coordinates
    triangles = tri.triangles  # Get triangle indices
    
    for simplex in triangles:
        # Get the 3 vertices of the triangle
        vertices_2d = points[simplex]
        
        # Convert 2D triangle vertices to 3D prism vertices
        # Convert from pixel coordinates to mm
        vertices_3d = []
        for vertex in vertices_2d:
            # x_mm = vertex[0] / resolution - gridx_size_mm/2  # Center at origin
            # y_mm = vertex[1] / resolution - gridy_size_mm/2  # Center at origin
            x_mm = vertex[0] - gridx_size_mm/2  # Center at origin (Removed resolution)
            y_mm = vertex[1] - gridy_size_mm/2  # Center at origin (Removed resolution)
            vertices_3d.append(mp.Vector3(x_mm, y_mm, 0))
        
        # Create prism with these vertices
        prism = mp.Prism(
            vertices=vertices_3d,
            height=thickness,
            material=material #mp.Medium(epsilon=material_eps)
        )
        geometries.append(prism)
    
    return geometries

