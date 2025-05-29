import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import matplotlib.pyplot as plt
from pygimli.physics import ert

def ERT_BB_grid():
    """
    Creates a pos array with 64 electrodes positions in an 8x8 grid.
    Horizontal spacing: 0.85 meters
    Vertical spacing: 0.2 meters
    """
    n_columns = 8
    n_lines = 8
    h_spacing = 0.85
    v_spacing = 0.2
    start_depth = 0.0  # Depth of the first row of electrodes

    # Initialize the pos array
    pos = np.zeros((n_columns * n_lines, 2))

    # Calculate positions for each electrode
    for col in range(n_columns):
        for line in range(n_lines):
            x = round(h_spacing * col-h_spacing*3.5, 3)  # Horizontal position
            y = round(start_depth - line * v_spacing, 3)  # Vertical position
            pos[col * n_lines + line] = [x, y]

    return pos  # Return the calculated positions

scheme = ert.createData(elecs=ERT_BB_grid(), schemeName="uk")

# Define the coordinates for the square mesh
x_min, x_max = -3.1, 3.1
y_min, y_max = -1.85, 0

# Create a grid
world = mt.createWorld(start=[x_min, y_min], end=[x_max, y_max], layers=[0], marker=0)

# add additional nodes around sensor locations
for p in scheme.sensors():
    world.createNode(p)
    world.createNode(p-0.01)

# Create a mesh with the polygon zone
mesh = mt.createMesh(world, quality=31, area=0.1, areaMax=0.3)

# Display the mesh
ax, cb = pg.show(mesh, markers=True)
ax.legend()
plt.show(block=True)

folder = "C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/"

# Save the grid to a file
mesh_file = folder + "Grids/BB_grid_coarse.bms"
mesh.save(mesh_file)
print(mesh)