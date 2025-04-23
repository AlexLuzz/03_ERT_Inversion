import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import matplotlib.pyplot as plt
from pygimli.physics import ert

# Define the coordinates for the square mesh
x_min, x_max = -3.1, 3.1
y_min, y_max = -1.85, 0

# Define the base filename
base_filename = "D:/01-Coding/01_BB_ERT/02_HRE/Data/11-20_18h_11-26_13h.shm"

# Load the data files
inv = ert.TimelapseERT(base_filename)

# Get the sensor positions
sensor_pos = inv.data.sensors()

# Convert sensor positions to a list of tuples
electrodes = [(pos.x(), pos.y(), pos.z()) for pos in sensor_pos]

# Create a grid
world = mt.createWorld(start=[x_min, y_min], end=[x_max, y_max], layers=[0], marker=0)

"""
road = [(x_min, 0), (0.5, 0), (0.5,-0.2), (x_min, -0.2)]
polygon_road = mt.createPolygon(road, isClosed=True, marker=1)
MG20 = [(x_min, -0.2), (0.5, -0.2), (0.5, -0.6), (x_min, -0.6)]
polygon_MG20 = mt.createPolygon(MG20, isClosed=True, marker=2)
IV = [(0.5, 0), (2, 0), (2, -0.85), (0.5, -1)]
polygon_IV = mt.createPolygon(IV, isClosed=True, marker=3)
water_table = [(x_min, -1.35), (x_max, -0.8), (x_max, y_min), (x_min, y_min)]
polygon_water_table = mt.createPolygon(water_table, isClosed=True, marker=4)

world += polygon_road + polygon_MG20 + polygon_IV + polygon_water_table
"""
# add additional nodes around sensor locations
for p in inv.data.sensors():
    world.createNode(p)
    world.createNode(p-0.01)
    world.createNode(p - 0.0075)
    world.createNode(p - 0.005)


# Create a mesh with the polygon zone
mesh = mt.createMesh(world, quality=34.7, area=0.01)

# Display the mesh
ax, cb = pg.show(mesh, markers=True)
ax.legend()
plt.show(block=True)

# Save the grid to a file
mesh_file = f"D:/01-Coding/01_BB_ERT/00_grid/mesh_high_densite.bms"
mesh.save(mesh_file)
print(f"Grid saved to {mesh_file}")