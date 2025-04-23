import pygimli as pg
import pygimli.meshtools as mt
import matplotlib.pyplot as plt
from pygimli.physics import ert
import numpy as np

# Define the base filename
base_filename = "01_BB_ERT/02_HRE/Data/11-20_18h_11-26_13h.shm"

# Load the data files
inv = ert.TimelapseERT(base_filename)

# Get the sensor positions
sensor_pos = inv.data.sensors()

# Convert sensor positions to a list of tuples
electrodes = [(pos.x(), pos.y(), pos.z()) for pos in sensor_pos]

# Find the leftmost, rightmost, deepest, and highest electrode positions
leftmost = min(electrodes, key=lambda pos: pos[0])
rightmost = max(electrodes, key=lambda pos: pos[0])
deepest = min(electrodes, key=lambda pos: pos[2])
highest = max(electrodes, key=lambda pos: pos[2])

# Parameters for creating the grid
square_size = 0.1  # Dimensions of a square of the grid
space = 2.0  # Space around the grid

def createGrid(electrodes, show=True, square_size=0.1, space=2.0):
    """
    Create a grid around the electrode positions.

    :param electrodes: List of electrode positions (x, y, z).
    :param show: Display grid on SciView window.
    :param square_size: Dimensions of a square of the grid.
    :param space: Space around the grid (space = 6 : 6 tiles around the electrodes).
    :return: The created grid.
    """
    ex = np.array([pos[0] for pos in electrodes])  # Extract x-coordinates
    ey = np.array([pos[1] for pos in electrodes])  # Extract y-coordinates

    xmin, xmax = min(ex) - space * square_size, max(ex) + space * square_size
    ymin, ymax = min(ey) - space * square_size, max(ey) + space * square_size

    x = np.arange(xmin, xmax + .001, square_size)
    y = np.arange(ymin, ymax + .001, square_size)

    grid = mt.createGrid(x, y, marker=5)

    if show:
        ax, cb = pg.show(grid)
        ax.plot(ex, ey, "mx")
        plt.show()

    return grid

if __name__ == "__main__":
    # Create the grid
    grid = createGrid(electrodes, show=True, square_size=square_size, space=space)

    # Save the grid to a file
    grid_file = f"01_BB_ERT/00_grid/grid_sq_{square_size}sp{space}.bms"
    grid.save(grid_file)
    print(f"Grid saved to {grid_file}")