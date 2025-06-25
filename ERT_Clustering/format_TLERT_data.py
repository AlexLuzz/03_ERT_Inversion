import numpy as np
import pygimli.physics.ert as ert
import pandas as pd
from matplotlib.patches import Polygon

folder = "C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/Results/"

results = folder + "HRE" 
data = folder + "HRE.shm"

clustering_folder = "C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/Clustering/"

# Load the data
inv = ert.TimelapseERT(data)

inv.loadResults(results)

timestamps = inv.times
res = inv.models
mesh = inv.pd

# Prepare data for CSV
csv_data = {"date": timestamps}
for i, cell_resistivity in enumerate(res.T):  # Iterate over cells
    csv_data[f"{i}"] = cell_resistivity

# Prepare geometry data
geometry_data = []
for cid, cell in enumerate(mesh.cells()):

    n = cell.allNodes()  # Get nodes of the cell

    # Append data to list
    geometry_data.append({
        "cid": cid,
        "eid1": n[0].id(),
        "eid2": n[1].id(),
        "eid3": n[2].id(),
        "xy": [[n[0].x(), n[0].y()],
               [n[1].x(), n[1].y()],
               [n[2].x(), n[2].y()],
               [n[0].x(), n[0].y()]
               ]
    })

# Convert geometry data to DataFrame
df_geometry = pd.DataFrame(geometry_data)

# Create a DataFrame
df_res = pd.DataFrame(csv_data)

# Save to CSV
df_res.to_csv(clustering_folder + "res.csv", index=False)

# Save geometry data to CSV
df_geometry.to_csv(clustering_folder + "geometry.csv", index=False)
