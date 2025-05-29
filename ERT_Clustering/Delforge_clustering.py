import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import sklearn
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_mutual_info_score

# Programming environment
print(f'Python {sys.version}')
for lib in [pd, np, mpl, sklearn]:
    print(f'{lib.__name__}:{lib.__version__}')

folder = "C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/Clustering/"

# Load resistivity data
resdf = pd.read_csv(folder+'ert_resistivity.csv', index_col=0, parse_dates=[0])
resdf.info()

# Load ERT model spatial grid
geometry = pd.read_csv(folder+'ert_geometry.csv', index_col=0)
geom = geometry['xy'].apply(lambda x: Polygon(np.array(eval(x))))

# Plot mean resistivities
fig = plt.figure()
ax = fig.gca()
ax.set_aspect('equal')

p = PatchCollection(geom, alpha=0.8)
colors = resdf.mean()
p.set_array(np.array(colors))
ax.add_collection(p)
ax.autoscale()
cbar = fig.colorbar(p, ax=ax)
ax.set_xlabel('x [$m$]')
ax.set_ylabel('z [$m$]')
cbar.ax.set_ylabel('$\\Omega.m$')

# Data z-standardization
zdf = (resdf - resdf.mean()) / resdf.std()

# Selecting the number of clusters
si_arr = []
for k in range(2, 11):
    clusters = AgglomerativeClustering(n_clusters=k).fit_predict(zdf.T)
    si = silhouette_score(zdf.T, clusters)
    si_arr.append(si)

fig, ax = plt.subplots()
ax.plot(range(2, 11), si_arr)
ax.set_xlabel('Number of cluster $k$')
ax.set_ylabel('SI [0-1]')

# Spatial distribution of clusters
clusters = AgglomerativeClustering(n_clusters=6).fit_predict(zdf.T)

fig, ax = plt.subplots()
p = PatchCollection(geom, cmap='tab20', alpha=0.8)
p.set_array(np.array(clusters))
ax.add_collection(p)

sample_silhouette_values = silhouette_samples(zdf.T, clusters)
outliers = sample_silhouette_values < 0
p2 = PatchCollection(geom[outliers], edgecolor='w', facecolor="None", alpha=0.8)
ax.add_collection(p2)

ax.set_xlabel("X [m]")
ax.set_ylabel("Z [m]")
ax.set_aspect('equal')
ax.autoscale()

# Cluster averaged time-series
ts = np.log10(resdf).groupby(clusters, axis=1).mean().resample('D').mean()
ax = ts.plot(cmap='tab20')
ax.set_ylabel('log-resistivity')

# Cluster statistics
fig, ax = plt.subplots()
mu = np.log10(resdf).mean()
sigma = np.log10(resdf).std()
sc = ax.scatter(mu, sigma, marker='.', alpha=0.8, c=clusters, cmap='tab20', edgecolor='None', facecolor='None')
ax.set_xlabel('$\\mu[log(\\\\rho)]$ [-]')
ax.set_ylabel('$\\sigma[log(\\\\rho)]$ [-]')

# Connectivity matrix
con_matrix = np.zeros((len(geom), len(geom)))
edge_df = geometry[['eid1', 'eid2', 'eid3']]

for i, edges in edge_df.iterrows():
    for e in edges:
        con_matrix[i] += (edge_df.values == e).any(axis=1)

np.fill_diagonal(con_matrix, 0)
con_matrix[con_matrix < 2] = 0
con_matrix[con_matrix > 1.] = 1.
print(f'number of connections: {int(con_matrix.sum())}')
plt.imshow(con_matrix, cmap='binary')

# Clustering with connectivity
clusters = AgglomerativeClustering(n_clusters=6, connectivity=con_matrix).fit_predict(zdf.T)

fig, ax = plt.subplots()
p = PatchCollection(geom, cmap='tab20', alpha=0.8)
p.set_array(np.array(clusters))
ax.add_collection(p)

sample_silhouette_values = silhouette_samples(zdf.T, clusters)
outliers = sample_silhouette_values < 0
p2 = PatchCollection(geom[outliers], edgecolor='w', facecolor="None", alpha=0.8)
ax.add_collection(p2)

ax.set_xlabel("X [m]")
ax.set_ylabel("Z [m]")
ax.set_aspect('equal')
ax.autoscale()

# Similarity of clustering partitions
clusters_1 = AgglomerativeClustering(n_clusters=6, connectivity=None).fit_predict(zdf.T)
clusters_2 = AgglomerativeClustering(n_clusters=6, connectivity=con_matrix).fit_predict(zdf.T)

ami = adjusted_mutual_info_score(clusters_1, clusters_2, average_method='max')
print(f'Adjusted mutual Information: {ami}')