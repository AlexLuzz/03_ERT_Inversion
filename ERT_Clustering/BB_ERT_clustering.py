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


folder = "C:/Users/AQ96560/OneDrive - ETS/02 - Alexis Luzy/ERT_Data/Clustering/"

resdf = pd.read_csv(folder + 'res.csv', index_col=0, parse_dates=[0])

geometry = pd.read_csv(folder + "geometry.csv", index_col=0)
# parsing geometry into matplotlib polygons
geom = geometry['xy'].apply(lambda x: Polygon(np.array(eval(x))))

fig = plt.figure() # create figure
ax = fig.gca() # get figure axes
ax.set_aspect('equal') # use the same scale for both axis

p = PatchCollection(geom, alpha=0.8) # create the ERT grid
# set color
colors = resdf.mean() 
p.set_array(np.array(colors))
ax.add_collection(p)
ax.autoscale() # rescale axes
# add colorbar
cbar = fig.colorbar(p, ax=ax)
# add axes labels
_ = ax.set_xlabel('x [$m$]')
_ = ax.set_ylabel('z [$m$]')
_ = cbar.ax.set_ylabel('$\Omega.m$')

# Compute mean and std
mean = resdf.mean()
std = resdf.std()

# Replace 0 std with 1 to avoid NaN
std_replaced = std.replace(0, 1)

# Standardize
zdf = (resdf - mean) / std_replaced

si_arr = [] # array storing SI values

for k in range(2, 8):
    # perform the clustering
    clusters = AgglomerativeClustering(n_clusters=k).fit_predict(zdf.T) # Tranposed because variable need to be columns
    si = silhouette_score(zdf.T, clusters)
    si_arr.append(si)

fig, ax = plt.subplots()
ax.plot(range(2, 8), si_arr)
ax.set_xlabel('Number of cluster $k$')
ax.set_ylabel('SI [0-1]')

clusters = AgglomerativeClustering(n_clusters=4).fit_predict(zdf.T)

fig, ax = plt.subplots()
p = PatchCollection(geom, cmap='tab20', alpha=0.8)
p.set_array(np.array(clusters))
ax.add_collection(p)

sample_silhouette_values = silhouette_samples(zdf.T, clusters)
outliers = sample_silhouette_values < 0
p2 = PatchCollection(geom[outliers], edgecolor='w', facecolor="None",  alpha=0.8)
ax.add_collection(p2)

ax.set_xlabel("X [m]")
ax.set_ylabel("Z [m]")
ax.set_aspect('equal')
ax.autoscale()

# time-series
ts = np.log10(resdf).groupby(clusters, axis=1).mean().resample('D').mean() 
ax = ts.plot(cmap='tab20')
ax.set_ylabel('log-resistivity')

fig, ax = plt.subplots()
mu = np.log10(resdf).mean()
sigma = np.log10(resdf).std()
sc = ax.scatter(mu, sigma, marker='.', alpha=0.8, c=clusters, cmap='tab20', edgecolor='None', facecolor='None')
ax.set_xlabel('$\mu[log(\\rho)]$ [-]')
ax.set_ylabel('$\sigma[log(\\rho)]$ [-]')

# building connectivity matrix
con_matrix = np.zeros((len(geom), len(geom)))
edge_df = geometry[['eid1', 'eid2', 'eid3']]

for i, edges in edge_df.iterrows():
    for e in edges:
        con_matrix[i] += (edge_df.values==e).any(axis=1)

# remove cells connected to itself
np.fill_diagonal(con_matrix, 0)
con_matrix[con_matrix < 2] = 0
con_matrix[con_matrix > 1.] = 1.
print(f'number of connections: {int(con_matrix.sum())}')
plt.imshow(con_matrix, cmap='binary')

clusters = AgglomerativeClustering(n_clusters=6, connectivity=con_matrix).fit_predict(zdf.T)

fig, ax = plt.subplots()
p = PatchCollection(geom, cmap='tab20', alpha=0.8)
p.set_array(np.array(clusters))
ax.add_collection(p)


sample_silhouette_values = silhouette_samples(zdf.T, clusters)
outliers = sample_silhouette_values < 0
p2 = PatchCollection(geom[outliers], edgecolor='w', facecolor="None",  alpha=0.8)
ax.add_collection(p2)

ax.set_xlabel("X [m]")
ax.set_ylabel("Z [m]")
ax.set_aspect('equal')
ax.autoscale()

clusters_1 = AgglomerativeClustering(n_clusters=6, connectivity=None).fit_predict(zdf.T)
clusters_2 = AgglomerativeClustering(n_clusters=6, connectivity=con_matrix).fit_predict(zdf.T)

ami = adjusted_mutual_info_score(clusters_1, clusters_2, average_method='max')

print(f'Adjusted mutual Information: {ami}')

plt.show(block=True)