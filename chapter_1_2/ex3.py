from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as sci_distance
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

folder = Path(__file__).resolve().parent
data = pd.read_csv(folder / 'Clusters.csv', sep=';')

columns = ['X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X21', 'X22', 'X23', 'X31', 'X32', 'X41', 'X42']
cdata = data[columns]

K = range(1, 20)
KM = [KMeans(n_clusters=k, random_state=111, n_init=10).fit(cdata) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [sci_distance.cdist(cdata, cent, 'euclidean') for cent in centroids]
dist = [np.min(D, axis=1) for D in D_k]
avgWithinSS = [sum(d) / cdata.shape[0] for d in dist]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(K), avgWithinSS, 'b*-')
ax.grid(True)
ax.set_xlabel('Количество кластеров')
ax.set_ylabel('Среднее расстояние до центра кластера')
ax.set_title('Метод локтя')
fig.tight_layout()
plt.show()

n_clusters = 5
means_cluster = KMeans(n_clusters=n_clusters, random_state=111, n_init=10)
data['cluster'] = means_cluster.fit_predict(cdata)

for c in range(n_clusters):
    cluster_members = data[data['cluster'] == c]
    print('Cluster{}(n={}):'.format(c, len(cluster_members)))

a = data.groupby('cluster')[columns].mean()
print(a)
a.to_excel(folder / 'Clusters_result.xlsx')

#Самостоятельная

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(cdata), columns=columns)
print(df_scaled.head())

linked = linkage(df_scaled, method='ward')
fig, ax = plt.subplots(figsize=(13, 6))
dendrogram(linked, ax=ax, leaf_rotation=90, leaf_font_size=7)
ax.set_title('Дендрограмма кластеризации')
ax.set_xlabel('Объекты')
ax.set_ylabel('Расстояние')
fig.tight_layout()
plt.show()
