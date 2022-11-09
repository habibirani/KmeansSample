import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans

sns.set()

columns = ['area_a', 'perimeter_p', 'compactness_c', 'length_k', 'width_k', 'asymmetry_coefficient', 'length_kernel_groove', 'class']
df = pd.read_csv("seeds_dataset.txt", sep = '\t', names = columns)
df.head()

data = df.iloc[:, 0:7] 

kmeans = KMeans(3)
kmeans.fit(data)

samples = data.values

varieties = list(data)
mergings = linkage( samples , method = 'complete')
dendrogram(mergings,
leaf_rotation=120)
plt.figure(figsize=(1440,14400))
plt.show()

identified_clusters = kmeans.fit_predict(data)

wcss = list()
number_clusters = range(1, 15)

for i in number_clusters:
    kmeans = KMeans(i)
    kmeans.fit(data)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

# each point is a separate cluster
print(wcss)

plt.plot(number_clusters, wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
