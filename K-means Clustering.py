# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generating sample data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.05, random_state=42)

# Visualizing the dataset before clustering
plt.scatter(X[:, 0], X[:, 1], s=50, color='blue', alpha=0.5)
plt.title("Dataset Before Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Applying K-means clustering
k = 4  # Number of clusters
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the clustered data
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, color='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, color='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, color='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, color='purple', label='Cluster 4')

# Plotting cluster centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, color='yellow', marker='X', label='Centroids')

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
