# K-Means Clustering using Iris Dataset

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 1 & 2. Load Iris dataset
iris = load_iris()

# 3. Extract input feature data
X = iris.data

# 4. Display number of samples and features
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# 5. Display feature names
print("\nFeature Names:")
for feature in iris.feature_names:
    print(feature)

# 6. Create K-Means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# 7. Train the K-Means model
kmeans.fit(X)

# 8. Obtain cluster labels
labels = kmeans.labels_

# 9. Display cluster assignment of each sample
print("\nCluster Assignment:")
for i, label in enumerate(labels):
    print("Sample", i + 1, "-> Cluster", label)

# 10. Display cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# 11. Count samples in each cluster
print("\nNumber of Samples in Each Cluster:")
for i in range(3):
    print("Cluster", i, ":", sum(labels == i))

# 12. Display K-Means inertia
print("\nK-Means Inertia:", kmeans.inertia_)

# 13. Create scatter plot
# Using petal length and petal width
plt.figure(figsize=(8, 6))

plt.scatter(
    X[:, 2],
    X[:, 3],
    c=labels,
    cmap="viridis",
    s=50
)

# 14. Plot cluster centers
plt.scatter(
    kmeans.cluster_centers_[:, 2],
    kmeans.cluster_centers_[:, 3],
    marker="X",
    s=200,
    color="red",
    label="Cluster Centers"
)

# 15. Add labels and title
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("K-Means Clustering of Iris Dataset")
plt.legend()

# 16. Display visualization
plt.show()