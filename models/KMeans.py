from matplotlib import pyplot as plt
import numpy as np


class KMeans:

    def __init__(self, df, class_column, columns_ignored=-1, k = 3):
     
        df_copy = df.copy()
        self.df = {}
        self.columns = list(df_copy.columns[: columns_ignored])
        self.class_column = class_column

        self.k = k
        self.centroids = None

        self.metrics = {
            'fscore': 0,
            'kappa': 0,
            'matthews': 0
        }

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((data_point - centroids) ** 2, axis=1))

    def fit(self, X, max_epochs=200):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_epochs):
            y = []

            for data_point in X:
                distances = KMeans.euclidean_distance(data_point, self.centroids)
                y.append(np.argmin(distances))

            y = np.array(y)

            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))
            
            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
            
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)
        return y
    
'''
random_points = np.random.randint(0, 100, size=(100, 2))

kmeans = KMeans(k=3)
labels = kmeans.fit(random_points)

print(random_points)
print(kmeans.centroids)
plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), marker='*', s=200)

plt.show()
'''