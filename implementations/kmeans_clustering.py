import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from helper import get_data

if __name__ == '__main__':
    x = get_data()

    wcss = []

    clusters_range = range(1, 11)
    for i in clusters_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(x)

        wcss.append(kmeans.inertia_)

    plt.plot(clusters_range, wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], c='red', label='Cluster 1')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], c='blue', label='Cluster 2')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], c='green', label='Cluster 3')
    plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], c='cyan', label='Cluster 4')
    plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', label='Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
