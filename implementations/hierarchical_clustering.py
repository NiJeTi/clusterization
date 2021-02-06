import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

from helper import get_data

if __name__ == '__main__':
    x = get_data()

    dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))

    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()

    hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(x)

    plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], c='red', label='Cluster 1')
    plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], c='blue', label='Cluster 2')
    plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], c='green', label='Cluster 3')
    plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], c='cyan', label='Cluster 4')
    plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], c='magenta', label='Cluster 5')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
