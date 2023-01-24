"""
Definitions of basic kmeans clustering methods
"""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from constants import KMEANS_RANDOM_STATE
from yellowbrick.cluster import KElbowVisualizer


def mbkmeans_clusters(X, y=None, k_range=10, mb=1000):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        y: Array-like object of weights assigned to each sample.
        k_range: single value of k or a range of k values where, k being the number of clusters.
        mb: Size of mini-batches.

    Returns:
        Trained clustering model and labels based on X.
    """

    if isinstance(k_range, range):

        sum_of_squared_distance = []
        silhouette_values = []

        for k in k_range:
            mbkm = MiniBatchKMeans(n_clusters=k, batch_size=mb, random_state=KMEANS_RANDOM_STATE)
            if y:
                km = mbkm.fit(X, sample_weight=y)
            else:
                km = mbkm.fit(X)
            sum_of_squared_distance.append(km.inertia_)
            silhouette_values.append(silhouette_score(X, km.labels_))

        plt.plot(k_range, silhouette_values, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhouette')
        plt.title('Silhouette as a function of k')
        plt.show()

    elif isinstance(k_range, int):

        mbkm = MiniBatchKMeans(n_clusters=k_range, batch_size=mb, random_state=KMEANS_RANDOM_STATE)
        if y:
            km = mbkm.fit(X, sample_weight=y)
        else:
            km = mbkm.fit(X)
        print('Silhouette score:')
        print(silhouette_score(X, km.labels_))

        return km, km.labels_


def plot_elbow(X, y=None, k_range=range(2, 10)):
    """Plot elbow diagram with optimal K value for K in k_range

       Args:
           X: Matrix of features.
           y: Array-like object of weights assigned to each sample.
           k_range: A range of k values where, k being the number of clusters.

       """
    model = KMeans(random_state=KMEANS_RANDOM_STATE)
    visualizer = KElbowVisualizer(model, k=k_range, timings=False)
    if y:
        visualizer.fit(X, y)
    else:
        visualizer.fit(X)
    visualizer.finalize()
    print("Optimal Value of K: " + str(visualizer.elbow_value_))
    print("Corresponding Distortion Score: " + str(visualizer.elbow_score_))
