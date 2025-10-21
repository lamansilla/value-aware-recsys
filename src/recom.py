import kmedoids
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances as cosine_distance
from sklearn.metrics.pairwise import euclidean_distances as euclidean_distance

from src.distances import jaccard_distance, madd_distance


def cluster_customers(S_train, distance_metric="madd", n_clusters=None):
    """
    Segments customers using the k-medoids algorithm with the specified distance metric,
    and optionally determines the optimal number of clusters using the silhouette score.

    Parameters
    ----------
    S_train : numpy.ndarray
        Feature matrix of shape (n_users, n_products), where each cell contains
        the normalized revenue share of the product for that user.
        Zero values indicate products not purchased.

    distance_metric : str, optional (default="madd")
        Distance metric to use:
        - "madd": Mean Absolute Difference of Distances (recommended for high dimensionality).
        - "euclidean": Traditional Euclidean distance.
        - "cosine": Cosine similarity.
        - "jaccard": Jaccard distance for binary data.

    n_clusters : int, optional (default=None)
        Fixed number of clusters to use. If None, it is automatically determined between
        2 and min(10, n_users - 1).

    Returns
    -------
    tuple (cluster_labels, silhouette, optimal_k, medoids_indices, medoids_data):
        cluster_labels: array with the cluster assignment for each user.
        silhouette: Silhouette coefficient of the clustering.
        optimal_k: Number of clusters used.
        medoids_indices: List of indices of the users that are the medoids.
        medoids_data: Matrix with the actual feature vectors of the medoids.
    """

    # Validation of distance metric
    valid_metrics = ["madd", "euclidean", "cosine", "jaccard"]
    if distance_metric not in valid_metrics:
        raise ValueError(
            f"Distance metric '{distance_metric}' is not valid. Options: {valid_metrics}"
        )

    # Compute distance matrix
    if distance_metric == "madd":
        distance_matrix = madd_distance(S_train)
    elif distance_metric == "euclidean":
        distance_matrix = euclidean_distance(S_train)
    elif distance_metric == "cosine":
        distance_matrix = cosine_distance(S_train)
    else:  # jaccard
        distance_matrix = jaccard_distance(S_train)

    # Define maximum number of possible clusters if not provided
    if n_clusters is None:
        max_clusters = min(10, len(S_train) - 1)

        # Ensure at least 2 clusters are possible
        if max_clusters < 2:
            optimal_k = 1
        else:
            silhouette_scores = []
            k_values = range(2, max_clusters + 1)

            for k in k_values:
                result = kmedoids.pam(distance_matrix, k)
                labels = result.labels

                if len(np.unique(labels)) > 1:  # Avoid single-cluster solutions
                    s = silhouette_score(distance_matrix, labels, metric="precomputed")
                    silhouette_scores.append(s)
                else:
                    silhouette_scores.append(-1)  # Worst case scenario

            optimal_k = k_values[np.argmax(silhouette_scores)]
    else:
        optimal_k = n_clusters

    # Final clustering with optimal_k
    result = kmedoids.pam(distance_matrix, optimal_k)
    cluster_labels = result.labels
    silhouette = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")

    medoids_indices = result.medoids
    medoids_data = S_train[medoids_indices]

    return cluster_labels, silhouette, optimal_k, medoids_indices, medoids_data


def recommend_items(S_train, cluster_labels, method="revenue", top_L=None):
    """
    Generates personalized recommendations for each user, excluding products already purchased
    and selecting the top-L products according to the specified ranking method.

    Parameters
    ----------
    S_train : numpy.ndarray
        Training matrix of shape (n_users, n_products) where each cell contains
        the normalized revenue share of the product for that user.
        Zero values indicate products not purchased.

    cluster_labels : numpy.ndarray
        Array of shape (n_users,) that assigns each user to a cluster.

    method : str, optional (default="revenue")
        Method for ranking products:
        - "popularity": Based on purchase frequency in the cluster.
        - "revenue": Based on total revenue generated in the cluster.
        - "expected_profit": Combination of popularity and revenue.

    top_L : int, optional (default=None)
        Maximum number of recommendations to generate per user. If None, all possible recommendations will be generated.

    Returns
    -------
    tuple (user_recommendations, cluster_stats):
        user_recommendations: dict {user_idx: [item1, item2,...]}, filtered recommendations
          excluding products already purchased.
        cluster_stats: dict {cluster_id: {"item_scores": array, "item_revenue": array, "cluster_size": int}},
          statistics by cluster.
    """

    if top_L is None:
        top_L = S_train.shape[1]

    user_recommendations = {}
    cluster_stats = {}
    unique_clusters = np.unique(cluster_labels)

    # Compute scores by cluster
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_data = S_train[cluster_mask]

        # Normalization
        total_revenue_cluster = cluster_data.sum()
        cluster_data = cluster_data / total_revenue_cluster

        if method == "popularity":
            item_scores = np.mean(cluster_data > 0, axis=0)
        elif method == "revenue":
            item_scores = np.sum(cluster_data, axis=0)
        elif method == "expected_profit":
            freq = np.mean(cluster_data > 0, axis=0)
            revenue = np.sum(cluster_data, axis=0)
            item_scores = freq * revenue

        else:
            raise ValueError(
                f"Method '{method}' not valid. Options: 'popularity', 'revenue', 'expected_profit'"
            )

        item_revenue = np.sum(cluster_data, axis=0)

        cluster_stats[cluster] = {
            "item_scores": item_scores,
            "item_revenue": item_revenue,
            "cluster_size": np.sum(cluster_mask),
        }

    # Generate recommendations for each user
    for user_id in range(S_train.shape[0]):
        cluster = cluster_labels[user_id]
        purchased = np.where(S_train[user_id] > 0)[0]

        # Sort products by descending score
        all_recs = np.argsort(cluster_stats[cluster]["item_scores"])[::-1]

        # Filter out already purchased items and take the top-L
        valid_recs = [item for item in all_recs if item not in purchased]
        user_recommendations[user_id] = valid_recs[:top_L]

    return user_recommendations, cluster_stats
