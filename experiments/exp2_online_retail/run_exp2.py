import os
import pickle

import numpy as np
from tqdm import tqdm

from src.recom import cluster_customers, recommend_items


def run_simulation(config, S_train, S_test, scenario_dir):

    results = []
    data_dir = os.path.join(scenario_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "S_train.npy"), S_train)
    np.save(os.path.join(data_dir, "S_test.npy"), S_test)

    for metric in tqdm(config["distance_metrics"], desc="Distance Metrics"):

        # Clustering
        cluster_labels, silhouette, optimal_k, medoids_indices, medoids_data = (
            cluster_customers(S_train, distance_metric=metric)
        )

        for method in tqdm(config["methods"], desc="Methods"):
            # Recommendation
            user_recommendations, cluster_stats_ = recommend_items(
                S_train, cluster_labels, top_L=None, method=method
            )

            with open(
                os.path.join(data_dir, f"recommendations_{metric}_{method}.pkl"),
                "wb",
            ) as f:
                pickle.dump(user_recommendations, f)

            with open(
                os.path.join(data_dir, f"cluster_stats_{metric}_{method}.pkl"),
                "wb",
            ) as f:
                pickle.dump(cluster_stats_, f)

            results.append(
                {
                    "cluster_labels": cluster_labels,
                    "distance_metric": metric,
                    "medoids_indices": medoids_indices,
                    "medoids_data": medoids_data,
                    "method": method,
                    "optimal_k": optimal_k,
                    "silhouette": silhouette,
                }
            )

    return results


top_L = 10
num_splits = 5
split_method = "random"

config = {
    "top_L": top_L,
    "distance_metrics": ["madd-euclidean", "jaccard", "cosine", "euclidean"],
    "methods": ["revenue", "popularity", "expected_profit"],
    "random_seed": 42,
}

exp_dir = "experiments/exp2_online_retail_final"
output_dir = os.path.join(exp_dir, f"results/splits_{split_method}")
os.makedirs(output_dir, exist_ok=True)

print("\nExperiment Configuration:")
for k, v in config.items():
    print(f"\t{k}: {v}")

for split in range(1, num_splits + 1):
    print(f"\n--- Split {split}/{num_splits} ---")

    data_dir = os.path.join(exp_dir, f"splits_{split_method}", f"split_{split:02d}")
    results_dir = os.path.join(output_dir, f"split_{split:02d}")
    os.makedirs(results_dir, exist_ok=True)

    S_train = np.load(os.path.join(data_dir, f"S_train.npy")).astype(float)
    S_test = np.load(os.path.join(data_dir, f"S_test.npy")).astype(float)

    results = run_simulation(config, S_train, S_test, results_dir)

    results_file = os.path.join(results_dir, "results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
