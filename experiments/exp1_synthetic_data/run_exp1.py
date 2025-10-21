import os
import pickle

import numpy as np
from tqdm import tqdm

from src.data import generate_data, generate_masked_data
from src.recom import cluster_customers, recommend_items


def run_simulation(config, scenario_dir):

    results = []
    base_seed = config.get("random_seed", 42)

    data_dir = os.path.join(scenario_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    for rep in tqdm(range(config["n_repeats"]), desc=f"Escenario {config['scenario']}"):
        current_seed = base_seed + rep
        np.random.seed(current_seed)

        # Generate the data
        data, true_labels = generate_data(
            scenario=config["scenario"],
            theta=config["theta"],
            n_customers=config["n_customers"],
            n_products=config["n_products"],
            max_expenditure=config["max_expenditure"],
            random_seed=current_seed,
        )

        # Mask to simulate training and test sets
        S_train, S_test, _ = generate_masked_data(data, beta=config["beta"])

        rep_dir = os.path.join(data_dir, f"rep_{rep}")
        os.makedirs(rep_dir, exist_ok=True)

        np.save(os.path.join(rep_dir, "data.npy"), data)
        np.save(os.path.join(rep_dir, "S_train.npy"), S_train)
        np.save(os.path.join(rep_dir, "S_test.npy"), S_test)
        np.save(os.path.join(rep_dir, "true_labels.npy"), true_labels)

        for metric in config["distance_metrics"]:

            # Clustering
            cluster_labels, silhouette, optimal_k, medoids_indices, medoids_data = (
                cluster_customers(S_train, distance_metric=metric)
            )

            for method in config["methods"]:
                # Recommendation
                user_recommendations, cluster_stats_ = recommend_items(
                    S_train, cluster_labels, top_L=None, method=method
                )

                with open(
                    os.path.join(rep_dir, f"recommendations_{metric}_{method}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(user_recommendations, f)

                with open(
                    os.path.join(rep_dir, f"cluster_stats_{metric}_{method}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(cluster_stats_, f)

                results.append(
                    {
                        "rep": rep,
                        "true_labels": true_labels,
                        "cluster_labels": cluster_labels,
                        "medoids_indices": medoids_indices,
                        "medoids_data": medoids_data,
                        "distance_metric": metric,
                        "method": method,
                        "optimal_k": optimal_k,
                        "silhouette": silhouette,
                    }
                )

    return results


# Experiment parameters
theta_range = (0.85, 0.95)
beta = 0.99

top_L = 10
n_repeats = 50

config = {
    "n_customers": 150,
    "n_products": 1500,
    "max_expenditure": 10000,
    "beta": beta,
    "theta": theta_range,
    "top_L": top_L,
    "distance_metrics": ["madd", "jaccard", "cosine", "euclidean"],
    "methods": ["revenue", "popularity", "expected_profit"],
    "n_repeats": n_repeats,
    "random_seed": 42,
}

output_dir = "experiments/exp1_synthetic_data/results"
os.makedirs(output_dir, exist_ok=True)

print("\nExperiment Configuration:")
for k, v in config.items():
    print(f"\t{k}: {v}")

for scenario in ["I", "II", "III"]:
    config["scenario"] = scenario

    scenario_dir = os.path.join(
        output_dir,
        f"scenario_{scenario}_t={theta_range[0]}-{theta_range[1]}_b={beta}",
    )
    os.makedirs(scenario_dir, exist_ok=True)

    results = run_simulation(config, scenario_dir)

    results_file = os.path.join(scenario_dir, "results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved in: {results_file}")
