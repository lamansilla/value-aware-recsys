# Value-Aware Product Recommendation by Customer Segmentation using a suitable High-Dimensional Similarity Measure

---

## Requirements

 Python ≥ 3.10
* Conda ≥ 23
* See `environment.yml` for all dependencies.

---

## Installation

Clone this repository and create the Conda environment with all required dependencies:

```bash
git clone https://github.com/lamansilla/value-aware-recsys
cd value-aware-recsys
conda env create -f environment.yml
conda activate recsys-env
```

---

## Data

To run experiments on real data, download the Online Retail II dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

Then, move the downloaded file into the folder:

```
experiments/exp2_online_retail/
```

### Preprocessing steps

1. Open and run `clean_data.ipynb` to preprocess and clean the dataset.
2. Then run `split_random.ipynb` to create the train/test random splits.

This will prepare the dataset for running the experiments.

---

## Running Experiments

All experiment scripts and configuration files are located in the `experiments/` directory.

### Available Experiments

* **Synthetic Data** (`run_exp1.py`): evaluate the recommendation system using simulated customer segments to test controlled behavioral patterns.

* **Real Data** (`run_exp2.py`): evaluate the system on the Online Retail II dataset to measure performance in real-world conditions.

Example usage:

```bash
python -m experiments.exp1_synthetic_data.run_exp1
python -m experiments.exp2_online_retail.run_exp2
```
