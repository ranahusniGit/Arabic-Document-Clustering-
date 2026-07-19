# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:34:23 2026

@author: User
"""


import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralCoclustering
from sklearn.preprocessing import normalize, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    f1_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    v_measure_score,
)

from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


INPUT_FOLDER = Path(r"C:\clustering outputs\all_embedding_outputs")
OUTPUT_FOLDER = Path(r"C:\Users\User\clustering_evaluation_outputs_FIXED")
OUTPUT_FOLDER.mkdir(exist_ok=True)

N_RUNS = 10
RANDOM_SEEDS = list(range(1, N_RUNS + 1))

ALGORITHMS = [
    "kmeansplusplus",
    "spherical_kmeans",
    "hac",
    "spectral_coclustering"
]


def spherical_kmeans_fit_predict(X, n_clusters, random_state=42, max_iter=100):
    rng = np.random.default_rng(random_state)
    X_norm = normalize(X, norm="l2")
    n_samples = X_norm.shape[0]

    init_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X_norm[init_idx]

    labels = np.full(n_samples, -1, dtype=int)

    for _ in range(max_iter):
        old_labels = labels.copy()

        similarities = X_norm @ centroids.T
        labels = np.argmax(similarities, axis=1)

        new_centroids = []

        for k in range(n_clusters):
            points = X_norm[labels == k]

            if len(points) == 0:
                centroid = X_norm[rng.integers(0, n_samples)]
            else:
                centroid = points.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

            new_centroids.append(centroid)

        centroids = np.vstack(new_centroids)

        if np.array_equal(labels, old_labels):
            break

    return labels


def build_contingency_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    matrix = np.zeros((len(pred_labels), len(true_labels)), dtype=np.int64)

    for i, pred in enumerate(pred_labels):
        for j, true in enumerate(true_labels):
            matrix[i, j] = np.sum((y_pred == pred) & (y_true == true))

    return matrix, pred_labels, true_labels


def map_clusters_to_labels(y_true, y_pred):
    matrix, pred_labels, true_labels = build_contingency_matrix(y_true, y_pred)

    row_ind, col_ind = linear_sum_assignment(-matrix)

    mapping = {
        pred_labels[row]: true_labels[col]
        for row, col in zip(row_ind, col_ind)
    }

    return np.array([mapping[label] for label in y_pred])


def clustering_accuracy(y_true, y_pred):
    matrix, _, _ = build_contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-matrix)
    return matrix[row_ind, col_ind].sum() / len(y_true)


def purity_score(y_true, y_pred):
    total = 0

    for cluster in np.unique(y_pred):
        idx = y_pred == cluster
        _, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()

    return total / len(y_true)


def entropy_score(y_true, y_pred, normalize_entropy=True):
    total_entropy = 0.0
    n = len(y_true)

    for cluster in np.unique(y_pred):
        idx = y_pred == cluster
        labels = y_true[idx]

        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()

        cluster_entropy = -np.sum(probs * np.log2(probs + 1e-12))
        total_entropy += (len(labels) / n) * cluster_entropy

    if normalize_entropy:
        max_entropy = np.log2(len(np.unique(y_true)))

        if max_entropy > 0:
            total_entropy = total_entropy / max_entropy

    return total_entropy


def evaluate_clustering(y_true, y_pred):
    y_pred_mapped = map_clusters_to_labels(y_true, y_pred)

    return {
        "F1_macro": f1_score(y_true, y_pred_mapped, average="macro"),
        "F1_weighted": f1_score(y_true, y_pred_mapped, average="weighted"),
        "Purity": purity_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "ARI": adjusted_rand_score(y_true, y_pred),
        "Entropy": entropy_score(y_true, y_pred, normalize_entropy=True),
        "Accuracy": clustering_accuracy(y_true, y_pred),
        "V_measure": v_measure_score(y_true, y_pred),
    }


def get_column(df, possible_names):
    for col in possible_names:
        if col in df.columns:
            return df[col]

    raise ValueError(f"None of these columns found: {possible_names}")


def make_cluster_output(df, y_pred, seed, algorithm_name):
    return pd.DataFrame({
        "text": get_column(df, ["text", "document"]),
        "ProcessedDocument": get_column(df, ["ProcessedDocument", "processed_document"]),
        "id": df["id"],
        "label": df["label"],
        "cluster": y_pred,
        "run": seed,
        "algorithm": algorithm_name
    })


def add_result(all_results, embedding_file, algorithm_name, seed, n_clusters, metrics):
    row = {
        "file_name": embedding_file.name,
        "dataset_embedding": embedding_file.stem,
        "algorithm": algorithm_name,
        "run": seed,
        "n_clusters": n_clusters,
    }

    row.update(metrics)
    all_results.append(row)


embedding_files = list(INPUT_FOLDER.glob("*.csv"))
print(f"Found {len(embedding_files)} embedding files")

all_run_results = []


for embedding_file in embedding_files:

    print("\n====================================")
    print(f"Processing: {embedding_file.name}")
    print("====================================")

    df = pd.read_csv(embedding_file)

    if "label" not in df.columns:
        print("No label column found. Skipping.")
        continue

    embedding_columns = [col for col in df.columns if col.startswith("emb_")]

    if len(embedding_columns) == 0:
        print("No embedding columns found. Skipping.")
        continue

    X = df[embedding_columns].values.astype(np.float32)
    X_normalized = normalize(X, norm="l2")

    # Spectral Co-Clustering requires a non-negative input matrix.
    # Feature-wise Min-Max scaling maps every embedding dimension to [0, 1].
    X_spectral = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

    y_true = LabelEncoder().fit_transform(df["label"].astype(str))
    n_clusters = len(np.unique(y_true))

    print(f"Number of clusters: {n_clusters}")

    file_cluster_outputs = {alg: [] for alg in ALGORITHMS}

    for seed in RANDOM_SEEDS:

        print(f"Run {seed} | kmeansplusplus")

        y_pred = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            random_state=seed,
            n_init=10
        ).fit_predict(X)

        metrics = evaluate_clustering(y_true, y_pred)
        add_result(all_run_results, embedding_file, "kmeansplusplus", seed, n_clusters, metrics)

        file_cluster_outputs["kmeansplusplus"].append(
            make_cluster_output(df, y_pred, seed, "kmeansplusplus")
        )

        print(f"Run {seed} | spherical_kmeans")

        y_pred = spherical_kmeans_fit_predict(
            X,
            n_clusters=n_clusters,
            random_state=seed,
            max_iter=100
        )

        metrics = evaluate_clustering(y_true, y_pred)
        add_result(all_run_results, embedding_file, "spherical_kmeans", seed, n_clusters, metrics)

        file_cluster_outputs["spherical_kmeans"].append(
            make_cluster_output(df, y_pred, seed, "spherical_kmeans")
        )

        print(f"Run {seed} | hac")

        y_pred = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward"
        ).fit_predict(X)

        metrics = evaluate_clustering(y_true, y_pred)
        add_result(all_run_results, embedding_file, "hac", seed, n_clusters, metrics)

        file_cluster_outputs["hac"].append(
            make_cluster_output(df, y_pred, seed, "hac")
        )

        print(f"Run {seed} | spectral_coclustering")

        model = SpectralCoclustering(
            n_clusters=n_clusters,
            random_state=seed
        )

        # Use the non-negative representation required by Spectral Co-Clustering.
        model.fit(X_spectral)
        y_pred = np.asarray(model.row_labels_)

        metrics = evaluate_clustering(y_true, y_pred)
        add_result(all_run_results, embedding_file, "spectral_coclustering", seed, n_clusters, metrics)

        file_cluster_outputs["spectral_coclustering"].append(
            make_cluster_output(df, y_pred, seed, "spectral_coclustering")
        )

    for algorithm_name, outputs in file_cluster_outputs.items():

        if len(outputs) == 0:
            continue

        output_file = OUTPUT_FOLDER / f"{embedding_file.stem}_{algorithm_name}_all_runs.csv"

        pd.concat(outputs, ignore_index=True).to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Saved: {output_file}")


all_results_df = pd.DataFrame(all_run_results)

all_results_path = OUTPUT_FOLDER / "all_clustering_run_results.csv"

all_results_df.to_csv(
    all_results_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSaved all run results: {all_results_path}")


metric_columns = [
    "F1_macro",
    "F1_weighted",
    "Purity",
    "NMI",
    "ARI",
    "Entropy",
    "Accuracy",
    "V_measure",
]

if len(all_results_df) > 0:

    summary_df = (
        all_results_df
        .groupby(["dataset_embedding", "algorithm"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_df.columns = [
        "_".join(col).strip("_")
        for col in summary_df.columns.values
    ]

    summary_path = OUTPUT_FOLDER / "summary_mean_std_results.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved summary mean/std results: {summary_path}")

    ranking_df = summary_df.copy()

    ranking_metrics = [
        "F1_macro_mean",
        "F1_weighted_mean",
        "Purity_mean",
        "NMI_mean",
        "ARI_mean",
        "Accuracy_mean",
        "V_measure_mean",
    ]

    for metric in ranking_metrics:
        ranking_df[f"{metric}_rank"] = rankdata(
            -ranking_df[metric],
            method="min"
        )

    ranking_df["Entropy_mean_rank"] = rankdata(
        ranking_df["Entropy_mean"],
        method="min"
    )

    rank_columns = [col for col in ranking_df.columns if col.endswith("_rank")]

    ranking_df["Total_Rank"] = ranking_df[rank_columns].sum(axis=1)
    ranking_df = ranking_df.sort_values("Total_Rank")

    ranking_path = OUTPUT_FOLDER / "ranking_summary_results.csv"

    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    print(f"Saved ranking summary: {ranking_path}")

    statistical_ready_path = OUTPUT_FOLDER / "statistical_testing_ready_results.csv"

    all_results_df.to_csv(
        statistical_ready_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved statistical testing ready file: {statistical_ready_path}")

else:
    print("No clustering results were generated.")

print("\nAll clustering evaluation completed successfully.")
