import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import shap

from uadr.models import probabilistic_NN as pnn
from uadr.utils.data_preprocessing import get_preprocessed_features

SEED = 42
np.random.seed(SEED)

CURRENT_SPLIT = 0
SCALING_MODE = "drug_z_norm"

CHECKPOINT_DIR = "model_checkpoints" + "_" + SCALING_MODE
CHECKPOINT_FILENAME = f"split_{CURRENT_SPLIT}_best.ckpt"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

SHAP_OUTPUT_DIR = "shap_partial_results" + "_" + SCALING_MODE
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

MAX_INSTANCES_SHAP = None
BATCH_SIZE = 200
NSAMPLES = 1100
N_CLUSTERS = 25

def load_data(base_path: str, current_split: int, data_dir="../data"):
    return get_preprocessed_features(
        current_split=current_split,
        base_path=base_path,
        cross_validation_type="cell_lines_cold_start",
        scaling_mode=SCALING_MODE,
        n_bits_drug_fingerprints=128,
        data_dir=data_dir,
    )


def load_model(model_kwargs: dict):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    model = pnn.ProbabilisticFeedForwardNetwork.load_from_checkpoint(CHECKPOINT_PATH, **model_kwargs)
    model.eval()
    model.to("cpu")
    return model


def _save_batch_csv(arr: np.ndarray, path: str, feature_names: list):
    df = pd.DataFrame(arr, columns=feature_names)
    df.to_csv(path, index=False)


def compute_shap_resumable(
    model,
    x_train,
    x_test,
    y_test,
    gene_mask,
    max_instances=None,
    batch_size=BATCH_SIZE,
    nsamples=NSAMPLES,
    n_clusters=N_CLUSTERS,
    seed=SEED,
):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(x_train)
    background_data = kmeans.cluster_centers_

    explainer_mean = shap.KernelExplainer(lambda x: model.predict_target(x).ravel(), background_data, link="identity")
    explainer_unc = shap.KernelExplainer(lambda x: model.predict_uncertainty(x).ravel(), background_data, link="identity")

    n_samples_total = x_test.shape[0]
    rng = np.random.default_rng(seed)

    progress_path = os.path.join(SHAP_OUTPUT_DIR, "progress.pkl")
    if os.path.exists(progress_path):
        with open(progress_path, "rb") as f:
            progress = pickle.load(f)
        done_batches = progress.get("done_batches", [])
        saved_sel_idx = progress.get("sel_idx", None)
        if saved_sel_idx is not None:
            sel_idx = np.array(saved_sel_idx, dtype=int)
        else:
            sel_idx = np.arange(n_samples_total)
        rng_state = progress.get("rng_state", None)
        if rng_state is not None:
            try:
                rng.bit_generator.state = rng_state
            except Exception:
                print("Warning: could not restore RNG state from progress file.")
    else:
        if (max_instances is None) or (n_samples_total <= max_instances):
            sel_idx = np.arange(n_samples_total)
        else:
            sel_idx = rng.choice(np.arange(n_samples_total), max_instances, replace=False)
        pd.DataFrame({"sel_idx": sel_idx}).to_csv(os.path.join(SHAP_OUTPUT_DIR, "sel_idx.csv"), index=False)
        done_batches = []
        with open(progress_path, "wb") as f:
            pickle.dump({"done_batches": done_batches, "sel_idx": sel_idx.tolist(), "rng_state": rng.bit_generator.state}, f)

    X_sub = x_test[sel_idx]
    meta_sub = y_test.iloc[sel_idx].reset_index(drop=True)

    n_genes = gene_mask.sum()
    n_total = x_test.shape[1]
    gene_feature_names = [f"gene_{i}" for i in range(n_genes)]

    n_selected = len(sel_idx)
    n_batches = int(np.ceil(n_selected / batch_size))

    def count_saved():
        files = os.listdir(SHAP_OUTPUT_DIR)
        mean_files = [f for f in files if f.startswith("shap_mean_batch_") and f.endswith(".csv")]
        total = 0
        for mf in mean_files:
            df = pd.read_csv(os.path.join(SHAP_OUTPUT_DIR, mf))
            total += len(df)
        return total

    print(f"Total selected instances to process: {n_selected}")

    for batch_id in range(n_batches):
        if batch_id in done_batches:
            print(f"Batch {batch_id} already completed. Skipping.")
            continue

        start = batch_id * batch_size
        end = min(start + batch_size, n_selected)
        batch_indices = np.arange(start, end)
        X_batch = X_sub[batch_indices]

        print(f"Processing batch {batch_id + 1}/{n_batches} with {len(batch_indices)} samples...")

        shap_mean_vals = explainer_mean.shap_values(X_batch, nsamples=nsamples)
        shap_unc_vals = explainer_unc.shap_values(X_batch, nsamples=nsamples)

        def _normalize_shap(arr):
            a = np.array(arr)
            if a.ndim == 3:
                return a[0]
            return a

        shap_mean_n = _normalize_shap(shap_mean_vals)
        shap_unc_n = _normalize_shap(shap_unc_vals)

        gene_cols = np.where(gene_mask)[0]
        shap_mean_genes = shap_mean_n[:, gene_cols]
        shap_unc_genes = shap_unc_n[:, gene_cols]

        mean_csv = os.path.join(SHAP_OUTPUT_DIR, f"shap_mean_batch_{batch_id}.csv")
        unc_csv = os.path.join(SHAP_OUTPUT_DIR, f"shap_unc_batch_{batch_id}.csv")
        meta_csv = os.path.join(SHAP_OUTPUT_DIR, f"meta_batch_{batch_id}.csv")

        _save_batch_csv(shap_mean_genes, mean_csv, gene_feature_names)
        _save_batch_csv(shap_unc_genes, unc_csv, gene_feature_names)
        meta_sub.iloc[start:end].to_csv(meta_csv, index=False)

        done_batches.append(batch_id)

        with open(progress_path, "wb") as f:
            pickle.dump({"done_batches": done_batches, "sel_idx": sel_idx.tolist(), "rng_state": rng.bit_generator.state}, f)

        so_far = count_saved()
        print(f"Saved progress after batch {batch_id}. Total instances saved so far: {so_far}")

    mean_files = sorted([f for f in os.listdir(SHAP_OUTPUT_DIR) if f.startswith("shap_mean_batch_") and f.endswith(".csv")])
    unc_files = sorted([f for f in os.listdir(SHAP_OUTPUT_DIR) if f.startswith("shap_unc_batch_") and f.endswith(".csv")])

    mean_dfs = [pd.read_csv(os.path.join(SHAP_OUTPUT_DIR, f)) for f in mean_files]
    unc_dfs = [pd.read_csv(os.path.join(SHAP_OUTPUT_DIR, f)) for f in unc_files]

    shap_mean_full = pd.concat(mean_dfs, ignore_index=True).to_numpy()
    shap_unc_full = pd.concat(unc_dfs, ignore_index=True).to_numpy()

    pd.DataFrame(shap_mean_full, columns=gene_feature_names).to_csv(
        os.path.join(SHAP_OUTPUT_DIR, "shap_mean_full.csv"), index=False
    )
    pd.DataFrame(shap_unc_full, columns=gene_feature_names).to_csv(
        os.path.join(SHAP_OUTPUT_DIR, "shap_unc_full.csv"), index=False
    )
    meta_sub.to_csv(os.path.join(SHAP_OUTPUT_DIR, "meta_full.csv"), index=False)

    return shap_mean_full, shap_unc_full, X_sub, meta_sub


if __name__ == "__main__":
    BASE_PATH = "../data/experiments/5_fold_cross_validation/"
    DATA_DIR = "../data"

    data = load_data(BASE_PATH, CURRENT_SPLIT, data_dir=DATA_DIR)

    x_train = data["train"]["features"]
    y_train = data["train"]["targets"]
    x_val = data["validation"]["features"]
    y_val = data["validation"]["targets"]
    x_test = data["test"]["features"]
    y_test = data["test"]["targets"]
    gene_list = data["gene_list"]

    model_params = {
        "n_features": x_train.shape[1],
        "n_units_per_layer": [128, 32, 16],
        "dropout_prob": 0.2,
    }

    model = load_model(model_params)

    n_genes = len(gene_list)
    n_total = x_test.shape[1]
    feature_names = list(gene_list) + [f"drug_feature_{i}" for i in range(n_total - n_genes)]
    gene_mask = np.array([not n.startswith("drug_feature_") for n in feature_names])

    shap_mean, shap_unc, X_sub, meta_sub = compute_shap_resumable(
        model, x_train, x_test, y_test, gene_mask, max_instances=MAX_INSTANCES_SHAP
    )

    print("SHAP computation finished.")
