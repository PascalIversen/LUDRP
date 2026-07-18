import os
import gc
import warnings
from typing import Type

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from uadr.utils.data_preprocessing import get_preprocessed_features
from uadr.utils.prediction_evaluation import prediction_quality_metrics
from uadr.utils.seeding import seed_everything
from uadr.models import uncertainty_aware_model as uam


warnings.filterwarnings("ignore", message="The 'train_dataloader' does not have many workers.*", category=UserWarning)
warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`.*", category=UserWarning)
warnings.filterwarnings("ignore", message="The number of training batches .* is smaller than the logging interval Trainer\\(log_every_n_steps=.*\\). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.")
warnings.filterwarnings("ignore", message="The 'val_dataloader' does not have many workers.*", category=UserWarning)
warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed as a dependency.*", category=UserWarning)

def finetune_and_predict(
    finetune_idx,
    model_class,
    model_kwargs,
    checkpoint_path,
    x_train,
    y_train,
    x_test,
    df_cl,
    x_validation,
    y_validation,
    idx_cl,
    max_epochs_finetune,
    batch_size,
    num_workers
):
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_ft = model_class.load_from_checkpoint(checkpoint_path, **model_kwargs)

    x_ft = np.concatenate([x_train, x_test[finetune_idx]], axis=0)
    y_ft = np.concatenate([y_train.response.values, df_cl.loc[finetune_idx].response.values], axis=0)
    cl_ft = np.concatenate([y_train.cell_lines.values, df_cl.loc[finetune_idx].cell_lines.values], axis=0)

    model_ft.finetune(
        X_train=x_ft,
        y_train=y_ft,
        X_eval=x_validation,
        y_eval=y_validation.response.values,
        y_train_cell_line_index=cl_ft,
        y_eval_cell_line_index=y_validation.cell_lines.values,
        trainer_params={"max_epochs": max_epochs_finetune},
        batch_size=batch_size,
        patience=0,
        num_workers=num_workers,
    )
    model_ft.eval()
    y_pred, _ = model_ft.predict_target_and_uncertainty(x_test[idx_cl])

    # Clean up memory
    del model_ft, x_ft, y_ft, cl_ft
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return y_pred

def case_specific_finetune_setup(
    run_id: str,
    model_class: Type[uam.UncertaintyAwareModel],
    model_kwargs: dict,
    scaling_mode: str = "drug_z_norm",
    n_bits_drug_fingerprints: int = 128,
    base_path: str = "examples/data/experiments/5_fold_cross_validation/",
    trainer_kwargs: dict = {"progress_bar_refresh_rate": 100},
    max_epochs: int = 100,
    max_epochs_finetune: int = 100,
    batch_size: int = 64,
    patience: int = 10,
    num_workers: int = 0,
    top_k_drugs: int = 50,
    save_frequency: int = 10,
    current_split: int = 0,
    seed: int = 0,

):
    # Seed per split so the random-baseline drug selection and model init are reproducible.
    seed_everything(seed + current_split)
    result_path = f"{base_path}/case-specific_finetune/{run_id}_{scaling_mode}/split_{current_split}/"
    os.makedirs(result_path, exist_ok=True)

    out = get_preprocessed_features(
        current_split=current_split,
        base_path=base_path,
        cross_validation_type="cell_lines_cold_start",
        scaling_mode=scaling_mode,
        n_bits_drug_fingerprints=n_bits_drug_fingerprints,
    )


    x_train = out["train"]["features"]
    y_train = out["train"]["targets"]
    x_validation = out["validation"]["features"]
    y_validation = out["validation"]["targets"]
    x_test = out["test"]["features"]
    y_test = out["test"]["targets"]

    # Convert to smaller data types if possible
    if x_train.dtype == np.float64:
        x_train = x_train.astype(np.float32)
        x_validation = x_validation.astype(np.float32)
        x_test = x_test.astype(np.float32)

    model_kwargs.update({"n_features": x_train.shape[1]})
    model = model_class(**model_kwargs)

    checkpoint_path = os.path.join(result_path, "best_model.ckpt")
    if os.path.exists(checkpoint_path):
        model = model_class.load_from_checkpoint(checkpoint_path, **model_kwargs)
    else:
        model = model_class(**model_kwargs)
        trainer_kwargs.update({"max_epochs": max_epochs})
        model.fit(
            X_train=x_train,
            y_train=y_train.response.values,
            X_eval=x_validation,
            y_eval=y_validation.response.values,
            y_train_cell_line_index=y_train.cell_lines.values,
            y_eval_cell_line_index=y_validation.cell_lines.values,
            trainer_params=trainer_kwargs,
            batch_size=batch_size,
            patience=patience,
            num_workers=num_workers,
            cross_validation_type="cell_lines_cold_start",
        )
        model.trainer.save_checkpoint(checkpoint_path)

    model.eval()

    # Process cell lines in smaller batches to avoid memory issues
    cell_lines = y_test.cell_lines.unique()

    # Try to resume from previous run if interrupted
    resume_file = f"{result_path}/case-specific_finetune_results_partial.csv"
    if os.path.exists(resume_file):
        existing_results = pd.read_csv(resume_file)
        completed_cell_lines = set(existing_results.cell_line.values)
        cell_lines = [
            cl for cl in cell_lines
            if cl not in completed_cell_lines
        ]
        results = existing_results.to_dict('records')
        print(f"Resuming from {len(existing_results)} completed cell lines")
    else:
        results = []
        # Get initial predictions for all test data
        y_preds, y_uncertainty = model.predict_target_and_uncertainty(x_test)
        y_test["y_preds"] = y_preds
        y_test["y_uncertainty"] = y_uncertainty

    prediction_dir = os.path.join(result_path, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    for i, cell_line in enumerate(tqdm(cell_lines, desc="Cell lines")):
        try:
            df_cl = y_test[y_test.cell_lines == cell_line].copy()
            idx_cl = df_cl.index

            # Use smaller top_k if cell line has fewer drugs
            actual_top_k = min(top_k_drugs, len(df_cl))

            idx_uncertain = df_cl.sort_values(
                "y_uncertainty", ascending=False
            ).head(actual_top_k).index
            idx_random = np.random.choice(
                df_cl.index, size=actual_top_k, replace=False
            )

            y_pred_uncertain = finetune_and_predict(
                idx_uncertain,
                model_class,
                model_kwargs,
                checkpoint_path,
                x_train,
                y_train,
                x_test,
                df_cl,
                x_validation,
                y_validation,
                idx_cl,
                max_epochs_finetune,
                batch_size,
                num_workers
            )

            y_pred_random = finetune_and_predict(
                idx_random,
                model_class,
                model_kwargs,
                checkpoint_path,
                x_train,
                y_train,
                x_test,
                df_cl,
                x_validation,
                y_validation,
                idx_cl,
                max_epochs_finetune,
                batch_size,
                num_workers
            )

            y_pred_base = y_test.loc[idx_cl, "y_preds"].values
            y_true = df_cl.response.values
            # Save predictions per cell line

            df_cl["y_base"] = y_pred_base
            df_cl["y_uncertain"] = y_pred_uncertain
            df_cl["y_random"] = y_pred_random

            idx_intersection = df_cl.index.difference(
                idx_uncertain
            ).difference(idx_random)
            idx_intersection_pos = [
                df_cl.index.get_loc(i) for i in idx_intersection
            ]
            y_true_inter = y_true[idx_intersection_pos]
            y_base_inter = y_pred_base[idx_intersection_pos]
            y_uncertain_inter = y_pred_uncertain[idx_intersection_pos]
            y_random_inter = y_pred_random[idx_intersection_pos]
            df_cl["is_uncertain"] = df_cl.index.isin(idx_uncertain)
            df_cl["is_random"] = df_cl.index.isin(idx_random)
            df_cl["is_intersection"] = (
                ~df_cl["is_uncertain"] & ~df_cl["is_random"]
            )

            df_cl.to_csv(
                os.path.join(prediction_dir, f"{cell_line}.csv"), index=False
            )
            try:
                results.append({
                        "cell_line": cell_line,
                        "mse_base": prediction_quality_metrics(y_pred_base, y_true)["mse"],
                        "mse_uncertain": prediction_quality_metrics(y_pred_uncertain, y_true)["mse"],
                        "mse_random": prediction_quality_metrics(y_pred_random, y_true)["mse"],
                        "mse_base_inter": prediction_quality_metrics(y_base_inter, y_true_inter)["mse"],
                        "mse_uncertain_inter": prediction_quality_metrics(y_uncertain_inter, y_true_inter)["mse"],
                        "mse_random_inter": prediction_quality_metrics(y_random_inter, y_true_inter)["mse"],
                        "pearson_base": prediction_quality_metrics(y_pred_base, y_true)["pearson"],
                        "pearson_uncertain": prediction_quality_metrics(y_pred_uncertain, y_true)["pearson"],
                        "pearson_random": prediction_quality_metrics(y_pred_random, y_true)["pearson"],
                        "pearson_base_inter": prediction_quality_metrics(y_base_inter, y_true_inter)["pearson"],
                        "pearson_uncertain_inter": prediction_quality_metrics(y_uncertain_inter, y_true_inter)["pearson"],
                        "pearson_random_inter": prediction_quality_metrics(y_random_inter, y_true_inter)["pearson"],
                    })

            except ValueError as e:
                results.append({
                    "cell_line": cell_line,
                    "mse_base": np.nan,
                    "mse_uncertain": np.nan,
                    "mse_random": np.nan,
                    "mse_base_inter": np.nan,
                    "mse_uncertain_inter": np.nan,
                    "mse_random_inter": np.nan,
                    "pearson_base": np.nan,
                    "pearson_uncertain": np.nan,
                    "pearson_random": np.nan,
                    "pearson_base_inter": np.nan,
                    "pearson_uncertain_inter": np.nan,
                    "pearson_random_inter": np.nan,
                })

            # Save intermediate results periodically
            if (i + 1) % save_frequency == 0:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(resume_file, index=False)
                print(f"Saved intermediate results after {i + 1} cell lines")

            if len(results) > 0:
                print(f"Processed {len(results)} cell lines")
                df_results = pd.DataFrame(results)
                current_means = df_results.select_dtypes(include=[np.number]).mean()

                print("Current mean results:")
                print(current_means)

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing cell line {cell_line}: {e}")
            # Save what we have so far
            if results:
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(f"{result_path}/case-specific_finetune_results_error_backup.csv", index=False)
            raise e

    df = pd.DataFrame(results)
    df.to_csv(f"{result_path}/case-specific_finetune_results.csv", index=False)

    # Clean up intermediate file
    if os.path.exists(resume_file):
        os.remove(resume_file)
    print(f"Final results saved to {result_path}/case-specific_finetune_results.csv")
    print("\nFinal Results:")
    print("\nMean MSE (full set):")
    print(df[["mse_base", "mse_random", "mse_uncertain"]].mean())

    print("\nMean MSE (intersection):")
    print(df[["mse_base_inter", "mse_random_inter", "mse_uncertain_inter"]].mean())

    print("\nMean Pearson (full set):")
    print(df[["pearson_base", "pearson_random", "pearson_uncertain"]].mean())

    print("\nMean Pearson (intersection):")
    print(df[["pearson_base_inter", "pearson_random_inter", "pearson_uncertain_inter"]].mean())

