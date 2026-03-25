if __name__ == "__main__":

    import os
    import logging
    import pytorch_lightning as pl
    import argparse

    from uadr.models import probabilistic_NN as pnn
    from uadr.experiments.run_case_specific_finetuning import case_specific_finetune_setup


    pl_logger = logging.getLogger("pytorch_lightning")
    pl_logger.setLevel(logging.WARNING)

    pl_logger = logging.getLogger("pytorch_lightning")


    parser = argparse.ArgumentParser(description="Run case-specific finetuning setup.")
    parser.add_argument("--run_id", type=str, default="test_run")
    parser.add_argument("--scaling_mode", type=str, default="z_norm")
    args = parser.parse_args()

    run_id = args.run_id
    scaling_mode = args.scaling_mode

    model_kwargs = {
        "n_units_per_layer": [[64, 16, 8]],
        "dropout_prob": [0.1],
        "importance_weighting": [False],
    }
    model_class = pnn.ProbabilisticFeedForwardNetwork

    # use first config only if tuning is off
    for k in model_kwargs:
        model_kwargs[k] = model_kwargs[k][0] if isinstance(model_kwargs[k], list) else model_kwargs[k]

    trainer_kwargs = {"progress_bar_refresh_rate": 0}
    top_k_drugs = 60
    finetuning_epochs = 10
    print(f"Running case-specific finetuning setup with run_id: {run_id}, scaling_mode: {scaling_mode}, top_k_drugs: {top_k_drugs}, finetuning_epochs: {finetuning_epochs}")

    run_id = f"{run_id}_top{top_k_drugs}_epochsfine{finetuning_epochs}"
    for split in range(5):
        case_specific_finetune_setup(
            run_id=run_id,
            model_class=model_class,
            model_kwargs=model_kwargs,
            scaling_mode=scaling_mode,
            base_path="data/experiments/5_fold_cross_validation/",
            trainer_kwargs=trainer_kwargs,
            max_epochs=80,
            max_epochs_finetune=finetuning_epochs,
            batch_size=128,
            patience=5,
            num_workers=0,
            top_k_drugs=top_k_drugs,
            current_split=split,
        )
