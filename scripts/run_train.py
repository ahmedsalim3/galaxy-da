import argparse
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import torch
import yaml
from geomloss.distance_metrics import DISTANCE_METRICS

from nebula.commons import Logger, log_config, set_all_seeds
from nebula.data.dataloaders import GalaxyDataModule
from nebula.modeling.configs import (BaseTrainerConfig, DAAdversarialConfig,
                                     DAFixedLambdaConfig,
                                     DATrainableWeightsConfig,
                                     DATrainableWeightsSigmaConfig, NoDAConfig)
from nebula.modeling.trainers import (DAAdversarialTrainer,
                                      DAFixedLambdaTrainer,
                                      DATrainableWeightsSigmaTrainer,
                                      DATrainableWeightsTrainer, NoDATrainer)
from nebula.models import (CNN, ESCNNConfig, ESCNNSteerable, ResNetBackbone,
                           ResNetConfig)
from nebula.visualizations import plot_diag_history, plot_training_history


def build_data_module(config: dict) -> GalaxyDataModule:
    data_config = config["data"]
    log_config(data_config, "Data Config")
    data_module = GalaxyDataModule(
        source_img_dir=data_config["source_img_dir"],
        source_labels=data_config["source_labels"],
        target_img_dir=data_config.get("target_img_dir", None),
        target_labels=data_config.get("target_labels", None),
        include_rotations=data_config.get("include_rotations", False),
        shared_norm=data_config.get("shared_norm", False),
        source_mean=data_config.get("source_mean"),
        source_std=data_config.get("source_std"),
        target_mean=data_config.get("target_mean"),
        target_std=data_config.get("target_std"),
        image_size=tuple(data_config.get("image_size", (28, 28))),
        batch_size=data_config.get("batch_size", 64),
        val_size=data_config.get("val_size", 0.2),
        num_workers=data_config.get("num_workers", 4),
        seed=config.get("seed", 42),
    )
    return data_module


def build_model(
    model_type, image_size, model_config: dict | None = None
) -> torch.nn.Module:
    model_config = model_config or {}
    if model_type == "cnn":
        model = CNN(
            num_classes=3,
            input_size=(3, *image_size),
        )
    elif model_type == "resnet":
        model = ResNetBackbone(
            config=ResNetConfig(
                num_classes=3,
                arch=model_config.get("arch", ResNetConfig.arch),
                pretrained=bool(
                    model_config.get("pretrained", ResNetConfig.pretrained)
                ),
                trainable_layers=int(
                    model_config.get("trainable_layers", ResNetConfig.trainable_layers)
                ),
                dropout=float(model_config.get("dropout", ResNetConfig.dropout)),
            )
        )
    elif model_type == "escnn":
        model = ESCNNSteerable(
            config=ESCNNConfig(
                num_classes=3,
                group=str(model_config.get("group", ESCNNConfig.group)),
                N=int(model_config.get("N", ESCNNConfig.N)),
                dropout=float(model_config.get("dropout", ESCNNConfig.dropout)),
                base_width=int(model_config.get("base_width", ESCNNConfig.base_width)),
            )
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def build_config(model: torch.nn.Module, config: dict, device: torch.device):

    train_config = config["training"]
    # Base parameters
    base_params = {
        "num_epochs": int(train_config.get("num_epochs", 10)),
        "warmup_epochs": int(
            train_config.get("warmup_epochs", BaseTrainerConfig.warmup_epochs)
        ),
        "lr": float(train_config.get("lr", 1e-4)),
        "optimizer": str(train_config.get("optimizer", BaseTrainerConfig.optimizer)),
        "weight_decay": float(
            train_config.get("weight_decay", BaseTrainerConfig.weight_decay)
        ),
        "max_norm": float(train_config.get("max_norm", BaseTrainerConfig.max_norm)),
        "criterion": str(train_config.get("criterion", BaseTrainerConfig.criterion)),
        "use_class_weights": bool(
            train_config.get("use_class_weights", BaseTrainerConfig.use_class_weights)
        ),
        "class_weight_method": str(
            train_config.get(
                "class_weight_method", BaseTrainerConfig.class_weight_method
            )
        ),
        "class_weight_beta": float(
            train_config.get("class_weight_beta", BaseTrainerConfig.class_weight_beta)
        ),
        "early_stopping_patience": (
            None
            if train_config.get(
                "early_stopping_patience", BaseTrainerConfig.early_stopping_patience
            )
            is None
            else int(train_config.get("early_stopping_patience"))
        ),
        "early_stopping_metric": str(
            train_config.get(
                "early_stopping_metric", BaseTrainerConfig.early_stopping_metric
            )
        ),
    }
    if base_params["criterion"] == "focal":
        base_params["focal_gamma"] = float(
            train_config.get("focal_gamma", BaseTrainerConfig.focal_gamma)
        )
        fa_val = train_config.get("focal_alpha", BaseTrainerConfig.focal_alpha)
        if fa_val is None:
            base_params["focal_alpha"] = None
        elif isinstance(fa_val, list):
            base_params["focal_alpha"] = [float(x) for x in fa_val]
        elif isinstance(fa_val, str):
            # a special value "class_weights"
            base_params["focal_alpha"] = fa_val
        else:
            base_params["focal_alpha"] = float(fa_val)
        base_params["focal_reduction"] = str(
            train_config.get("focal_reduction", BaseTrainerConfig.focal_reduction)
        )

    method = train_config["method"]
    if method == "baseline":
        trainer_config = NoDAConfig(**base_params)
        trainer = NoDATrainer(model, trainer_config, device)
    elif method == "adversarial":
        # calculate latent dimension from the model if not provided
        if train_config.get("latent_dim", None) is None:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, *config["data"]["image_size"]).to(
                    device
                )
                _, dummy_latent = model(dummy_input)
            latent_dim = dummy_latent.shape[1]
            if was_training:
                model.train()
        else:
            latent_dim = int(train_config["latent_dim"])

        adv_params = {
            **base_params,
            "lambda_grl": float(
                train_config.get("lambda_grl", DAAdversarialConfig.lambda_grl)
            ),
            "latent_dim": latent_dim,
            "domain_hidden_dim": int(
                train_config.get(
                    "domain_hidden_dim", DAAdversarialConfig.domain_hidden_dim
                )
            ),
            "use_projection": bool(
                train_config.get("use_projection", DAAdversarialConfig.use_projection)
            ),
            "domain_projection_dim": int(
                train_config.get(
                    "domain_projection_dim", DAAdversarialConfig.domain_projection_dim
                )
            ),
        }
        trainer_config = DAAdversarialConfig(**adv_params)
        trainer = DAAdversarialTrainer(model, trainer_config, device)
    #elif method in ["sinkhorn", "mmd", "energy","inner_product","cosine","jaccard","dice_coefficient","kumar_hassebrook","harmonic_mean","euclidean","manhattan","chebyshev","minkowski","weighted_minkowski","gower","avg_l1_linf"]:
    elif method in ["sinkhorn", "mmd", "energy"] or method in DISTANCE_METRICS:
        da_params = {
            **base_params,
            "lambda_da": float(
                train_config.get("lambda_da", DAFixedLambdaConfig.lambda_da)
            ),
            "method": method,
            "sinkhorn_blur": float(
                train_config.get("sinkhorn_blur", DAFixedLambdaConfig.sinkhorn_blur)
            ),
            "sinkhorn_p": int(
                train_config.get("sinkhorn_p", DAFixedLambdaConfig.sinkhorn_p)
            ),
        }
        use_trainable_weights = train_config.get("use_trainable_weights", False)
        use_sigma_schedule = train_config.get("use_sigma_schedule", False)

        if use_trainable_weights and use_sigma_schedule:
            sigma_params = {
                **da_params,
                "eta_1_init": float(
                    train_config.get(
                        "eta_1_init", DATrainableWeightsSigmaConfig.eta_1_init
                    )
                ),
                "eta_2_init": float(
                    train_config.get(
                        "eta_2_init", DATrainableWeightsSigmaConfig.eta_2_init
                    )
                ),
                "sigma_schedule_type": train_config.get(
                    "sigma_schedule_type",
                    DATrainableWeightsSigmaConfig.sigma_schedule_type,
                ),
                "sigma_initial_blur": float(
                    train_config.get(
                        "sigma_initial_blur",
                        DATrainableWeightsSigmaConfig.sigma_initial_blur,
                    )
                ),
                "sigma_decay_rate": float(
                    train_config.get(
                        "sigma_decay_rate",
                        DATrainableWeightsSigmaConfig.sigma_decay_rate,
                    )
                ),
                "sigma_final_blur": float(
                    train_config.get(
                        "sigma_final_blur",
                        DATrainableWeightsSigmaConfig.sigma_final_blur,
                    )
                ),
                "sigma_step_size": int(
                    train_config.get(
                        "sigma_step_size", DATrainableWeightsSigmaConfig.sigma_step_size
                    )
                ),
                "sigma_step_gamma": float(
                    train_config.get(
                        "sigma_step_gamma",
                        DATrainableWeightsSigmaConfig.sigma_step_gamma,
                    )
                ),
                "sigma_poly_power": float(
                    train_config.get(
                        "sigma_poly_power",
                        DATrainableWeightsSigmaConfig.sigma_poly_power,
                    )
                ),
            }
            trainer_config = DATrainableWeightsSigmaConfig(**sigma_params)
            trainer = DATrainableWeightsSigmaTrainer(model, trainer_config, device)
        elif use_trainable_weights:
            weights_params = {
                **da_params,
                "eta_1_init": float(
                    train_config.get("eta_1_init", DATrainableWeightsConfig.eta_1_init)
                ),
                "eta_2_init": float(
                    train_config.get("eta_2_init", DATrainableWeightsConfig.eta_2_init)
                ),
            }
            trainer_config = DATrainableWeightsConfig(**weights_params)
            trainer = DATrainableWeightsTrainer(model, trainer_config, device)
        else:
            trainer_config = DAFixedLambdaConfig(**da_params)
            trainer = DAFixedLambdaTrainer(model, trainer_config, device)
    else:
        raise ValueError(f"Unknown method: {method}")

    log_config(trainer_config.__dict__, "Trainer Config")
    return trainer, trainer_config


def main():
    p = argparse.ArgumentParser(
        epilog="See configs/config.template.yml for info about the config"
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--use_diagnostics", action="store_true")
    p.add_argument("--eval_interval", type=int, default=0)
    p.add_argument("--diag_max_batches", type=int, default=5)
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    path = Path(args.config)
    config["config_file_path"] = str(path.resolve())
    config.setdefault("experiment_name", path.stem)

    output_root = (
        Path(config.get("output", {}).get("root_dir", "experiments"))
        / config["experiment_name"]
    )
    ckpt_dir, logs_dir = output_root / "ckpts", output_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(config.get("seed", 42))
    logger = Logger(output_root / "train_logs.log")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"=" * 70)
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment: {config['experiment_name']}")
    logger.info(f"Config: {config['config_file_path']}")
    logger.info(f"=" * 70)

    data_module = build_data_module(config)
    model = build_model(
        config["model"]["type"],
        config["data"]["image_size"],
        config.get("model", {}),
    )
    model.to(device)
    trainer, _ = build_config(model, config, device)

    use_target = config["training"]["method"] != "baseline"
    # use_target = True
    source_train_loader = data_module.source_train_loader
    target_train_loader = data_module.target_train_loader if use_target else None

    if args.use_diagnostics:
        assert (
            args.eval_interval > 0
        ), "eval_interval must be greater than 0 when using diagnostics\n      call with --eval_interval 1 to diagnose every epoch"
        assert (
            args.diag_max_batches is not None
        ), "diag_max_batches must be set when using diagnostics\n      call with --diag_max_batches 10 to use 10 batches for diagnostics"
        assert (
            target_train_loader is not None
        ), "target_train_loader must be set when using diagnostics as the target domain is used for diagnostics\n      set target_img_dir, target_labels in the config file or train without diagnostics"
        histories = trainer.train(
            source_loader=source_train_loader,
            target_loader=target_train_loader,
            eval_interval=args.eval_interval,
            diag_max_batches=args.diag_max_batches,
        )
    else:
        histories = trainer.train(
            source_loader=source_train_loader,
            target_loader=target_train_loader,
            eval_interval=args.eval_interval,
            diag_max_batches=args.diag_max_batches,
        )

    history = histories["history"]
    diag_history = histories["diag_history"]
    history_df = histories["history_df"]
    diag_history_df = histories["diag_history_df"]

    # save history plots, and csv logs
    _ = plot_training_history(
        history,
        trainer=trainer,
        save_path=logs_dir / f"{config['experiment_name']}_history.png",
    )
    history_df.to_csv(
        logs_dir / f"{config['experiment_name']}_history.csv", index=False
    )
    if args.use_diagnostics:
        _ = plot_diag_history(
            diag_history,
            warmup_epochs=int(config.get("training", {}).get("warmup_epochs", 0) or 0),
            save_path=logs_dir / f"{config['experiment_name']}_diag.png",
        )
        diag_history_df.to_csv(
            logs_dir / f"{config['experiment_name']}_diag.csv", index=False
        )

        df = pd.merge(history_df, diag_history_df, on="epoch", how="left")
        diag_cols = [c for c in diag_history_df.columns if c != "epoch"]
        df[diag_cols] = df[diag_cols].fillna(0)
        df.to_csv(logs_dir / f"{config['experiment_name']}.csv", index=False)
    logger.info(f"Logs saved: {logs_dir}")

    ckpt_path = ckpt_dir / f"{config['experiment_name']}.pt"
    trainer.save_checkpoint(str(ckpt_path), full_config=config)

    logger.info


if __name__ == "__main__":
    main()
