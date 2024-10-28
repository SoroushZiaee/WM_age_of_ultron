# utils/config_validator.py
from typing import Dict, Any
import yaml


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and convert config values to appropriate types."""

    # Validate model config
    model_config = config["model"]
    model_config["num_classes"] = int(model_config["num_classes"])
    model_config["hidden_size"] = int(model_config["hidden_size"])
    model_config["num_layers"] = int(model_config["num_layers"])
    model_config["dropout_rate"] = float(model_config["dropout_rate"])
    model_config["learning_rate"] = float(model_config["learning_rate"])
    model_config["weight_decay"] = float(model_config["weight_decay"])

    # Validate data config
    data_config = config["data"]
    data_config["num_timesteps"] = int(data_config["num_timesteps"])
    data_config["batch_size"] = int(data_config["batch_size"])
    data_config["num_workers"] = int(data_config["num_workers"])
    data_config["train_val_split"] = float(data_config["train_val_split"])

    # Validate training config
    training_config = config["training"]
    training_config["max_epochs"] = int(training_config["max_epochs"])
    training_config["precision"] = int(training_config["precision"])
    training_config["accumulate_grad_batches"] = int(
        training_config["accumulate_grad_batches"]
    )
    training_config["gradient_clip_val"] = float(training_config["gradient_clip_val"])
    training_config["early_stopping_patience"] = int(
        training_config["early_stopping_patience"]
    )

    return config


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load config file and validate its contents."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return validate_config(config)
