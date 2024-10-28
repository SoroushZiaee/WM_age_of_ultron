import sys
import os
from argparse import ArgumentParser, Namespace

# Add the parent directory to the Python path
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
parent_dir = os.path.dirname(parent_dir)

print(f"{script_dir = }")
print(f"{parent_dir = }")
sys.path.append(parent_dir)

# train.py
import lightning as l
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from datetime import datetime
import os
import yaml

from lit_modules.datamodule.coco1600_datamodule import Coco1600DataModule
from lit_modules.modules.wm_model_lightning import ResNetLSTMModule

import logging

logger = logging.getLogger(__name__)


def setup_cuda_optimization():
    """Setup CUDA optimization settings."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        cuda_device = torch.cuda.get_device_name(0)
        cuda_capability = torch.cuda.get_device_capability(0)
        logger.info(
            f"Using CUDA device: {cuda_device} with capability {cuda_capability}"
        )
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(
            "CUDA optimizations enabled: TF32, cuDNN benchmark, and high precision matmul"
        )


def format_scientific(number: float) -> str:
    """
    Format a number in scientific notation without trailing zeros.
    Example: 0.001 -> 1e-3, 0.0001 -> 1e-4

    Args:
        number: Number to format

    Returns:
        str: Formatted number
    """
    if number == 0:
        return "0"
    exp = int(f"{number:e}".split("e")[1])
    if exp == 0:
        return str(number)
    return f"1e{exp}"


def get_experiment_name(config: dict) -> str:
    """
    Generate experiment name with model architecture and key hyperparameters.

    Args:
        config: Configuration dictionary

    Returns:
        str: Formatted experiment name
    """
    try:
        model_config = config.get("model", {})
        training_config = config.get("training", {})
        optimizer_config = config.get("optimizer", {})
        data_config = config.get("data", {})

        # Get model type (default to resnet_lstm_feedback for backward compatibility)
        model_type = model_config.get("model_type", "resnet_lstm_feedback")

        # Get key parameters (add defaults in case they're not in config)
        hidden_size = model_config.get("hidden_size", "unk")
        num_layers = model_config.get("num_layers", "unk")
        learning_rate = model_config.get("learning_rate", "unk")
        weight_decay = model_config.get("weight_decay", "unk")
        batch_size = data_config.get("batch_size", "unk")
        max_epochs = training_config.get("max_epochs", "unk")

        # Format learning rate and weight decay in scientific notation
        if isinstance(learning_rate, (int, float)):
            lr_str = format_scientific(learning_rate)
        else:
            lr_str = str(learning_rate)

        if isinstance(weight_decay, (int, float)):
            wd_str = format_scientific(weight_decay)
        else:
            wd_str = str(weight_decay)

        # Create model-specific suffix
        model_suffix = ""
        if model_type == "resnet_lstm_feedback":
            model_suffix = "_feedback"

        # Format the experiment name
        exp_name = (
            f"{model_type}"  # Start with model type
            f"{model_suffix}"  # Add model-specific suffix
            f"_h{hidden_size}"
            f"_l{num_layers}"
            f"_lr{lr_str}"
            f"_wd{wd_str}"
            f"_bs{batch_size}"
            f"_e{max_epochs}"
        )

        return exp_name

    except Exception as e:
        logger.warning(f"Error creating detailed experiment name: {e}")
        return f"experiment_unknown_architecture"  # Fallback name


def train(config_path: str):
    setup_cuda_optimization()
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create unique experiment name with hyperparameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = get_experiment_name(config)

    # Initialize logger with more configurations
    logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "logs"),
        name=exp_name,
        default_hp_metric=True,
        log_graph=True,
        version=timestamp,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(os.getcwd(), "checkpoints", exp_name),
            filename="{epoch}-{val/loss:.2f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=config["training"]["early_stopping_patience"],
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Initialize trainer with logging configurations
    trainer = l.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=config["training"]["precision"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        log_every_n_steps=1,
        enable_model_summary=True,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    # Log the hyperparameters
    trainer.logger.log_hyperparams(config)

    # Initialize data module and model
    data_module = Coco1600DataModule(config_path, logger=trainer.logger)
    model = ResNetLSTMModule(config_path)

    # Train the model
    trainer.fit(model, data_module)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Train the ResNet-LSTM model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the configuration file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Parse command line arguments
    args = parse_args()

    # Start training
    train(args.config)
