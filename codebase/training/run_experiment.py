"""Experiment-running framework."""
import argparse

import numpy as np
import pytorch_lightning as pl
import torch

import os, sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import lit_model
from training.util import DATA_CLASS_MODULE, import_class, MODEL_CLASS_MODULE, setup_data_and_model_from_args


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help=f"auto will choose gpu if available else specify cpu/gpu",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default= 10,
        help=f"Stop training once this number of epochs is reached",
    )

    # Basic arguments
    parser.add_argument(
        "--data_class",
        type=str,
        default="datamodule",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="TransformerModel",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
    )
    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
        help="If non-zero, applies early stopping, with the provided value as the 'patience' argument."
        + " Default is 0.",
    )

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_model.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser



def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0' 
    ```
    """
   
    parser = _setup_parser()
    args = parser.parse_args()
    data, model = setup_data_and_model_from_args(args)

    lit_model_class = lit_model.BaseLitModel

    if args.load_checkpoint is not None:
        lit_mode = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_mode = lit_model_class(args=args, model=model)

    log_dir = lit_mode.log_dirname()
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    goldstar_metric = "validation/loss"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs= 1, # can be passed as an argument
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]
    if args.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="validation/loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    trainer = pl.Trainer( callbacks=callbacks, logger=logger, accelerator = args.accelerator, devices = 1, max_epochs = args.max_epochs)

    trainer.fit(lit_mode, datamodule=data)
    
    """"
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(lit_mode, datamodule=data)
    """

if __name__ == "__main__":
    main()