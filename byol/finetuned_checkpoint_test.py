import pytorch_lightning as pl
import torch
import json
import os

from finetuning import FineTune
from datamodules import RGZ_DataModule_Finetune
from paths import Path_Handler
from config import load_config_finetune
from models import BYOL
from finetuning import FineTune, MLPHead

def get_save_folder(ckpt_config):
    with open(ckpt_config, "r") as f:
        data = json.load(f)
        folder = data["save_folder"]
        wandb_project = data["wandb_project"]
        save_folder = folder + "/" + wandb_project
    return save_folder

def get_checkpoint_paths(ckpt_config):
    with open(ckpt_config, "r") as f:
        data = json.load(f)
        ckpt_folder = data["ckpt_folder"]
        wandb_project = data["wandb_project"]
        ckpt_names = data["ckpt_names"]
    ckpt_paths = []
    for ckpt_name in ckpt_names:
        path = ckpt_folder + "/" + ckpt_name + "/" + wandb_project + "/" + ckpt_name + "/" + "checkpoints/" + "epoch=299-step=3600.ckpt"
        ckpt_paths.append(path)
    return ckpt_paths

def get_checkpoint_names(ckpt_config):
    with open(ckpt_config, "r") as f:
        data = json.load(f)
        ckpt_names = data["ckpt_names"]
    return ckpt_names

def load_checkpoint(ckpt_path):
    # Load finetuning configuration
    config = load_config_finetune()

    # Load the pretrained BYOL model to get the encoder.
    byol_ckpt_path = "byol.ckpt"
    byol_model = BYOL.load_from_checkpoint(byol_ckpt_path)
    encoder = byol_model.encoder

    # Recreate the head based on your configuration.
    if config["finetune"]["head"] == "linear":
        head = "linear"
    elif config["finetune"]["head"] == "mlp":
        head = MLPHead(
            input_dim=encoder.dim,
            depth=config["finetune"]["depth"],
            width=config["finetune"]["width"],
            output_dim=config["finetune"]["n_classes"],
        )
    else:
        raise ValueError("Unsupported head type specified in config.")

    # Load the finetuned checkpoint
    model = FineTune.load_from_checkpoint(ckpt_path, encoder=encoder, head=head)
    return model

def load_dataloader():
    paths = Path_Handler()._dict()
    config = load_config_finetune()
    datamodule = RGZ_DataModule_Finetune(
        paths["mb"],
        batch_size=config["finetune"]["batch_size"],
        center_crop=config["augmentations"]["center_crop"],
        val_size=config["finetune"]["val_size"],
        num_workers=config["dataloading"]["num_workers"],
        prefetch_factor=config["dataloading"]["prefetch_factor"],
        pin_memory=config["dataloading"]["pin_memory"],
        seed=config["finetune"]["seed"],
    )
    dataloader = datamodule.test_dataloader()
    return dataloader

def get_accuracy(ckpt_path):
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    model = load_checkpoint(ckpt_path)
    prediction_loader = load_dataloader()
    batch_results = trainer.predict(model, dataloaders=prediction_loader)
    # Initialize aggregated counts for each class
    aggregated = {}
    for class_idx in range(model.n_classes):
        aggregated[f"class_{class_idx}"] = {"correct": 0, "total": 0}

    # Aggregate counts over all batches
    for batch_result in batch_results:
        for class_key, counts in batch_result.items():
            aggregated[class_key]["correct"] += counts["correct"]
            aggregated[class_key]["total"] += counts["total"]

    # Compute overall accuracy for each class
    overall_accuracy = {}
    for class_key, counts in aggregated.items():
        if counts["total"] > 0:
            overall_accuracy[class_key] = counts["correct"] / counts["total"]
        else:
            overall_accuracy[class_key] = None
    return overall_accuracy

def save_accuracy(ckpt_name, ckpt_path, save_folder):
    # Ensure save folder exists
    os.makedirs(save_folder, exist_ok=True)
    # Get overall per-class accuracies
    overall_accuracy = get_accuracy(ckpt_path)
    # Save overall per-class accuracies to a JSON file
    output_filepath = os.path.join(save_folder, f"{ckpt_name}_accuracy.json")
    with open(output_filepath, "w") as f:
        json.dump(overall_accuracy, f, indent=4)

def main():
    ckpt_names = get_checkpoint_names("ckpt_config.json")
    ckpt_paths = get_checkpoint_paths("ckpt_config.json")
    save_folder = get_save_folder("ckpt_config.json")
    for i in range (0, len(ckpt_names)):
        save_accuracy(ckpt_names[i], ckpt_paths[i], save_folder)

if __name__ == "__main__":
    main()