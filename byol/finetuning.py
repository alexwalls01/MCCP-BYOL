import wandb
import pytorch_lightning as pl
import logging
import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import os
import json

from pathlib import Path
from einops import rearrange
from typing import Any, Dict, List, Tuple, Type, Union
from torch import Tensor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from paths import Path_Handler
from config import load_config, update_config, load_config_finetune, load_config_evaluation
from models import BYOL
from datamodules import RGZ_DataModule_Finetune
from datasets import MBFRFull, RGZ108k
from plot_embedding import Reducer, plot_embedding

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.linear(x)
        return x


class FineTune(pl.LightningModule):
    """
    Parent class for self-supervised LightningModules to perform linear evaluation with multiple
    data-sets.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head,
        dim: int,
        n_classes,
        n_epochs=100,
        n_layers=0,
        batch_size=1024,
        lr_decay=0.75,
        seed=69,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "head"])

        self.n_layers = n_layers
        self.batch_size = batch_size
        self.encoder = encoder
        self.lr_decay = lr_decay
        self.n_epochs = n_epochs
        self.seed = seed
        self.head = head
        self.n_classes = n_classes
        self.layers = []

        # Set head
        if head == "linear":
            self.head = LogisticRegression(input_dim=dim, output_dim=n_classes)
            self.head_type = "linear"
        elif isinstance(head, nn.Module):
            self.head = head
            self.head_type = "custom"
        else:
            raise ValueError("Head must be either 'linear' or a PyTorch Module")

        # Set finetuning layers for easy access
        if self.n_layers:
            layers = self.encoder.finetuning_layers
            assert self.n_layers <= len(
                layers
            ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"

            self.layers = layers[::-1][:n_layers]

        self.train_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.val_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.test_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    def on_fit_start(self):
        # Log size of data-sets #

        self.train_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)
        self.val_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.test_acc = nn.ModuleList(
            [
                tm.Accuracy(
                    task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
                ).to(self.device)
            ]
            * len(self.trainer.datamodule.data["test"])
        )

        logging_params = {f"n_{key}": len(value) for key, value in self.trainer.datamodule.data.items()}
        self.logger.log_hyperparams(logging_params)

        # Make sure network that isn't being finetuned is frozen
        # probably unnecessary but best to be sure
        set_grads(self.encoder, False)
        if self.n_layers:
            for layer in self.layers:
                set_grads(layer, True)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y, _ = batch
        logits = self.forward(x)
        y_pred = logits.softmax(dim=-1)
        loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.n_layers else 0)
        self.log("finetuning/train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, _ = batch
        preds = self.forward(x)
        self.val_acc(preds, y)
        self.log("finetuning/val_acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, _ = batch
        name = list(self.trainer.datamodule.data["test"].keys())[dataloader_idx]

        preds = self.forward(x)
        self.test_acc[dataloader_idx](preds, y)
        self.log(
            f"finetuning/test/{name}_acc",
            self.test_acc[dataloader_idx],
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # Unpack the batch: x (inputs), y (labels), filenames (identifiers)
        x, y, filenames = batch

        # Run the forward pass to get logits and predictions.
        logits = self.forward(x)
        preds = logits.argmax(dim=1)
        
        # Convert tensors to lists where needed.
        y_list = y.tolist() if torch.is_tensor(y) else y
        preds_list = preds.tolist() if torch.is_tensor(preds) else preds
        # Ensure filenames is a list (if it isn’t already).
        if not isinstance(filenames, list):
            filenames = list(filenames)

        results = {}
        for class_idx in range(self.n_classes):
            # Get the indices of samples that belong to the current class using torch.where.
            indices = torch.where(y == class_idx)[0].tolist()
            total = len(indices)
            if total > 0:
                # Using the indices from the mask, select the filenames for correct/incorrect predictions.
                correct_ids = [filenames[i] for i in indices if preds_list[i] == y_list[i]]
                incorrect_ids = [filenames[i] for i in indices if preds_list[i] != y_list[i]]
                correct = len(correct_ids)
                all_files = [filenames[i] for i in indices]
                all_predictions = [preds_list[i] for i in indices]
            else:
                correct = 0
                correct_ids = []
                incorrect_ids = []
                all_files = []
                all_predictions = []
            results[f"class_{class_idx}"] = {
                "correct": correct,
                "total": total,
                "correct_ids": correct_ids,
                "incorrect_ids": incorrect_ids,
                "all_files": all_files,
                "all_predictions": all_predictions
            }
        return results


    def configure_optimizers(self):
        if not self.n_layers and self.head_type == "linear":
            # Scale base lr=0.1
            lr = 0.1 * self.batch_size / 256
            params = self.head.parameters()
            return torch.optim.SGD(params, momentum=0.9, lr=lr)
        else:
            lr = 0.001 * self.batch_size / 256
            params = [{"params": self.head.parameters(), "lr": lr}]
            # layers.reverse()

            # Append parameters of layers for finetuning along with decayed learning rate
            for i, layer in enumerate(self.layers):
                params.append({"params": layer.parameters(), "lr": lr * (self.lr_decay**i)})

            # Initialize AdamW optimizer with cosine decay learning rate
            opt = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.n_epochs)
            return [opt], [scheduler]


class MLPHead(nn.Module):
    """
    Fully connected head with a single hidden layer. Batchnorm applied as first layer so that
    feature space of encoder doesn't need to be normalized.
    """

    def __init__(self, input_dim, depth, width, output_dim):
        super(MLPHead, self).__init__()

        self.input_layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(depth):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(width, width),
                    nn.GELU(),
                )
            )

        self.output_layer = nn.Sequential(
            nn.Linear(width, output_dim),
        )

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)

        return x

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

def load_dataloader(stage):
    paths = Path_Handler()._dict()

    # Get model config
    config_finetune = load_config_finetune()
    byol_model = BYOL.load_from_checkpoint("byol.ckpt")
    config = byol_model.config
    config.update(config_finetune)
    config["finetune"]["dim"] = byol_model.encoder.dim

    # Compatibility with old style config
    if config["augmentations"]["center_crop"] is True:
        config["augmentations"]["center_crop"] = config["augmentations"]["center_crop_size"]
    
    seed = 42
    config["finetune"]["seed"] = seed
    pl.seed_everything(seed)

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
    datamodule.setup(stage=stage)
    if stage == "test":
        dataloader = datamodule.test_dataloader()
    elif stage == "val":
        dataloader = datamodule.val_dataloader()
    else:
        raise ValueError("Unsupported dataloader stage.")
    return dataloader

def calculate_accuracy(model, stage):
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    prediction_loader = load_dataloader(stage)
    batch_results = trainer.predict(model, dataloaders=prediction_loader)

    # Initialize aggregated counts for each class
    aggregated = {}
    for class_idx in range(model.n_classes):
        aggregated[f"class_{class_idx}"] = {"correct": 0, "total": 0}

    # Flatten nested results (if needed)
    flat_results = []
    for result in batch_results:
        if isinstance(result, list):
            flat_results.extend(result)
        else:
            flat_results.append(result)
    
    # Initialize aggregated counts for each class with keys for both correct and incorrect ids.
    aggregated = {f"class_{i}": {"correct": 0, "total": 0, "correct_ids": [], "incorrect_ids": [], "all_files": [], "all_predictions": []} 
                  for i in range(model.n_classes)}
    
    # Aggregate counts over all flattened batches
    for batch_result in flat_results:
        for class_key, counts in batch_result.items():
            aggregated[class_key]["correct"] += counts["correct"]
            aggregated[class_key]["total"] += counts["total"]
            aggregated[class_key]["correct_ids"].extend(counts.get("correct_ids", []))
            aggregated[class_key]["incorrect_ids"].extend(counts.get("incorrect_ids", []))
            aggregated[class_key]["all_files"].extend(counts.get("all_files", []))
            aggregated[class_key]["all_predictions"].extend(counts.get("all_predictions", []))
    
    # Compute overall accuracy per class
    overall_accuracy = {}
    for class_key, counts in aggregated.items():
        if counts["total"] > 0:
            overall_accuracy[class_key] = {
                "accuracy": counts["correct"] / counts["total"],
                "total": counts["total"],
                "correct": counts["correct"],
                "correct_ids": counts["correct_ids"],
                "incorrect_ids": counts["incorrect_ids"],
                "all_files": counts["all_files"],
                "all_predictions": counts["all_predictions"]
            }
        else:
            overall_accuracy[class_key] = {
                "accuracy": None,
                "total": counts["total"],
                "correct": counts["correct"],
                "correct_ids": counts["correct_ids"],
                "incorrect_ids": counts["incorrect_ids"],
                "all_files": counts["all_files"],
                "all_predictions": counts["all_predictions"]
            }
    
    return overall_accuracy

def save_accuracy(ckpt_name, model, save_dir, stage):
    # Ensure save folder exists
    os.makedirs(save_dir, exist_ok=True)
    # Get overall per-class accuracies
    overall_accuracy = calculate_accuracy(model, stage)
    # Save overall per-class accuracies to a JSON file
    output_filepath = os.path.join(save_dir, f"{ckpt_name}_{stage}_accuracy.json")
    with open(output_filepath, "w") as f:
        json.dump(overall_accuracy, f, indent=4)

def run_post_evaluation(run_id):

    paths = Path_Handler()._dict()

    eval_config = load_config_evaluation()
    finetune_config = load_config_finetune()

    ckpt_folder = eval_config['ckpt_folder']
    wandb_project = finetune_config['finetune']['wandb_project']
    save_dir = eval_config['save_dir'] + "/" + wandb_project
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_folder, run_id, wandb_project, run_id, "checkpoints", "epoch=299-step=3600.ckpt")
    model = load_checkpoint(ckpt_path)

    # Save accuracy data for the run to a JSON file
    save_accuracy(run_id, model, save_dir, "val")
    save_accuracy(run_id, model, save_dir, "test")

    byol_model = BYOL.load_from_checkpoint("byol.ckpt")
    config = byol_model.config
    mu, sig = config["data"]["mu"], config["data"]["sig"]
    encoder = byol_model.encoder

    transform = T.Compose(
        [
            T.CenterCrop(70),
            T.ToTensor(),
            T.Normalize((mu,), (sig,)),
        ]
    )

    PCA_COMPONENTS = 200
    UMAP_N_NEIGHBOURS = 75
    UMAP_MIN_DIST = 0.01
    METRIC = "cosine"

    rgz = RGZ108k(
        paths["rgz"],
        train=True,
        transform=transform,
        download=False,
        remove_duplicates=False,
        cut_threshold=25,
        mb_cut=True,
    )

    reducer = Reducer(encoder, PCA_COMPONENTS, UMAP_N_NEIGHBOURS, UMAP_MIN_DIST, METRIC)
    reducer.fit(rgz)

    # Get umap embeddings for data with original MiraBest labels
    mb = MBFRFull(root=paths["mb"],
                  train=False,
                  transform=transform,
                  download=False,
                  aug_type="torchvision"
                  )
    mb_fri = mb.subset_by_label(0)
    mb_frii = mb.subset_by_label(1)
    mb_hybrid = mb.subset_by_label(2)

    # Get umap embeddings for data with model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    mb.assign_pseudo_labels(model)
    predictions_fri = mb.subset_by_label(0)
    predictions_frii = mb.subset_by_label(1)
    predictions_hybrid = mb.subset_by_label(2)

    mb_fri_umap = reducer.transform(mb_fri)
    mb_frii_umap = reducer.transform(mb_frii)
    mb_hybrid_umap = reducer.transform(mb_hybrid)
    predictions_fri_umap = reducer.transform(predictions_fri)
    predictions_frii_umap = reducer.transform(predictions_frii)
    predictions_hybrid_umap = reducer.transform(predictions_hybrid)

    # Put together data to plot
    mb_data = {"fri_umap": mb_fri_umap,
               "frii_umap": mb_frii_umap,
               "hybrid_umap": mb_hybrid_umap,
               "title": "MiraBest labels",
               }
    predictions_data = {"fri_umap": predictions_fri_umap,
                        "frii_umap": predictions_frii_umap,
                        "hybrid_umap": predictions_hybrid_umap,
                        "title": "Model classifications",
                        }
    
    plot_data = [mb_data, predictions_data]

    # Plot embedding
    plot_embedding(save_dir + "/embedding.png", plot_data)

    
def run_finetuning(config, encoder, datamodule, logger):
    checkpoint = ModelCheckpoint(
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        # dirpath=config["files"] / config["run_id"] / "finetuning",
        # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt.
        filename="{epoch}",  # filename may not work here TODO
        save_weights_only=True,
        # save_top_k=3,
    )

    callbacks = []

    early_stop_callback = EarlyStopping(
        monitor="finetuning/train_loss", min_delta=0.00, patience=3, verbose=True, mode="min"
    )

    if config["finetune"]["early_stopping"]:
        callbacks.append(early_stop_callback)

    ## Initialise pytorch lightning trainer ##
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config["finetune"]["n_epochs"],
        **config["trainer"],
    )

    # Initialize linear head
    if config["finetune"]["head"] == "linear":
        # head = LogisticRegression(input_dim=encoder.dim, output_dim=config["finetune"]["n_classes"])
        head = "linear"

    elif config["finetune"]["head"] == "mlp":
        head = MLPHead(
            input_dim=encoder.dim,
            depth=config["finetune"]["depth"],
            width=config["finetune"]["width"],
            output_dim=config["finetune"]["n_classes"],
        )
    else:
        raise ValueError("Head must be either linear or mlp")

    model = FineTune(
        encoder,
        head,
        dim=encoder.dim,
        n_classes=config["finetune"]["n_classes"],
        n_epochs=config["finetune"]["n_epochs"],
        n_layers=config["finetune"]["n_layers"],
        batch_size=config["finetune"]["batch_size"],
        lr_decay=config["finetune"]["lr_decay"],
        seed=config["seed"],
        head_type=config["finetune"]["head"],
    )

    trainer.fit(model, datamodule)

    trainer.test(model, dataloaders=datamodule)

    return checkpoint, model


def set_grads(module, value: bool):
    for params in module.parameters():
        params.requires_grad = value


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Load paths
    paths = Path_Handler()._dict()

    # Load up finetuning config
    config_finetune = load_config_finetune()

    ## Run finetuning ##
    for seed in range(config_finetune["finetune"]["iterations"]):
        # for seed in range(1, 10):

        if config_finetune["finetune"]["run_id"].lower() != "none":
            experiment_dir = paths["files"] / config_finetune["finetune"]["run_id"] / "checkpoints"
            model = BYOL.load_from_checkpoint(experiment_dir / "last.ckpt")
        else:
            model = BYOL.load_from_checkpoint("byol.ckpt")

        ## Load up config from model to save correct hparams for easy logging ##
        config = model.config
        config.update(config_finetune)
        config["finetune"]["dim"] = model.encoder.dim

        # Compatibility with old style config
        if config["augmentations"]["center_crop"] is True:
            config["augmentations"]["center_crop"] = config["augmentations"]["center_crop_size"]

        project_name = config_finetune["finetune"]["wandb_project"]

        config["finetune"]["seed"] = seed
        pl.seed_everything(seed)

        # Initiate wandb logging
        wandb.init(project=project_name, config=config)

        logger = pl.loggers.WandbLogger(
            project=project_name,
            save_dir=paths["files"] / "finetune" / str(wandb.run.id),
            reinit=True,
            config=config,
        )

        finetune_datamodule = RGZ_DataModule_Finetune(
            paths["mb"],
            batch_size=config["finetune"]["batch_size"],
            center_crop=config["augmentations"]["center_crop"],
            val_size=config["finetune"]["val_size"],
            num_workers=config["dataloading"]["num_workers"],
            prefetch_factor=config["dataloading"]["prefetch_factor"],
            pin_memory=config["dataloading"]["pin_memory"],
            seed=config["finetune"]["seed"],
        )
        run_finetuning(config, model.encoder, finetune_datamodule, logger)
        run_post_evaluation(str(wandb.run.id))
        logger.experiment.finish()
        wandb.finish()


if __name__ == "__main__":
    main()
