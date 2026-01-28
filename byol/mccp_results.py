import argparse
import logging
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
import torchvision.transforms as T

from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
from umap import UMAP

from config import load_config_finetune
from datamodules import RGZ_DataModule_Finetune
from datasets import RGZ108k
from finetuning import MLPHead, FineTune
from models import BYOL
from paths import Path_Handler

parser = argparse.ArgumentParser()
parser.add_argument("--wandb-group", type=str, required=True)
parser.add_argument("--calibration-batch", type=int, required=True)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

class Reducer:
    
    def __init__(self, encoder, PCA_COMPONENTS, UMAP_N_NEIGHBOURS, UMAP_MIN_DIST, METRIC, embedding=None, seed=42):
        self.encoder = encoder
        self.pca = PCA(n_components=PCA_COMPONENTS, random_state=seed)
        self.umap = UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBOURS,
            min_dist=UMAP_MIN_DIST,
            metric=METRIC,
            random_state=seed,
        )
        if embedding is not None:
            if not os.path.exists(embedding):
                logger.info("Specified embedding file does not exist - will compute embedding")
                self.embedded = False
            else:
                self.filename = embedding
                self.embedded = True
        else:
            self.embedded = False

    def read_file(self):
        logger.info("Reading embedding from file: {}".format(self.filename))
        df = pd.read_parquet(self.filename)
        features = df[[f"feat_{i}" for i in range(512)]].values
        if 'target' in df.columns:
            targets = df["target"].values
        else:
            targets = np.ones(features.shape[0])
        return features, targets

    def write_file(self, filename):
        cols = [f"feat_{i}" for i in range(512)]
        logger.info(self.features.shape, self.targets.shape)
        df = pd.DataFrame(data=self.features, columns=cols)
        df.to_parquet(filename)
        return 
    
    def embed_dataset(self, data, batch_size=400):
        train_loader = DataLoader(data, batch_size, shuffle=False)
        device = next(self.encoder.parameters()).device
        feature_bank = []
        for data in tqdm(train_loader):
            # Load data and move to correct device
            if len(data) == 2:
                x, _ = data
            elif len(data) == 3:
                x, _, _ = data
            else: x = data
            x_enc = self.encoder(x.to(device))
            feature_bank.append(x_enc.squeeze().detach().cpu())
        # Save full feature bank for validation epoch
        features = torch.cat(feature_bank)
        targets = np.ones(features.shape[0])
        return features, targets

    def fit(self, data=None):
        logger.info("Fitting reducer ...")
        if data!=None: features, targets = self.embed_dataset(data)
        if data==None and self.embedded: features, targets = self.read_file()
        if data==None and not self.embedded:
            logger.error("No data/embedding provided - exiting")
            return
        self.features = features
        self.targets = targets
        self.pca.fit(self.features)
        self.umap.fit(self.pca.transform(self.features))
        logger.info("Finished fitting reducer.")
        return

    def transform(self, data=None):
        logger.info("Performing transformation ...")
        if data!=None: 
            x, _ = self.embed_dataset(data)
        elif data==None and hasattr(self, "features"): 
            x = self.features
        elif data==None and not hasattr(self, "features") and self.embedded: 
            x, _ = self.read_file()  
        elif data==None and not hasattr(self, "features") and not self.embedded: 
            logger.error("No data/embedding provided - exiting")
            return
        x = self.pca.transform(x)
        x = self.umap.transform(x)
        logger.info("Finished transformation.")
        return x

    def transform_pca(self, data):
        x, _ = self.embed_dataset(data)
        x = self.pca.transform(x)
        return x
    
def get_reducer(model, save_dir, run_name, transform):
    paths = Path_Handler()._dict()
    encoder = model.encoder
    encoder.eval()
    reducer_path = os.path.join(save_dir, run_name + "_reducer.pkl")
    if os.path.isfile(reducer_path):
        reducer = pickle.load((open(reducer_path, "rb")))
    else:
        PCA_COMPONENTS = 200
        UMAP_N_NEIGHBOURS = 75
        UMAP_MIN_DIST = 0.01
        METRIC = "cosine"
        reducer = Reducer(encoder, PCA_COMPONENTS, UMAP_N_NEIGHBOURS, UMAP_MIN_DIST, METRIC)
        rgz = RGZ108k(
        paths["rgz"],
        train=True,
        transform=transform,
        download=False,
        remove_duplicates=False,
        cut_threshold=25,
        mb_cut=True,
        )
        reducer.fit(rgz)
        pickle.dump(reducer, open(reducer_path, "wb"))
    return reducer

def get_deepest_file(root_dir):
    deepest_file = None
    max_depth = -1
    for dirpath, dirnames, filenames in os.walk(root_dir):
        depth = dirpath.count(os.sep)
        for filename in filenames:
            if depth > max_depth:
                max_depth = depth
                deepest_file = os.path.join(dirpath, filename)
    return deepest_file

def load_checkpoint(ckpt_path, encoder, config):
    # Recreate the head based on your configuration
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

@torch.no_grad()
def get_results(model, dataloader, split_name, reducer):
    device = next(model.parameters()).device
    model.eval()
    results = []
    for x, _, meta in tqdm(dataloader, desc=f"Getting {split_name} results"):
        x = x.to(device)
        filenames = meta["filename"]
        label_dists = meta["label_dist"]
        mb_labels = meta["mb_label"]
        feats = model.encoder(x)
        logits = model.head(feats)
        logits = logits.cpu().numpy()
        for i in range(len(filenames)):
            label_dist = label_dists[i]
            if torch.is_tensor(label_dist):
                label_dist = label_dist.cpu().numpy()
            elif isinstance(label_dist, list):
                label_dist = np.array(label_dist, dtype=np.float32)
            results.append({
                "filename": filenames[i],
                "split": split_name,
                "label_dist": label_dist,
                "mb_label": mb_labels[i],
                "logits": logits[i],
            })
    umap_coords = reducer.transform(dataloader.dataset)
    for result, coords in zip(results, umap_coords):
        result["umap"] = coords
    return results

def main():
    paths = Path_Handler()._dict()
    finetune_config = load_config_finetune()

    out_dir = paths["files"] / "mccp" / args.wandb_group
    out_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.wandb_group + "_CB" + str(args.calibration_batch + 1)
    ckpt_dir = paths["files"] / "finetune" / run_name / "MCCP-BYOL"
    ckpt_path = get_deepest_file(ckpt_dir)

    byol_model = BYOL.load_from_checkpoint("byol.ckpt")
    config = byol_model.config
    config.update(finetune_config)
    encoder = byol_model.encoder

    datamodule = RGZ_DataModule_Finetune(
        paths["mb"],
        batch_size=1024,
        center_crop=config["augmentations"]["center_crop"],
        val_size=config["finetune"]["val_size"],
        num_workers=config["dataloading"]["num_workers"],
        prefetch_factor=config["dataloading"]["prefetch_factor"],
        pin_memory=config["dataloading"]["pin_memory"],
        seed = config["finetune"]["seed"] + args.calibration_batch,
        calibration_batch=args.calibration_batch,
    )
    datamodule.setup()
    
    mu, sig = config["data"]["mu"], config["data"]["sig"]
    transform = T.Compose(
        [
            T.CenterCrop(70),
            T.ToTensor(),
            T.Normalize((mu,), (sig,)),
        ]
    )
    
    model = load_checkpoint(ckpt_path, encoder, config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    reducer = get_reducer(
        model,
        save_dir=paths["files"] / "mccp" / args.wandb_group,
        run_name=f"CB_{args.calibration_batch + 1}",
        transform=transform,
    )

    dataloaders = {
        "train": datamodule.val_dataloader(),
        "calibration": datamodule.calibration_dataloader(),
        "test_conf": datamodule.test_dataloader()[0],
        "test_uncert": datamodule.test_dataloader()[1],
    }

    results = []
    for split, dataloader in dataloaders.items():
        split_results = get_results(model, dataloader, split, reducer)
        results.extend(split_results)

    out_file = out_dir / f"CB{args.calibration_batch + 1}_results.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved results to {out_file}.")

if __name__ == "__main__":
    main()