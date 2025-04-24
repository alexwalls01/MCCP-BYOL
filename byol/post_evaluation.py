import pylab
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from umap import UMAP
from matplotlib import gridspec
from matplotlib import colors
import json
import torchvision.transforms as T
import pickle
import torch.nn.functional as F

from paths import Path_Handler
from config import load_config, update_config, load_config_finetune, load_config_evaluation
from models import BYOL
from datamodules import RGZ_DataModule_Finetune
from datasets import MBFRFull, RGZ108k
from finetuning import MLPHead, FineTune

RUN_ID = os.environ.get("RUN_ID", "0")

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
                print("Specified embedding file does not exist - will compute embedding")
                self.embedded = False
            else:
                self.filename = embedding
                self.embedded = True
        else:
            self.embedded = False

    def read_file(self):

        print("Reading embedding from file: {}".format(self.filename))
        
        df = pd.read_parquet(self.filename)
        features = df[[f"feat_{i}" for i in range(512)]].values
        if 'target' in df.columns:
            targets = df["target"].values
        else:
            targets = np.ones(features.shape[0])
        
        return features, targets

    def write_file(self, filename):

        cols = [f"feat_{i}" for i in range(512)]
        print(self.features.shape, self.targets.shape)
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
        
        print("Fitting reducer")

        if data!=None: features, targets = self.embed_dataset(data)
        if data==None and self.embedded: features, targets = self.read_file()
        if data==None and not self.embedded:
            print("No data/embedding provided - exiting")
            return
         
        self.features = features
        self.targets = targets

        self.pca.fit(self.features)
        self.umap.fit(self.pca.transform(self.features))

        return

    def transform(self, data=None):
        
        print("Performing transformation")

        if data!=None: 
            x, _ = self.embed_dataset(data)
        elif data==None and hasattr(self, 'features'): 
            x = self.features
        elif data==None and not hasattr(self, 'features') and self.embedded: 
            x, _ = self.read_file()  
        elif data==None and not hasattr(self, 'features') and not self.embedded: 
            print("No data/embedding provided - exiting")
            return
        
        x = self.pca.transform(x)
        x = self.umap.transform(x)
        return x

    def transform_pca(self, data):
        x, _ = self.embed_dataset(data)
        x = self.pca.transform(x)
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
    elif stage == "calibration":
        dataloader = datamodule.calibration_dataloader()
    else:
        raise ValueError("Unsupported dataloader stage.")
    return dataloader

def create_calibration_set(model, label_dist, RA_dec, m):

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    prediction_loader = load_dataloader("calibration")
    batch_predictions = trainer.predict(model, dataloaders=prediction_loader)

    predictions = []
    for batch in batch_predictions:
        # Ensure logits are in a list format.
        logits_list = batch["logits"].tolist() if isinstance(batch["logits"], torch.Tensor) else batch["logits"]
        for filename, logit in zip(batch["filenames"], logits_list):
            predictions.append({"filename": filename, "logits": logit})
    
    mbfr_annotated = MBFRFull().with_annotator_labels(label_dist, RA_dec)
    for sample in predictions:
        filename = sample["filename"]
        # Extract the target class using the method provided by MBFRFull.
        target = mbfr_annotated.get_target(filename)
        dist = mbfr_annotated.get_dist(filename)
        sample["class"] = target
        sample["label_dist"] = dist

    calibration_set = []
    for sample in predictions:
        calibration_set.append(sample)
        for i in range(m):
            # Sample new label according to label distribution
            sampled_target = np.random.choice([0, 1, 2], p=sample["label_dist"])
            duplicate_sample = sample
            duplicate_sample["class"] = sampled_target
            calibration_set.append(duplicate_sample)

    return calibration_set

def calculate_threshold(calibration_set, alpha):
    non_conformity_scores = []
    for sample in calibration_set:
        softmax = F.softmax(torch.tensor(sample["logits"]), dim=0).tolist()
        target = sample["class"]
        score = 1 - softmax[target]
        non_conformity_scores.append(score)
    non_conformity_scores = np.array(non_conformity_scores)
    threshold = np.quantile(non_conformity_scores, 1 - alpha)
    return threshold




def create_subplots(n):

    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig = pylab.figure(constrained_layout=True)
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    axes = []
    # Create axes for all full rows except the last one
    for i in range(rows - 1):
        for j in range(cols):
            ax = fig.add_subplot(gs[i, j])
            ax.set_aspect('equal', adjustable='box')
            ax.set_box_aspect(1) # Make all subplots square
            axes.append(ax)
    
    # Centre the subplots in the last row
    num_last = n - (rows - 1) * cols  # Number of axes needed in the last row
    offset = (cols - num_last) // 2   # Left offset to centre the last row
    for j in range(num_last):
        ax = fig.add_subplot(gs[rows - 1, offset + j])
        ax.set_aspect('equal', adjustable='box')
        ax.set_box_aspect(1) # Make all subplots square
        axes.append(ax)
    
    return fig, axes


def plot_embedding(fig_path, plot_data):

    fig, axes = create_subplots(len(plot_data))
    marker_size = 2
    xmin = np.min(plot_data[0]["umap"][:, 0]) - 0.5
    xmax = np.max(plot_data[0]["umap"][:, 0]) + 0.5
    ymin = np.min(plot_data[0]["umap"][:, 1]) - 0.5
    ymax = np.max(plot_data[0]["umap"][:, 1]) + 0.5

    cmap = colors.LinearSegmentedColormap.from_list("", ["#648FFF","#DC267F","#FFB000"])

    #for ax in axes[1:]:
        #ax.sharex(axes[0])
        #ax.sharey(axes[0])

    for index, ax in enumerate(axes):
        fr_classes = []
        for label in plot_data[index]["labels"]:
            if label == 0:
                fr_classes.append("FRI")
            elif label == 1:
                fr_classes.append("FRII")
            else:
                fr_classes.append("Hybrid")

        clset = set(zip(plot_data[index]["labels"], fr_classes))
        ax.set_title(plot_data[index]["title"])
        sc = ax.scatter(plot_data[index]["umap"][:, 0], plot_data[index]["umap"][:, 1], c=plot_data[index]["labels"], cmap=cmap, alpha=0.5, s=marker_size)
        handles = [pylab.plot([],color=sc.get_cmap()(sc.norm(c)),ls="", marker="o")[0] for c,l in clset]
        labels = [l for c,l in clset]
        ax.legend(handles, labels)
        #ax.set_xlim(xmin, xmax)
        #ax.set_ylim(ymin, ymax)
        ax.set_xlabel("UMAP x")
        ax.set_ylabel("UMAP y")
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

    pylab.gca().set_aspect("equal", "datalim")

    fig.savefig(fig_path, bbox_inches="tight", dpi=600)

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

    byol_model = BYOL.load_from_checkpoint("byol.ckpt")
    config = byol_model.config
    mu, sig = config["data"]["mu"], config["data"]["sig"]
    label_dist=np.load(config["conformal_prediction"]["label_dist"])
    RA_dec=np.load(config["conformal_prediction"]["RA_dec"])

    encoder = model.encoder
    encoder.eval()

    transform = T.Compose(
        [
            T.CenterCrop(70),
            T.ToTensor(),
            T.Normalize((mu,), (sig,)),
        ]
    )

    reducer_path = os.path.join(save_dir, run_id + "_reducer.pkl")
    if os.path.isfile(reducer_path):
        reducer = pickle.load((open(reducer_path, 'rb')))
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
        pickle.dump(reducer, open(reducer_path, 'wb'))

    # Get umap embeddings for data with original MiraBest labels
    mb_test = MBFRFull(root=paths["mb"],
                  train=False,
                  transform=transform,
                  download=False,
                  aug_type="torchvision"
                  )
    mb_train = MBFRFull(root=paths["mb"],
                  train=True,
                  transform=transform,
                  download=False,
                  aug_type="torchvision"
                  )
    mb_test_pseudo = mb_test.with_pseudo_labels(model)
    mb_train_pseudo = mb_train.with_pseudo_labels(model)
    
    mb_test_labels = np.array(mb_test.targets)
    mb_test_preds = np.array(mb_test_pseudo.targets)
    mb_train_labels = np.array(mb_train.targets)
    mb_train_preds = np.array(mb_train_pseudo.targets)

    # Get umap embeddings for data with model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    mb_test_umap = reducer.transform(mb_test)
    mb_train_umap = reducer.transform(mb_train)

    # Put together data to plot
    plot_data_orig = {"umap": np.vstack((mb_train_umap, mb_test_umap)),
                      "labels": np.concatenate((mb_train_labels, mb_test_labels), axis=0),
                      "title": "MiraBest labels",
                      }
    plot_data_preds = {"umap": np.vstack((mb_train_umap, mb_test_umap)),
                       "labels": np.concatenate((mb_train_preds, mb_test_preds), axis=0),
                       "title": "Model predictions",
                       }
    plot_data_orig_test = {"umap": mb_test_umap,
                           "labels": mb_test_labels,
                           "title": "MiraBest labels",
                           }
    plot_data_preds_test = {"umap": mb_test_umap,
                            "labels": mb_test_preds,
                            "title": "Model predictions",
                            }
    plot_data = [plot_data_orig, plot_data_preds]
    plot_data_test = [plot_data_orig_test, plot_data_preds_test]

    # Plot embedding
    plot_embedding(save_dir + "/" + run_id + "_embedding_test.png", plot_data_test)
    plot_embedding(save_dir + "/" + run_id + "_embedding.png", plot_data)

def main():
    run_post_evaluation(RUN_ID)

if __name__ == "__main__":
    main()