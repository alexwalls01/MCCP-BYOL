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
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

from paths import Path_Handler
from config import load_config, update_config, load_config_finetune, load_config_evaluation
from models import BYOL
from datamodules import RGZ_DataModule_Finetune
from datasets import MBFRFull, RGZ108k, MBFRConfidentNoHybrids
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

def load_dataloader(stage, label_dist=None, RA_dec=None):
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
    datamodule.setup(stage=stage, label_dist=label_dist, RA_dec=RA_dec)
    if stage == "test":
        dataloader = datamodule.test_dataloader()
    elif stage == "val":
        dataloader = datamodule.val_dataloader()
    elif stage == "calibration":
        dataloader = datamodule.calibration_dataloader()
    elif stage == "test_conf":
        dataloader = datamodule.test_conf_dataloader()
    else:
        raise ValueError("Unsupported dataloader stage.")
    return dataloader

def create_calibration_set(model, mb_calibration, m, label_dist, RA_dec):

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    prediction_loader = load_dataloader("calibration", label_dist=label_dist, RA_dec=RA_dec)
    batch_predictions = trainer.predict(model, dataloaders=prediction_loader)

    predictions = []
    for batch in batch_predictions:
        # Ensure logits are in a list format.
        logits_list = batch["logits"].tolist() if isinstance(batch["logits"], torch.Tensor) else batch["logits"]
        for filename, logit in zip(batch["filenames"], logits_list):
            predictions.append({"filename": filename, "logits": logit})
    
    for sample in predictions:
        filename = sample["filename"]
        # Extract the target class using the method provided by MBFRFull.
        target = mb_calibration.get_target(filename)
        dist = mb_calibration.get_dist(filename)
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

def create_class_conditional_calibration_sets(model, mb_calibration, m, label_dist, RA_dec):

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    prediction_loader = load_dataloader("calibration", label_dist=label_dist, RA_dec=RA_dec)
    batch_predictions = trainer.predict(model, dataloaders=prediction_loader)

    predictions = []
    for batch in batch_predictions:
        # Ensure logits are in a list format.
        logits_list = batch["logits"].tolist() if isinstance(batch["logits"], torch.Tensor) else batch["logits"]
        for filename, logit in zip(batch["filenames"], logits_list):
            predictions.append({"filename": filename, "logits": logit})
    
    for sample in predictions:
        filename = sample["filename"]
        # Extract the target class using the method provided by MBFRFull.
        target = mb_calibration.get_target(filename)
        dist = mb_calibration.get_dist(filename)
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
    
    FRI_set = []
    FRII_set = []
    hybrid_set = []
    for sample in calibration_set:
        if sample["class"] == 0:
            FRI_set.append(sample)
        elif sample["class"] == 1:
            FRII_set.append(sample)
        else:
            hybrid_set.append(sample)

    return FRI_set, FRII_set, hybrid_set

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

def create_prediction_sets(model, mb_test, threshold, label_dist, RA_dec, stage):
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    prediction_loader = load_dataloader(stage, label_dist=label_dist, RA_dec=RA_dec)
    batch_predictions = trainer.predict(model, dataloaders=prediction_loader)
    if stage == "test":
        # Because the test set has confident and uncertain subsets
        batch_predictions_1 = batch_predictions[0]
        batch_predictions_2 = batch_predictions[1]
        predictions = []
        for batch in batch_predictions_1:
            # Ensure logits are in a list format.
            logits_list = batch["logits"].tolist() if isinstance(batch["logits"], torch.Tensor) else batch["logits"]
            for filename, logit in zip(batch["filenames"], logits_list):
                predictions.append({"filename": filename, "logits": logit})
        for batch in batch_predictions_2:
            # Ensure logits are in a list format.
            logits_list = batch["logits"].tolist() if isinstance(batch["logits"], torch.Tensor) else batch["logits"]
            for filename, logit in zip(batch["filenames"], logits_list):
                predictions.append({"filename": filename, "logits": logit})
    else:
        predictions = []
        for batch in batch_predictions:
            # Ensure logits are in a list format.
            logits_list = batch["logits"].tolist() if isinstance(batch["logits"], torch.Tensor) else batch["logits"]
            for filename, logit in zip(batch["filenames"], logits_list):
                predictions.append({"filename": filename, "logits": logit})

    for sample in predictions:
        filename = sample["filename"]
        target = mb_test.get_target(filename)
        dist = mb_test.get_dist(filename)
        sample["class"] = target
        sample["label_dist"] = dist
    
    for sample in predictions:
        prediction_set = []
        softmax = F.softmax(torch.tensor(sample["logits"]), dim=0).tolist()
        for i in range(3):
            if 1 - softmax[i] <= threshold:
                prediction_set.append(softmax[i])
            else:
                prediction_set.append(0)
        sample["prediction_set"] = prediction_set
    
    return predictions

def calculate_class_conditional_scores(model, mb_test, FRI_set, FRII_set, hybrid_set, label_dist, RA_dec, stage):
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    prediction_loader = load_dataloader(stage, label_dist=label_dist, RA_dec=RA_dec)
    batch_predictions = trainer.predict(model, dataloaders=prediction_loader)
    if stage == "test":
        # Because the test set has confident and uncertain subsets
        batch_predictions_1 = batch_predictions[0]
        batch_predictions_2 = batch_predictions[1]
        predictions = []
        for batch in batch_predictions_1:
            # Ensure logits are in a list format
            logits_list = batch["logits"].tolist() if isinstance(batch["logits"], torch.Tensor) else batch["logits"]
            for filename, logit in zip(batch["filenames"], logits_list):
                predictions.append({"filename": filename, "logits": logit})
        for batch in batch_predictions_2:
            # Ensure logits are in a list format
            logits_list = batch["logits"].tolist() if isinstance(batch["logits"], torch.Tensor) else batch["logits"]
            for filename, logit in zip(batch["filenames"], logits_list):
                predictions.append({"filename": filename, "logits": logit})
    else:
        predictions = []
        for batch in batch_predictions:
            # Ensure logits are in a list format
            logits_list = batch["logits"].tolist() if isinstance(batch["logits"], torch.Tensor) else batch["logits"]
            for filename, logit in zip(batch["filenames"], logits_list):
                predictions.append({"filename": filename, "logits": logit})
    
    scores = []
    for sample in predictions:
        filename = sample["filename"]
        sample["class"] = mb_test.get_target(filename)
        sample["label_dist"] = mb_test.get_dist(filename)

        if np.argmax(sample["class"]) == 0:
            calibration_set = FRI_set
        elif np.argmax(sample["class"]) == 1:
            calibration_set = FRII_set
        else:
            calibration_set = hybrid_set

        find_score = True
        alphas = np.arange(0, 1, 0.01)
        idx = -1
        while find_score is True:
            idx += 1
            threshold = calculate_threshold(calibration_set, alphas[idx])
            softmax = F.softmax(torch.tensor(sample["logits"]), dim=0).tolist()
            prediction_set = []
            for i in range(3):
                if 1 - softmax[i] <= threshold:
                    prediction_set.append(softmax[i])
                else:
                    prediction_set.append(0)
            size = 3 - prediction_set.count(0)
            if size == 1:
                find_score = False
            if idx == len(alphas) - 1:
                find_score = False
        scores.append(alphas[idx-1])
    return np.array(scores)

def test_alpha(model, mb_calibration, mb_test, m, fig_path, label_dist, RA_dec):
    if m == 1:
        alphas = np.arange(0, 1, 0.01)
    else:
        alphas = np.arange(0, 0.5, 0.01)
    fig, ax = pylab.subplots(constrained_layout=True)

    calibration_set = create_calibration_set(model, mb_calibration, m, label_dist, RA_dec)

    empty = []
    single = []
    double = []
    full = []
    for alpha in alphas:

        threshold = calculate_threshold(calibration_set, alpha)
        predictions = create_prediction_sets(model, mb_test, threshold, label_dist, RA_dec, "test")

        prediction_sets = []
        for sample in predictions:
            prediction_sets.append(sample["prediction_set"])

        sizes = [0,0,0,0]
        for set in prediction_sets:
            size = 3 - set.count(0)
            sizes[size] += 1
        empty.append(sizes[0])
        single.append(sizes[1])
        double.append(sizes[2])
        full.append(sizes[3])
    
    empty = np.flip(np.array(empty))
    single = np.flip(np.array(single))
    double = np.flip(np.array(double))
    full = np.flip(np.array(full))
    alphas = np.flip(alphas)

    empty_percent = (empty * 100) / (len(prediction_sets))
    single_percent = (single * 100) / (len(prediction_sets))
    double_percent = (double * 100) / (len(prediction_sets))
    full_percent = (full * 100) / (len(prediction_sets))
    
    if m > 1:
        alphas = 2 * alphas
        ax.set_xlabel(r'1 - 2$\alpha$', fontsize=18)
    else:
        ax.set_xlabel(r'1 - $\alpha$', fontsize=18)
    ax.set_ylabel("Percentage of test samples", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.plot(1-alphas, empty_percent, label="Empty", c="#648FFF")
    ax.plot(1-alphas, single_percent, label="1", c="#DC267F")
    ax.plot(1-alphas, double_percent, label="2", c="#FE6100")
    ax.plot(1-alphas, full_percent, label="3", c="#FFB000")
    ax.legend(title="Prediction set size")
    ax.set_xlim(-0.025, 1.025)
    ax.set_box_aspect(1)
    fig.savefig(fig_path, bbox_inches="tight", dpi=600)


def plot_embedding(fig_path, plot_data):

    fig, ax = pylab.subplots(constrained_layout=True)

    marker_size = 15
    xmin = np.min(plot_data["umap"][:, 0]) - 0.5
    xmax = np.max(plot_data["umap"][:, 0]) + 0.5
    ymin = np.min(plot_data["umap"][:, 1]) - 0.5
    ymax = np.max(plot_data["umap"][:, 1]) + 0.5

    cmap = colors.LinearSegmentedColormap.from_list("", ["#648FFF","#DC267F","#FFB000"])

    fr_classes = []
    for label in plot_data["labels"]:
        if label == 0:
            fr_classes.append("FRI")
        elif label == 1:
            fr_classes.append("FRII")
        else:
            fr_classes.append("Hybrid")

    sc = ax.scatter(plot_data["umap"][:, 0], plot_data["umap"][:, 1], c=plot_data["labels"], cmap=cmap, ec=None, alpha=0.5, s=marker_size)
    cl_pairs = list(zip(plot_data["labels"], fr_classes))
    cl_unique = sorted(set(cl_pairs), key=lambda x: x[0])
    handles = [
        Line2D([], [], color=cmap(sc.norm(code)), marker="o", ls="")
        for code, lbl in cl_unique
    ]
    labels = [lbl for code, lbl in cl_unique]
    ax.legend(handles, labels, fontsize=14)
    ax.set_xlabel("UMAP x", fontsize=18)
    ax.set_ylabel("UMAP y", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    pylab.gca().set_aspect("equal", "datalim")

    fig.savefig(fig_path, bbox_inches="tight", dpi=600)

def plot_embedding_uncertainty(fig_path, plot_data):

    fig, ax = pylab.subplots(constrained_layout=True)

    marker_size = 15
    xmin = np.min(plot_data["umap"][:, 0]) - 0.5
    xmax = np.max(plot_data["umap"][:, 0]) + 0.5
    ymin = np.min(plot_data["umap"][:, 1]) - 0.5
    ymax = np.max(plot_data["umap"][:, 1]) + 0.5

    normalize = colors.Normalize(vmin=np.min(plot_data["uncertainty"]), vmax=np.max(plot_data["uncertainty"]))
    sc = ax.scatter(plot_data["umap"][:, 0], plot_data["umap"][:, 1], c=plot_data["uncertainty"], cmap='viridis', norm=normalize, ec=None, alpha=0.75, s=marker_size)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label=plot_data["cbar_label"], size=18)
    if plot_data["cbar_ticks"] is not None:
        cbar.set_ticks(ticks=plot_data["cbar_ticks"])
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel("UMAP x", fontsize=18)
    ax.set_ylabel("UMAP y", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    pylab.gca().set_aspect("equal", "datalim")

    fig.savefig(fig_path, bbox_inches="tight", dpi=600)

def violin_plot(fig_path, entropy, prediction_set_size, ylabel):

    fig, ax = pylab.subplots(constrained_layout=True)

    size_one_indices = [i for i, val in enumerate(prediction_set_size) if val == 1]
    size_two_indices = [i for i, val in enumerate(prediction_set_size) if val == 2]
    size_three_indices = [i for i, val in enumerate(prediction_set_size) if val == 3]

    plot_data = [entropy[size_one_indices], entropy[size_two_indices], entropy[size_three_indices]]
    ax.violinplot(plot_data, showmeans=False, showmedians=True)
    ax.set_xlabel("Prediction set size", fontsize=18)
    ax.set_xticks([y + 1 for y in range(len(plot_data))], labels=['1', '2', '3'])
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.yaxis.grid(True)
    #ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    #pylab.gca().set_aspect("equal", "datalim")
    ax.set_ylim(-0.1, 1.1)
    fig.savefig(fig_path, bbox_inches="tight", dpi=600)

def scatter_plot(fig_path, entropy, scores, ylabel):

    fig, ax = pylab.subplots(constrained_layout=True)

    ax.scatter(scores, entropy, marker="x")
    ax.set_xlabel(r'$\alpha$', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_xlim(np.min(scores)-0.05, 0.45)
    ax.set_ylim(np.min(entropy)-0.05, np.max(entropy)+0.05)
    #ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    #pylab.gca().set_aspect("equal", "datalim")
    fig.savefig(fig_path, bbox_inches="tight", dpi=600)


def run_post_evaluation(run_id):

    paths = Path_Handler()._dict()

    eval_config = load_config_evaluation()
    finetune_config = load_config_finetune()

    ckpt_folder = eval_config['ckpt_folder']
    wandb_project = finetune_config['finetune']['wandb_project']
    save_dir = eval_config['save_dir'] + "/" + wandb_project
    os.makedirs(save_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_folder, run_id, wandb_project, run_id, "checkpoints", "epoch=299-step=3000.ckpt")
    model = load_checkpoint(ckpt_path)

    byol_model = BYOL.load_from_checkpoint("byol.ckpt")
    config = byol_model.config
    config.update(finetune_config)
    mu, sig = config["data"]["mu"], config["data"]["sig"]
    label_dist = np.load(config["conformal_prediction"]["label_dist"])
    RA_dec = np.load(config["conformal_prediction"]["RA_dec"])
    mb_conf_entropy = np.genfromtxt(config["conformal_prediction"]["hmc_data"], delimiter=',', skip_header=1)[:,1]

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
    mb_conf = MBFRConfidentNoHybrids(root=paths["mb"],
                                     train=False,
                                     transform=transform,
                                     download=False,
                                     aug_type="torchvision"
                                     )
    mb_test_pseudo = mb_test.with_pseudo_labels(model)
    mb_train_pseudo = mb_train.with_pseudo_labels(model)
    mb_conf_pseudo = mb_conf.with_pseudo_labels(model)
    mb_test_annotator = mb_test.with_annotator_labels(label_dist, RA_dec)
    mb_train_annotator = mb_train.with_annotator_labels(label_dist, RA_dec)
    mb_conf_annotator = mb_conf.with_annotator_labels(label_dist, RA_dec)
    
    mb_test_labels = np.array(mb_test.targets)
    mb_test_preds = np.array(mb_test_pseudo.targets)
    mb_test_annotations = np.array(mb_test_annotator.targets)
    mb_train_labels = np.array(mb_train.targets)
    mb_train_preds = np.array(mb_train_pseudo.targets)
    mb_train_annotations = np.array(mb_train_annotator.targets)
    mb_conf_labels = np.array(mb_conf.targets)
    mb_conf_preds = np.array(mb_conf_pseudo.targets)
    mb_conf_annotations = np.array(mb_conf_annotator.targets)

    # Get relevant uncertainty measures
    annotator_entropy_train = mb_train_annotator.get_annotator_entropy()
    annotator_entropy_test = mb_test_annotator.get_annotator_entropy()
    annotator_entropy_conf = mb_conf_annotator.get_annotator_entropy()

    # Get umap embeddings for data with model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    mb_test_umap = reducer.transform(mb_test)
    mb_train_umap = reducer.transform(mb_train)
    mb_conf_umap = reducer.transform(mb_conf)

    # Put together data to plot
    plot_data_orig = {"umap": np.vstack((mb_train_umap, mb_test_umap)),
                      "labels": np.concatenate((mb_train_labels, mb_test_labels), axis=0),
                      }
    plot_data_preds = {"umap": np.vstack((mb_train_umap, mb_test_umap)),
                       "labels": np.concatenate((mb_train_preds, mb_test_preds), axis=0),
                       }
    plot_data_annotator = {"umap": np.vstack((mb_train_umap, mb_test_umap)),
                           "labels": np.concatenate((mb_train_annotations, mb_test_annotations), axis=0),
                           "uncertainty": np.concatenate((annotator_entropy_train, annotator_entropy_test)),
                           "cbar_label": "Entropy of label distribution",
                           "cbar_ticks": None
                           }
    plot_data_annotator_conf = {"umap": mb_conf_umap,
                                "labels": mb_conf_annotations,
                                "uncertainty": annotator_entropy_conf,
                                "cbar_label": "Entropy of label distribution",
                                "cbar_ticks": None}
    plot_data_hmc_conf = {"umap": mb_conf_umap,
                                "labels": mb_conf_annotations,
                                "uncertainty": mb_conf_entropy,
                                "cbar_label": "Predictive entropy",
                                "cbar_ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
    
    # Plot embedding
    plot_embedding(save_dir + "/" + run_id + "_embedding_MiraBest.png", plot_data_orig)
    plot_embedding(save_dir + "/" + run_id + "_embedding_predictions.png", plot_data_preds)
    plot_embedding(save_dir + "/" + run_id + "_embedding_annotations.png", plot_data_annotator)

    plot_embedding_uncertainty(save_dir + "/" + run_id + "_embedding_annotator_entropy.png", plot_data_annotator)
    plot_embedding_uncertainty(save_dir + "/" + run_id + "_embedding_annotator_entropy_conf.png", plot_data_annotator_conf)
    plot_embedding_uncertainty(save_dir + "/" + run_id + "_embedding_hmc_conf.png", plot_data_hmc_conf)

    # Monte Carlo conformal prediction
    mb_calibration = MBFRFull(root=paths["mb"],
                              train=False,
                              calibration=True,
                              transform=transform,
                              download=False,
                              aug_type="torchvision"
                              ).with_annotator_labels(label_dist, RA_dec)

    calibration_set = create_calibration_set(model, mb_calibration, 1, label_dist, RA_dec)
    threshold = calculate_threshold(calibration_set, 0.25)
    predictions = create_prediction_sets(model, mb_test_annotator, threshold, label_dist, RA_dec, "test")
    predictions_conf = create_prediction_sets(model, mb_conf_annotator, threshold, label_dist, RA_dec, "test_conf")
    prediction_set_sizes = []
    prediction_set_sizes_conf = []
    for sample in predictions:
        prediction_set_sizes.append(3 - sample["prediction_set"].count(0))
    prediction_set_sizes = np.array(prediction_set_sizes)
    for sample in predictions_conf:
        prediction_set_sizes_conf.append(3 - sample["prediction_set"].count(0))
    prediction_set_sizes_conf = np.array(prediction_set_sizes_conf)

    plot_data_mccp = {"umap": mb_test_umap,
                           "labels": mb_test_annotations,
                           "title": "Annotator labels",
                           "uncertainty": prediction_set_sizes,
                           "cbar_label": "Prediction set size",
                           "cbar_ticks": [1, 2, 3]
                           }
    plot_data_mccp_conf = {"umap": mb_conf_umap,
                           "labels": mb_conf_annotations,
                           "title": "Annotator labels",
                           "uncertainty": prediction_set_sizes_conf,
                           "cbar_label": "Prediction set size",
                           "cbar_ticks": [1, 2, 3]
                           }
    plot_embedding_uncertainty(save_dir + "/" + run_id + "_embedding_mccp_cov75.png", plot_data_mccp)
    plot_embedding_uncertainty(save_dir + "/" + run_id + "_embedding_mccp_conf_cov75.png", plot_data_mccp_conf)

    violin_plot(save_dir + "/" + run_id + "_violin_PE_cov75.png", mb_conf_entropy, prediction_set_sizes_conf, "Predictive entropy")
    violin_plot(save_dir + "/" + run_id + "_violin_annotator_cov75.png", annotator_entropy_test, prediction_set_sizes, "Entropy of label distribution")

    # Class-conditional conformal prediction
    FRI_set, FRII_set, hybrid_set = create_class_conditional_calibration_sets(model, mb_calibration, 1, label_dist, RA_dec)
    test_scores = calculate_class_conditional_scores(model, mb_test_annotator, FRI_set, FRII_set, hybrid_set, label_dist, RA_dec, "test")
    test_conf_scores = calculate_class_conditional_scores(model, mb_conf_annotator, FRI_set, FRII_set, hybrid_set, label_dist, RA_dec, "test_conf")

    plot_data_conditional = {"umap": mb_test_umap,
                           "labels": mb_test_annotations,
                           "title": "Annotator labels",
                           "uncertainty": test_scores,
                           "cbar_label": r'$\alpha$',
                           "cbar_ticks": None
                           }
    plot_data_conditional_conf = {"umap": mb_conf_umap,
                           "labels": mb_conf_annotations,
                           "title": "Annotator labels",
                           "uncertainty": test_conf_scores,
                           "cbar_label": r'$\alpha$',
                           "cbar_ticks": None
                           }
    
    plot_embedding_uncertainty(save_dir + "/" + run_id + "_embedding_conditional.png", plot_data_conditional)
    plot_embedding_uncertainty(save_dir + "/" + run_id + "_embedding_conditional_conf.png", plot_data_conditional_conf)

    scatter_plot(save_dir + "/" + run_id + "_scatter.png", annotator_entropy_test, test_scores, "Entropy of label distribution")
    scatter_plot(save_dir + "/" + run_id + "_scatter_conf.png", mb_conf_entropy, test_conf_scores, "Predictive entropy")

    # Test values of alpha

    #test_alpha(model, mb_calibration, mb_test, 1, save_dir + "/" + run_id + "_alphatest_m=1.png", label_dist, RA_dec)
    #test_alpha(model, mb_calibration, mb_test, 10, save_dir + "/" + run_id + "_alphatest_m=10.png", label_dist, RA_dec)
    #test_alpha(model, mb_calibration, mb_test, 100, save_dir + "/" + run_id + "_alphatest_m=100.png", label_dist, RA_dec)


def main():
    run_post_evaluation(RUN_ID)

if __name__ == "__main__":
    main()