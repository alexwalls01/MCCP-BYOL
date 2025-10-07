import pytorch_lightning as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from umap import UMAP
import torchvision.transforms as T
import pickle
import torch.nn.functional as F

from paths import Path_Handler
from config import load_config_finetune
from models import BYOL
from datamodules import RGZ_DataModule_Finetune
from datasets import MBFRFull, RGZ108k, MBFRConfidentNoHybrids, MBFRUncertainNoHybrids, Hybrids
from finetuning import MLPHead, FineTune

RUN_ID = os.environ.get("RUN_ID", "0")
PROJECT = os.environ.get("PROJECT", "0")
DIR = os.environ.get("DIR", "0")
CKPT_NAME = os.environ.get("CKPT_NAME", "0")
ALPHA = os.environ.get("ALPHA", 0)

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
    byol_model = BYOL.load_from_checkpoint('byol.ckpt')
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
    elif stage == "test_uncert":
        dataloader = datamodule.test_uncert_dataloader()
    elif stage == "test_hybrids":
        dataloader = datamodule.test_hybrids_dataloader()
    elif stage == "train":
        dataloader = datamodule.train_dataloader2()
    elif stage == "train_test":
        dataloader = datamodule.train_test_dataloader()
    elif stage == "rgz":
        dataloader = datamodule.rgz_dataloader()
    else:
        raise ValueError("Unsupported dataloader stage.")
    return dataloader

def create_calibration_set(model, mb_calibration, m, label_dist, RA_dec, alpha, save_dir):
    alpha = np.float64(alpha)

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
        sample["class"] = np.argmax(target)
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

    with open(save_dir + '/calibration_set_' + str(1 - alpha) + '.pkl', 'wb') as file:
        pickle.dump(calibration_set, file)

    return calibration_set

def calculate_threshold(calibration_set, alpha, save_dir):
    alpha = np.float64(alpha)
    non_conformity_scores = []
    for sample in calibration_set:
        softmax = F.softmax(torch.tensor(sample["logits"]), dim=0).tolist()
        target = sample["class"]
        score = softmax[target]
        non_conformity_scores.append(score)
    with open(save_dir + '/non_conformity_scores_' + str(1 - alpha) + '.pkl', 'wb') as file:
        pickle.dump(non_conformity_scores, file)
    non_conformity_scores = np.sort(np.array(non_conformity_scores))
    threshold = non_conformity_scores[int(np.floor(alpha * (len(calibration_set) + 1)) - 1)]
    return threshold

def create_prediction_sets(model, threshold, label_dist, RA_dec, stage):
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    prediction_loader = load_dataloader(stage, label_dist=label_dist, RA_dec=RA_dec)
    batch_predictions = trainer.predict(model, dataloaders=prediction_loader)
    if stage == "test":
        # Because the MiraBest test set has confident and uncertain subsets
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
        prediction_set = []
        softmax = F.softmax(torch.tensor(sample["logits"]), dim=0).tolist()
        sample["softmax"] = softmax
        for i in range(3):
            if softmax[i] > threshold:
                prediction_set.append(softmax[i])
            else:
                prediction_set.append(0)
        sample["prediction_set"] = prediction_set
    
    return predictions

def get_reducer(model, save_dir, run_id, transform):

    paths = Path_Handler()._dict()
    encoder = model.encoder
    encoder.eval()
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

    return reducer

def get_umap_embedding(data, model, reducer):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    umap = reducer.transform(data)

    return umap

def main():

    paths = Path_Handler()._dict()

    ckpt_path = DIR + '/finetune/' + RUN_ID + '/' + PROJECT + '/' + RUN_ID + '/checkpoints/' + CKPT_NAME
    save_dir = DIR + '/conformal_prediction/' + PROJECT + '/' + RUN_ID
    os.makedirs(save_dir, exist_ok=True)

    finetune_config = load_config_finetune()

    model = load_checkpoint(ckpt_path)
    byol_model = BYOL.load_from_checkpoint("byol.ckpt")
    config = byol_model.config
    config.update(finetune_config)
    label_dist = np.load(config["conformal_prediction"]["label_dist"])
    RA_dec = np.load(config["conformal_prediction"]["RA_dec"])
    mu, sig = config["data"]["mu"], config["data"]["sig"]

    transform = T.Compose(
        [
            T.CenterCrop(70),
            T.ToTensor(),
            T.Normalize((mu,), (sig,)),
        ]
    )

    
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
    mb_conf_test = MBFRConfidentNoHybrids(root=paths["mb"],
                                     train=False,
                                     transform=transform,
                                     download=False,
                                     aug_type="torchvision"
                                     )
    mb_uncert_test = MBFRUncertainNoHybrids(root=paths["mb"],
                                     train=False,
                                     transform=transform,
                                     download=False,
                                     aug_type="torchvision"
                                     )
    mb_hybrids_test = Hybrids(root=paths["mb"],
                                     train=False,
                                     transform=transform,
                                     download=False,
                                     aug_type="torchvision"
                                     )
    mb_calibration = MBFRFull(root=paths["mb"],
                              train=False,
                              calibration=True,
                              transform=transform,
                              download=False,
                              aug_type="torchvision"
                              ).with_annotator_labels(label_dist, RA_dec)

    
    # Get prediction sets
    calibration_set = create_calibration_set(model, mb_calibration, 1, label_dist, RA_dec, ALPHA, save_dir)
    threshold = calculate_threshold(calibration_set, ALPHA, save_dir)
    mb_test_predictions = create_prediction_sets(model, threshold, label_dist, RA_dec, "test")
    mb_train_predictions = create_prediction_sets(model, threshold, label_dist, RA_dec, "train_test")
    mb_conf_test_predictions = create_prediction_sets(model, threshold, label_dist, RA_dec, "test_conf")
    mb_uncert_test_predictions = create_prediction_sets(model, threshold, label_dist, RA_dec, "test_uncert")
    mb_hybrids_test_predictions = create_prediction_sets(model, threshold, label_dist, RA_dec, "test_hybrids")
    
    # Get umap embeddings
    reducer = get_reducer(model, save_dir, RUN_ID, transform)
    mb_test_umap = get_umap_embedding(mb_test, model, reducer)
    mb_train_umap = get_umap_embedding(mb_train, model, reducer)
    mb_conf_test_umap = get_umap_embedding(mb_conf_test, model, reducer)
    mb_uncert_test_umap = get_umap_embedding(mb_uncert_test, model, reducer)
    mb_hybrids_test_umap = get_umap_embedding(mb_hybrids_test, model, reducer)

    # Get predictive entropy
    hmc_entropy = np.genfromtxt(config["conformal_prediction"]["hmc_data"], delimiter=',', skip_header=1)
    hmc_conf_test = hmc_entropy[:,1][~np.isnan(hmc_entropy[:,1])]
    hmc_uncert_test = hmc_entropy[:,4][~np.isnan(hmc_entropy[:,4])]
    hmc_hybrids_test = hmc_entropy[:,5][~np.isnan(hmc_entropy[:,5])]

    for i in range (0, len(mb_test_predictions)):
        mb_test_predictions[i]["umap_x"] = mb_test_umap[i,0]
        mb_test_predictions[i]["umap_y"] = mb_test_umap[i,1]
        filename = mb_test_predictions[i]["filename"]
        mb_test_predictions[i]["label_dist"] = mb_test.with_annotator_labels(label_dist, RA_dec).get_dist(filename)
        mb_test_predictions[i]["label_entropy"] = mb_test.with_annotator_labels(label_dist, RA_dec).get_annotator_entropy()[i]
        with open(save_dir + '/test_predictions_' + str(1 - np.float64(ALPHA)) + '.pkl', 'wb') as file:
            pickle.dump(mb_test_predictions, file)

    for i in range (0, len(mb_train_predictions)):
        mb_train_predictions[i]["umap_x"] = mb_train_umap[i,0]
        mb_train_predictions[i]["umap_y"] = mb_train_umap[i,1]
        filename = mb_train_predictions[i]["filename"]
        mb_train_predictions[i]["label_dist"] = mb_train.with_annotator_labels(label_dist, RA_dec).get_dist(filename)
        mb_train_predictions[i]["label_entropy"] = mb_train.with_annotator_labels(label_dist, RA_dec).get_annotator_entropy()[i]
        with open(save_dir + '/train_predictions_' + str(1 - np.float64(ALPHA)) + '.pkl', 'wb') as file:
            pickle.dump(mb_train_predictions, file)

    for i in range (0, len(mb_conf_test_predictions)):
        mb_conf_test_predictions[i]["umap_x"] = mb_conf_test_umap[i,0]
        mb_conf_test_predictions[i]["umap_y"] = mb_conf_test_umap[i,1]
        filename = mb_conf_test_predictions[i]["filename"]
        mb_conf_test_predictions[i]["label_dist"] = mb_conf_test.with_annotator_labels(label_dist, RA_dec).get_dist(filename)
        mb_conf_test_predictions[i]["label_entropy"] = mb_conf_test.with_annotator_labels(label_dist, RA_dec).get_annotator_entropy()[i]
        mb_conf_test_predictions[i]["predictive_entropy"] = hmc_conf_test[i]
        with open(save_dir + '/conf_test_predictions_' + str(1 - np.float64(ALPHA)) + '.pkl', 'wb') as file:
            pickle.dump(mb_conf_test_predictions, file)

    for i in range (0, len(mb_uncert_test_predictions)):
        mb_uncert_test_predictions[i]["umap_x"] = mb_uncert_test_umap[i,0]
        mb_uncert_test_predictions[i]["umap_y"] = mb_uncert_test_umap[i,1]
        filename = mb_uncert_test_predictions[i]["filename"]
        mb_uncert_test_predictions[i]["label_dist"] = mb_uncert_test.with_annotator_labels(label_dist, RA_dec).get_dist(filename)
        mb_uncert_test_predictions[i]["label_entropy"] = mb_uncert_test.with_annotator_labels(label_dist, RA_dec).get_annotator_entropy()[i]
        mb_uncert_test_predictions[i]["predictive_entropy"] = hmc_uncert_test[i]
        with open(save_dir + '/uncert_test_predictions_' + str(1 - np.float64(ALPHA)) + '.pkl', 'wb') as file:
            pickle.dump(mb_uncert_test_predictions, file)

    for i in range (0, len(mb_hybrids_test_predictions)):
        mb_hybrids_test_predictions[i]["umap_x"] = mb_hybrids_test_umap[i,0]
        mb_conf_test_predictions[i]["umap_y"] = mb_hybrids_test_umap[i,1]
        filename = mb_hybrids_test_predictions[i]["filename"]
        mb_hybrids_test_predictions[i]["label_dist"] = mb_hybrids_test.with_annotator_labels(label_dist, RA_dec).get_dist(filename)
        mb_hybrids_test_predictions[i]["label_entropy"] = mb_hybrids_test.with_annotator_labels(label_dist, RA_dec).get_annotator_entropy()[i]
        mb_hybrids_test_predictions[i]["predictive_entropy"] = hmc_hybrids_test[i]
        with open(save_dir + '/conf_hybrids_predictions_' + str(1 - np.float64(ALPHA)) + '.pkl', 'wb') as file:
            pickle.dump(mb_hybrids_test_predictions, file)
    
if __name__ == "__main__":
    main()