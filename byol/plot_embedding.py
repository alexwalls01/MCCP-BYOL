import pylab as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from umap import UMAP
from matplotlib import gridspec

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
            x, _ = data
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

def create_subplots(n):

    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig = pl.figure(constrained_layout=True)
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    
    axes = []
    # Create axes for all full rows except the last one
    for i in range(rows - 1):
        for j in range(cols):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
    
    # Centre the subplots in the last row
    num_last = n - (rows - 1) * cols  # Number of axes needed in the last row
    offset = (cols - num_last) // 2   # Left offset to centre the last row
    for j in range(num_last):
        ax = fig.add_subplot(gs[rows - 1, offset + j])
        axes.append(ax)
    
    return fig, axes


def plot_embedding(fig_path, plot_data):

    fig, axes = create_subplots(len(plot_data))

    for index, ax in enumerate(axes):
        ax.set_title(plot_data[index]["title"])
        ax.scatter(plot_data[index]["fri_umap"][:, 0], plot_data[index]["fri_umap"][:, 1], label="FRI")
        ax.scatter(plot_data[index]["frii_umap"][:, 0], plot_data[index]["frii_umap"][:, 1], label="FRII")
        ax.scatter(plot_data[index]["hybrid_umap"][:, 0], plot_data[index]["hybrid_umap"][:, 1], label="Hybrid")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    pl.gca().set_aspect("equal", "datalim")

    fig.savefig(fig_path, bbox_inches="tight", dpi=600)


