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
from matplotlib import colors

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
            ax.set_box_aspect(1) # Make all subplots square
            #ax.set_aspect('equal', adjustable='box')
            axes.append(ax)
    
    # Centre the subplots in the last row
    num_last = n - (rows - 1) * cols  # Number of axes needed in the last row
    offset = (cols - num_last) // 2   # Left offset to centre the last row
    for j in range(num_last):
        ax = fig.add_subplot(gs[rows - 1, offset + j])
        ax.set_box_aspect(1) # Make all subplots square
        #ax.set_aspect('equal', adjustable='box')
        axes.append(ax)
    
    return fig, axes


def plot_embedding(fig_path, plot_data):

    fig, axes = create_subplots(len(plot_data))
    marker_size = 2
    xmin = np.min(plot_data[0]["umap"][:, 0]) - 1
    xmax = np.max(plot_data[0]["umap"][:, 0]) + 1
    ymin = np.min(plot_data[0]["umap"][:, 1]) - 1
    ymax = np.max(plot_data[0]["umap"][:, 1]) + 1

    cmap = colors.LinearSegmentedColormap.from_list("", ["blue","orange","green"])

    for ax in axes[1:]:
        ax.sharex(axes[0])
        ax.sharey(axes[0])

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
        handles = [pl.plot([],color=sc.get_cmap()(sc.norm(c)),ls="", marker="o")[0] for c,l in clset]
        labels = [l for c,l in clset]
        ax.legend(handles, labels)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("UMAP x")
        ax.set_ylabel("UMAP y")
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

    pl.gca().set_aspect("equal", "datalim")

    fig.savefig(fig_path, bbox_inches="tight", dpi=600)


