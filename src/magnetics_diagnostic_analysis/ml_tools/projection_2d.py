import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

from magnetics_diagnostic_analysis.project_vae.setting_vae import config


def project_tsne(embedding: np.ndarray, 
                 seed: int = 42,
                 **kwargs) -> np.ndarray:
    tsne = TSNE(
        n_components=2, 
        random_state=seed, 
        verbose=True,
        **kwargs
    )
    projection = tsne.fit_transform(embedding)
    return projection

def project_umap(embedding: np.ndarray, 
                 seed: int = 42,
                 **kwargs) -> tuple[np.ndarray, umap.UMAP]:
    """
    Projection UMAP in visualization embedded space and return the UMAP model.

    UMAP parameters:
        n_components: Dimension of the projection space (2 or 3)
        n_neighbors: Size of the local neighborhood
        min_dist: Minimum distance between points in the projection space
        random_state: Random seed for reproducibility
        **kwargs: Additional arguments for UMAP
    """
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=seed,
        verbose=True,
        **kwargs
    )
    projection = umap_model.fit_transform(embedding)
    return projection, umap_model

def apply_umap(model: umap.UMAP, new_embedding: np.ndarray) -> np.ndarray:
    """ Apply a trained UMAP model to new data. """
    return model.transform(new_embedding)



def plot_projection(projection: np.ndarray, 
                    labels: np.ndarray = None,
                    title: str = "Visualisation t-SNE",
                    filename: str = "tsne_plot.png",
                    figsize: tuple = (12, 8),
                    alpha: float = 0.7,
                    legend: bool = True) -> None:
    """
    Create and save a t-SNE or UMAP projection plot.

    Args:
        projection: (np.ndarray) numpy array of shape (n_samples, 2) or (n_samples, 3)
        labels: (np.ndarray) numpy array of shape (n_samples,) or None
        title: (str) Title of the plot
        filename: (str) Name of the output file
        figsize: (tuple) Size of the figure (width, height)
        alpha: (float) Transparency of the points
        legend: (bool) Show legend if labels are provided
    """
    if projection.shape[1] not in [2, 3]:
        raise ValueError("La projection doit avoir 2 ou 3 dimensions")
    
    plt.figure(figsize=figsize)
    if projection.shape[1] == 2:    # plot 2D
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(projection[mask, 0], projection[mask, 1],
                           c=[colors[i]], label=str(label), alpha=alpha, s=20)
            if legend:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(projection[:, 0], projection[:, 1], 
                       alpha=alpha, s=20, c='blue')
        plt.xlabel('projection dimension 1')
        plt.ylabel('projection dimension 2')

    else:   # plot 3D
        ax = plt.axes(projection='3d')
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter3D(projection[mask, 0], projection[mask, 1], projection[mask, 2],
                            c=[colors[i]], label=str(label), alpha=alpha, s=20)
            if legend:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter3D(projection[:, 0], projection[:, 1], projection[:, 2],
                        alpha=alpha, s=20, c='blue')
        ax.set_xlabel('projection dimension 1')
        ax.set_ylabel('projection dimension 2')
        ax.set_zlabel('projection dimension 3')

    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot enregistré sous : {filename}")




if __name__ == "__main__":

    seed = config.SEED
    data_nd = np.random.randn(1000, 16)

    projection_tsne = project_tsne(data_nd, seed=seed)
    projection_umap, umap_model = project_umap(data_nd, seed=seed)
    labels = np.random.choice([0, 1, 2], size=1000)


    plot_projection(
        projection=projection_tsne,
        labels=labels,
        title="Projection visualization t-SNE",
        filename="example_tsne.png",
        legend=True
    )

    plot_projection(
        projection=projection_umap,
        labels=labels,
        title="Projection visualization UMAP",
        filename="example_umap.png",
        legend=True
    )