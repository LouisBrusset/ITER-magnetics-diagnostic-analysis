import numpy as np
from sklearn.manifold import TSNE

def project_tsne(embedding: np.ndarray) -> np.ndarray:
    tsne = TSNE(verbose=True)
    projection = tsne.fit_transform(embedding)
    return projection
