import pandas as pd
import numpy as np
import openai
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from tools import strml


def reduce_dimensions(reduction_method=None):
    """Performs dimensionality reduction on the current state's embeddings. \
    Algorithm options are PCA, UMAP, and t-SNE."""
    strml.update_settings(st.session_state.dimred_params)
    if reduction_method is None:
        reduction_method = st.session_state.reduction_method
    dims = 3 if st.session_state.map_in_3d else 2
    if reduction_method == 'PCA':
        reducer = PCA(n_components=dims)
    elif reduction_method == 'UMAP':
        reducer = UMAP(n_components=dims,
                       precomputed_knn=st.session_state.precomputed_nn,
                       n_neighbors=st.session_state.umap_n_neighbors,
                       min_dist=st.session_state.umap_min_dist)
    elif reduction_method == 't-SNE':
        reducer = TSNE(n_components=dims,
                       perplexity=st.session_state.tsne_perplexity,
                       n_iter=st.session_state.tsne_n_iter)
    with st.spinner('Running the dimensionality reduction algorithm. Please \
                    wait.'):
        reduction = reducer.fit_transform(st.session_state.embeddings)
    colnames = ['d' + str(i + 1) for i in range(dims)]
    reduction = pd.DataFrame(reduction, columns=colnames)
    st.session_state.reduction = reduction
    st.session_state.source_file[colnames] = reduction
    return


def average_embeddings(embeddings, weights=None, axis=0):
    """Calculates a (potentially weighted) average of an array of embeddings."""
    if weights is not None:
        embeddings = embeddings * weights
    return np.sum(embeddings) / embeddings.shape[axis]
