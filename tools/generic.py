import pandas as pd
import numpy as np
import openai
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from umap.umap_ import nearest_neighbors


def reduce_dimensions(reduction_method=None):
    """Performs dimensionality reduction on the current state's embeddings. \
    Algorithm options are PCA, UMAP, and t-SNE."""
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
    st.session_state.reduction_dict.update({name_reduction(): reduction})
    return


def name_reduction():
    """Generates a string name for a particular reduction."""
    param_dict = {
        'UMAP': {
            'name': 'umap',
            'params': ['n_neighbors', 'min_dist'],
            'param_abbrevs': ['nn', 'dist']
        },
        't-SNE': {
            'name': 'tsne',
            'params': ['perplexity', 'learning_rate', 'n_iter'],
            'param_abbrevs': ['perp', 'lr', 'iter']
        }
    }
    curr_method = st.session_state.reduction_method
    curr_name = param_dict[curr_method]['name']
    if curr_method != 'PCA':
        param_vals = [
            str(st.session_state[curr_name + '_' + p])
            for p in param_dict[curr_method]['params']
        ]
        param_abbrevs = param_dict[curr_method]['param_abbrevs']
        param_str = ', '.join([param_abbrevs[i] + '=' + param_vals[i]
                               for i in range(len(param_vals))])
        name_str = curr_method + '(' + param_str + ')'
    else:
        name_str = 'PCA'
    return name_str


def average_embeddings(embeddings, weights=None, axis=0):
    """Calculates a (potentially weighted) average of an array of embeddings."""
    if weights is not None:
        embeddings = embeddings * weights
    return np.sum(embeddings) / embeddings.shape[axis]


def compute_nn():
    embeddings = st.session_state.embeddings
    with st.spinner('Calculating nearest neighbors. Please weight.'):
        nn = nearest_neighbors(embeddings,
                               n_neighbors=250,
                               metric='euclidean',
                               metric_kwds=None,
                               angular=False,
                               random_state=None)
    st.session_state.precomputed_nn = nn
    return
