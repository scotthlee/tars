import numpy as np
import pandas as pd
import io
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize, scale
from umap import UMAP
from umap.umap_ import nearest_neighbors
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


class ClusterModel:
    """A container class for a scikit-learn clustering model."""
    def __init__(self, model_name):
        self.centers = None
        self.model = None
        self.labels = None
        self.model_name = model_name
        self.model_choices = {
            'DBSCAN': {
                'sklearn_name': 'DBSCAN',
                'lower_name': 'dbscan'
            },
            'HDBSCAN': {
                'sklearn_name': 'HDBSCAN',
                'lower_name': 'hdbscan'
            },
            'k-means': {
                'sklearn_name': 'KMeans',
                'lower_name': 'kmeans'
            },
            'Agglomerative': {
                'sklearn_name': 'AgglomerativeClustering',
                'lower_name': 'aggl'
            }
        }

    def fit(self, X,
            main_kwargs={},
            aux_kwargs={}):
        """Fits the chosen clustering model to the data."""
        algo = self.model_name
        mod_name = self.model_choices[algo]['sklearn_name']
        lower_name = self.model_choices[algo]['lower_name']
        kwargs = {**main_kwargs, **aux_kwargs}
        mod = globals()[algo](**kwargs)
        with st.spinner('Running the clustering algorithm...'):
            mod.fit(X)
        if algo == 'DBSCAN':
            self.centers = mod.core_sample_indices_
        elif algo == 'KMeans':
            self.centers = mod.cluster_centers_
        labels = np.array(mod.labels_).astype(str)
        self.labels = pd.DataFrame(labels, columns=[lower_name + '_id'])
        self.model = mod
        self.params = kwargs
        return


class EmbeddingReduction:
    """A container class for a dimensionally-reduced set of embeddings."""
    def __init__(self, method='UMAP', dimensions=3):
        self.method = method
        self.dimensions = dimensions
        self.points = None
        self.label_df = None
        self.cluster_models = {}

    def name_reduction(self, param_vals):
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
        if self.method != 'PCA':
            curr_name = param_dict[self.method]['name']
            param_abbrevs = param_dict[self.method]['param_abbrevs']
            param_str = ', '.join([param_abbrevs[i] + '=' + str(param_vals[i])
                                   for i in range(len(param_vals))])
            name_str = self.method + '(' + param_str + ')'
        else:
            name_str = 'PCA'
        self.name = name_str

    def cluster(self, method='HDBSCAN', main_kwargs={}, aux_kwargs={}):
        """Adds a ClusterModel to the current reduction."""
        mod = ClusterModel(model_name=method)
        mod.fit(X=self.points,
                main_kwargs=main_kwargs,
                aux_kwargs=aux_kwargs)
        self.cluster_models.update({mod.model_name: mod})
        if self.label_df is None:
            self.label_df = mod.labels
        else:
            self.label_df[mod.labels.columns.values] = mod.labels
        return

    def fit(self, X,
            do_scale=False,
            main_kwargs={},
            aux_kwargs={}):
        """Performs dimensionality reduction on a set of embeddings. \
        Algorithm options are PCA, UMAP, and t-SNE."""
        reduction_method = self.method
        dims = self.dimensions
        kwargs = {**main_kwargs, **aux_kwargs}
        if do_scale:
            X = scale(X)
        if reduction_method == 'PCA':
            reducer = PCA(n_components=dims)
        elif reduction_method == 'UMAP':
            reducer = UMAP(n_components=dims, **kwargs)
        elif reduction_method == 't-SNE':
            reducer = TSNE(n_components=dims, **kwargs)
        with st.spinner('Running ' + reduction_method + '...'):
            reduction = reducer.fit_transform(X)
        colnames = ['d' + str(i + 1) for i in range(dims)]
        self.points = pd.DataFrame(reduction, columns=colnames)
        self.name_reduction(list(main_kwargs.values()))
        return


@st.dialog('Dendrogram')
def show_dendrogram():
    current_reduc = st.session_state.current_reduction
    if 'aggl' not in st.session_state.reduction_dict[current_reduc]['cluster_mods']:
        st.write('Please run the agglomerative clustering algorithm to see \
                 a dendrogram.')
    else:
        mod = st.session_state.reduction_dict[current_reduc]['cluster_mods']['aggl']
        fig = make_dendrogram(mod)
        st.pyplot(fig=fig, clear_figure=True)
    st.write('Hello!')


def make_dendrogram(model, as_bytes=True):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Assemble the dendrogram

    # Make the plot
    fig = plt.figure()
    dn = dendrogram(linkage_matrix)
    return fig


def compute_nn(embeddings,
               n_neighbors=250,
               metric='euclidean'):
    """Pre-computes the nearest neighbors graph for UMAP."""
    with st.spinner('Calculating nearest neighbors...'):
        nn = nearest_neighbors(embeddings,
                               n_neighbors=n_neighbors,
                               metric=metric,
                               metric_kwds=None,
                               angular=False,
                               random_state=None)
    return nn


def reduce_dimensions(reduction_method=None):
    """Performs dimensionality reduction on the current state's embeddings. \
    Algorithm options are PCA, UMAP, and t-SNE."""
    if reduction_method is None:
        reduction_method = st.session_state.reduction_method
    else:
        st.session_state.reduction_method = reduction_method
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
    with st.spinner('Running ' + reduction_method + '...'):
        reduction = reducer.fit_transform(st.session_state.embeddings)
    colnames = ['d' + str(i + 1) for i in range(dims)]
    reduction = pd.DataFrame(reduction, columns=colnames)
    reduction_name = name_reduction()
    st.session_state.reduction_dict.update({
        reduction_name: {
            'points': reduction,
            'cluster_ids': None,
            'cluster_mods': {}
        }
    })
    st.session_state.current_reduction = reduction_name
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
    if curr_method != 'PCA':
        curr_name = param_dict[curr_method]['name']
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
