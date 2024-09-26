import pandas as pd
import numpy as np
import openai
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from umap import UMAP
from umap.umap_ import nearest_neighbors
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


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


def average_embeddings(embeddings, weights=None, axis=0):
    """Calculates a (potentially weighted) average of an array of embeddings."""
    if weights is not None:
        embeddings = embeddings * weights
    return np.sum(embeddings) / embeddings.shape[axis]


def compute_nn():
    """Pre-computes the nearest neighbors graph for UMAP."""
    embeddings = st.session_state.embeddings
    with st.spinner('Calculating nearest neighbors...'):
        nn = nearest_neighbors(embeddings,
                               n_neighbors=250,
                               metric='euclidean',
                               metric_kwds=None,
                               angular=False,
                               random_state=None)
    st.session_state.precomputed_nn = nn
    return


def run_clustering():
    """Runs a cluster analysis on the user's chosen reduction. Cluster IDs and \
    centers are saved to the reduction's dictionary entry for plotting.
    """
    algo = st.session_state.clustering_algorithm
    cd = st.session_state.cluster_dict
    mod_name = cd[algo]['sklearn_name']
    lower_name = cd[algo]['lower_name']
    kwargs = {p: st.session_state[lower_name + '_' + p]
              for p in cd[algo]['params']}
    mod = globals()[mod_name](**kwargs)
    reduc_name = st.session_state.current_reduction
    with st.spinner('Running the clustering algorithm...'):
        mod.fit(st.session_state.reduction_dict[reduc_name]['points'])
    if algo == 'DBSCAN':
        centers = mod.core_sample_indices_
    elif algo == 'KMeans':
        centers = mod.cluster_centers_
    cluster_df = st.session_state.reduction_dict[reduc_name]['cluster_ids']
    labels = np.array(mod.labels_).astype(str)
    if cluster_df is not None:
        cluster_df[lower_name + '_id'] = labels
    else:
        cluster_df = pd.DataFrame(mod.labels_, columns=[lower_name + '_id'])
    st.session_state.reduction_dict[reduc_name]['cluster_ids'] = cluster_df.astype(str)
    st.session_state.reduction_dict[reduc_name]['cluster_mods'].update({lower_name: mod})
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



def make_dendrogram(model):
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
