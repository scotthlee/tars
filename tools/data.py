import numpy as np
import pandas as pd
import io
import streamlit as st

from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
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
        self.topics = None
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

    def name_clusters(self,
                      docs,
                      method='TF-IDF',
                      norm='l1',
                      top_k=10,
                      main_kwargs={},
                      aux_kwargs={}):
        """Names clusters based on the text samples they contain. Uses one of
        two approaches: cluster TF-IDF (the last step of BERTopic), or direct
        labeling with ChatGPT."""
        # Merge docs with cluster IDs
        lower_name = self.model_choices[self.model_name]['lower_name']
        id_name = lower_name + '_id'
        cluster_df = deepcopy(self.labels)
        cluster_df['docs'] = docs
        cluster_df.docs = cluster_df.docs.astype(str)

        if method == 'TF-IDF':
            # Merge the docs in each cluster
            cluster_ids = cluster_df[id_name].unique()
            st.write(id_name)
            st.write(cluster_ids)
            clustered_docs = []
            for id in cluster_ids:
                doc_blob = ' '.join(cluster_df.docs[cluster_df[id_name] == id])
                clustered_docs.append(doc_blob)

            # Vectorize the clustered documents and fetch the vocabulary
            veccer = CountVectorizer(stop_words='english')
            count_vecs = veccer.fit_transform(clustered_docs)
            vocab = veccer.vocabulary_
            reverse_vocab = {vocab[k]: k for k in list(vocab.keys())}

            # Conver the count vectors to TF-IDF vectors
            tiffer = TfidfTransformer(norm=norm)
            tfidf_vecs = tiffer.fit_transform(count_vecs)

            # Get the top terms for each set of clustered docs based on TF-IDF
            sorted = np.flip(np.argsort(tfidf_vecs, axis=1), axis=1)
            sorted_terms = np.array([[reverse_vocab[k] for k in r]
                                     for r in sorted])
            top_terms = sorted_terms[:, :top_k]
            self.topics = top_terms


class EmbeddingReduction:
    """A container class for a dimensionally-reduced set of embeddings."""
    def __init__(self, method='UMAP', dimensions=3):
        self.method = method
        self.dimensions = dimensions
        self.points = None
        self.label_df = None
        self.cluster_models = {}
        self.cluster_names = {}

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

    def name_clusters(self,
                      model,
                      docs,
                      method='TF-IDF',
                      top_k=10,
                      norm='l1',
                      main_kwargs={},
                      aux_kwargs={}):
        """Names clusters based on the text samples they contain. Uses one of
        two approaches: cluster TF-IDF (the last step of BERTopic), or direct
        labeling with ChatGPT."""
        self.cluster_models[model].name_clusters(method=method,
                                                 top_k=top_k,
                                                 norm=norm,
                                                 docs=docs,
                                                 main_kwargs=main_kwargs,
                                                 aux_kwargs=aux_kwargs)


def make_dendrogram(model, as_bytes=True):
    """Makes a dendrogram for an agglomerative clustering model."""
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
