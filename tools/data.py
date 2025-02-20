import numpy as np
import pandas as pd
import streamlit as st

from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import normalize, scale
from sklearn import metrics
from umap import UMAP
from umap.umap_ import nearest_neighbors
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import rankdata


class ClusterModel:
    """A container class for a scikit-learn clustering model."""
    def __init__(self, model_name):
        self.centers = None
        self.model = None
        self.labels = None
        self.keywords = None
        self.counts = None
        self.model_name = model_name
        self.id_str = None
        self.score_functions = [
            'silhouette_score',
            'calinski_harabasz_score',
            'davies_bouldin_score',
        ]
        self.scores = {}
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

    def fit(self, X, id_str=None, main_kwargs={}, aux_kwargs={}):
        """Fits the chosen clustering model to the data."""
        # Setting up the sklearn function and associated args
        algo = self.model_name
        mod_name = self.model_choices[algo]['sklearn_name']
        lower_name = self.model_choices[algo]['lower_name']
        kwargs = {**main_kwargs, **aux_kwargs}
        mod = globals()[mod_name](**kwargs)

        # Set the column name, if not provided
        if id_str is None:
            id_str = lower_name + '_id'
        self.id_str = id_str

        # Doing a deep copy so as not to change the input data
        X = deepcopy(X)

        # Kludge; dropping columns with "_id" to make sure X is numeric
        to_drop = [s for s in X.columns.values if s not in ['d1', 'd2', 'd3']]
        X = X.drop(labels=to_drop, axis=1)

        # Fit the underlying model
        with st.spinner('Running the clustering algorithm...'):
            mod.fit(X)

        # Set some class-level attributes, like the cluster ID data frame and
        # the sklearn model name
        labels = np.array(mod.labels_).astype(str)
        self.labels = pd.DataFrame(labels, columns=[id_str])
        self.model = mod
        self.params = kwargs

        # Calculate cluster size
        ids = self.labels[id_str].unique()
        counts = {}
        for id in ids:
            id_samp = self.labels[self.labels[id_str] == id]
            counts.update({id : id_samp.shape[0]})
        self.counts = counts
        count_ranks = rankdata(list(counts.values()))
        self.count_ranks = {id: count_ranks[i] for i, id in enumerate(ids)}

        # Calculate intracluster variances
        vars = {}
        for id in ids:
            id_rows = np.where(self.labels[id_str] == id)[0]
            id_samp = X.iloc[id_rows, :]
            vars.update({id: id_samp.var().sum()})
        self.vars = vars
        var_ranks = rankdata(list(vars.values()))
        self.var_ranks = {id: var_ranks[i] for i, id in enumerate(ids)}

        # Calculate the cluster centers and scoring metrics
        self._calculate_centers(X, id_str)
        self._calculate_metrics(X, id_str)

        # Scoring the current clustering scheme across the three metrics
        self._score(X, id_str, labels)

        return

    def _calculate_centers(self, X, id):
        """Calculates the mean (i.e., the centroid) for each cluster."""
        if self.labels is not None:
            X[id] = self.labels[id]
            by_cluster = X.groupby(id, as_index=False)
            self.centers = by_cluster.mean(numeric_only=True)
        return

    def _calculate_metrics(self, X, id):
        """Calculates a few metrics for measuring cluster quality."""
        if self.labels is not None:
            if self.centers is None:
                self._calculate_centers(X, id)
            centers = self.centers.drop(id, axis=1)
            X[id] = self.labels[id]
            grouped = X.groupby(id)
            w_vars = grouped.var(numeric_only=True).sum(axis=1)
            c_vars = centers.var(numeric_only=True).sum()
            self.icc = c_vars / (w_vars.mean() + c_vars)
        return

    def _score(self, X, id, labels):
        for score_function in self.score_functions:
            sklearn_func = getattr(metrics, score_function)
            self.scores.update({id: {score_function: sklearn_func(X, labels)}})

    def generate_keywords(
        self,
        docs,
        id_str=None,
        method='TF-IDF',
        norm='l1',
        top_k=10,
        main_kwargs={},
        aux_kwargs={}
        ):
        """Geneates the to keywords for each cluster."""
        # Merge docs with cluster IDs
        lower_name = self.model_choices[self.model_name]['lower_name']
        cluster_df = deepcopy(self.labels)
        cluster_df['docs'] = docs
        cluster_df.docs = cluster_df.docs.astype(str)

        # Set default cluster labeling name
        if id_str is None:
            id_str = lower_name + '_id'

        if method == 'TF-IDF':
            # Merge the docs in each cluster
            cluster_ids = cluster_df[id_str].unique()
            clustered_docs = []
            for id in cluster_ids:
                doc_blob = ' '.join(cluster_df.docs[cluster_df[id_str] == id])
                clustered_docs.append(doc_blob)

            # Vectorize the clustered documents and fetch the vocabulary
            veccer = CountVectorizer(stop_words='english')
            count_vecs = veccer.fit_transform(clustered_docs)
            vocab = veccer.vocabulary_
            reverse_vocab = {vocab[k]: k for k in list(vocab.keys())}

            # Conver the count vectors to TF-IDF vectors
            tiffer = TfidfTransformer(norm=norm)
            tfidf_vecs = tiffer.fit_transform(count_vecs).toarray()

            # Get the top terms for each set of clustered docs based on TF-IDF
            sorted = np.flip(np.argsort(tfidf_vecs, axis=1), axis=1)
            sorted_terms = []
            for r in sorted:
                row_terms = []
                for k in r[:top_k]:
                    row_terms.append(reverse_vocab.get(k, k))
                sorted_terms.append(row_terms)
            sorted_terms = np.array(sorted_terms)

            # Save the keywords as a dict
            keyword_dict = {}
            for i, id in enumerate(cluster_ids):
                keyword_dict.update({id: [w for w in sorted_terms[i]]})
            self.keywords = keyword_dict

    def sample_points(self, max_count=20, min_count=5):
        """Randomly samples points from each cluster."""
        if self.labels is not None:
            algo = self.model_name
            lower_name = self.model_choices[algo]['lower_name']
            id_str = lower_name + '_id'
            ids = self.labels[id_str].unique()
            out = {}
            for id in ids:
                samp = self.labels[self.labels[id_str] == id]
                cluster_size = samp.shape[0]
                if cluster_size > min_count:
                    n = np.min([max_count, cluster_size])
                    id_dict = {
                        id: [i for i in samp.sample(n=n).index.values]
                    }
                else:
                    id_dict = {id: [i for i in samp.index.values]}
                out.update(id_dict)
            return out

    def sample_docs(self, docs, max_count=20, min_count=5):
        """Samples points from each cluster and then samples the corresponding
        documents from the provided list of documents.
        """
        points = self.sample_points(max_count=max_count, min_count=min_count)
        cluster_ids = list(points.keys())
        doc_samples = {id: [docs[i] for i in points[id]] for id in cluster_ids}
        return doc_samples



class EmbeddingReduction:
    """A container class for a dimensionally-reduced set of embeddings."""
    def __init__(self, method='UMAP', dimensions=3):
        self.method = method
        self.dimensions = dimensions
        self.points = None
        self.label_df = None
        self.id_strs = []
        self.cluster_models = {}
        self.cluster_names = {}
        self.doc_samples = {}
        self.cluster_scores = {}

    def cluster(
        self,
        method='HDBSCAN',
        id_str=None,
        main_kwargs={},
        aux_kwargs={}
        ):
        """Adds a ClusterModel to the current reduction."""
        mod = ClusterModel(model_name=method)
        mod.fit(
            X=deepcopy(self.points),
            id_str=id_str,
            main_kwargs=main_kwargs,
            aux_kwargs=aux_kwargs
        )
        self.id_strs.append(mod.id_str)
        mod_name = mod.model_name
        current_models = self.cluster_models.keys()
        mod_count = sum([mod_name in s for s in current_models])
        if mod_count > 0:
            mod_name += '_' + str(mod_count)

        self.cluster_models.update({mod_name: mod})
        self.cluster_scores.update(mod.scores)
        if self.label_df is None:
            self.label_df = mod.labels
        else:
            self.label_df[mod.labels.columns.values] = mod.labels
        return

    def fit(self, X, do_scale=False, main_kwargs={}, aux_kwargs={}):
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

    def generate_cluster_keywords(
        self,
        docs,
        model,
        id_str=None,
        method='TF-IDF',
        top_k=10,
        norm='l1',
        main_kwargs={},
        aux_kwargs={}
        ):
        """Names clusters based on the text samples they contain. Uses one of
        two approaches: cluster TF-IDF (the last step of BERTopic), or direct
        labeling with ChatGPT."""
        self.cluster_models[model].generate_keywords(
            method=method,
            id_str=id_str,
            top_k=top_k,
            norm=norm,
            docs=docs,
            main_kwargs=main_kwargs,
            aux_kwargs=aux_kwargs
        )

    def sample_docs(self, model, docs, max_count=20, min_count=5):
        """Randomly samples documents from each cluster given by the specified
        clustering model.
        """
        doc_samples = self.cluster_models[model].sample_docs(
            docs=docs,
            ax_count=max_count,
            min_count=min_count
        )
        self.doc_samples.update({model: doc_samples})



def make_dendrogram(model):
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


def compute_nn(embeddings, n_neighbors=250, metric='euclidean'):
    """Pre-computes the nearest neighbors graph for UMAP."""
    with st.spinner('Calculating nearest neighbors...'):
        nn = nearest_neighbors(
            embeddings,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds=None,
            angular=False,
            random_state=None
        )
    return nn
