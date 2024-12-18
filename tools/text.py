import pandas as pd
import numpy as np
import openai
import streamlit as st
import io
import openai
import spacy

from matplotlib import pyplot as plt
from multiprocessing import Pool

from tools import oai, data


class TextData:
    def __init__(self,
                 docs=None,
                 metadata=None,
                 embeddings=None):
        self.docs = docs
        self.metadata = metadata
        self.embeddings = embeddings
        self.reductions = {}
        self.last_reduction = None

    def embed(self,
              model_name='ada-002',
              engine='text-embedding-ada-002'):
        """Embeds the object's text."""
        if model_name == 'ada-002':
            oai.load_openai_settings(mode='embeddings')
            with st.spinner('Fetching the embeddings...'):
                response = openai.Embedding.create(
                    input=self.docs,
                    engine=engine,
                )
            embeddings = np.array([response['data'][i]['embedding']
                      for i in range(len(self.docs))])
        elif model_type == 'huggingface':
            pass
        self.embeddings = pd.DataFrame(embeddings)
        self.precomputed_knn = data.compute_nn(self.embeddings)

    def reduce(self,
               method='PCA',
               dimensions=3,
               main_kwargs={},
               aux_kwargs={}):
        """Reduces the dimensionality of the text embeddings using one of
        three methods: PCA, t-SNE, or UMAP. The reduced-dimensionality
        embeddings are stored as a data.EmbeddingReduction() object stored in
        the TextData object's reductions dict attribute.
        """
        reducer = data.EmbeddingReduction(method=method,
                                          dimensions=dimensions)
        reducer.fit(self.embeddings,
                    main_kwargs=main_kwargs,
                    aux_kwargs=aux_kwargs)
        self.reductions.update({reducer.name: reducer})
        self.last_reduction = reducer.name
        return

    def cluster(self,
                reduction,
                method,
                main_kwargs={},
                aux_kwargs={}):
        """Runs a clustering algorithm on one of the object's reductions."""
        self.reductions[reduction].cluster(method=method,
                                           main_kwargs=main_kwargs,
                                           aux_kwargs=aux_kwargs)
        return

    def generate_cluster_keywords(self,
                      reduction,
                      model,
                      docs=None,
                      method='TF-IDF',
                      top_k=10,
                      norm='l1',
                      main_kwargs={},
                      aux_kwargs={}):
        """Names clusters based on the text samples they contain. Uses one of
        two approaches: cluster TF-IDF (the last step of BERTopic), or direct
        labeling with ChatGPT.
        """
        if docs is None:
            docs = self.docs
        self.reductions[reduction].generate_cluster_keywords(model=model,
                                                             method=method,
                                                             top_k=top_k,
                                                             norm=norm,
                                                             docs=docs,
                                                             main_kwargs=main_kwargs,
                                                             aux_kwargs=aux_kwargs)
        return


def docs_to_sents():
    """Converts a set of documents to a set of sentences."""
    sf = st.session_state.source_file
    docs = st.session_state.text['documents']
    nlp = spacy.load(name='en_core_web_sm',
                     enable='senter',
                     config={'nlp': {'disabled': []}})
    sents = [[s for s in nlp(d).sents] for d in docs]
    sents_flat = [s for l in sents for s in l]
    rows = [[i] * len(l) for i, l in enumerate(sents)]
    rows_flat = [r for l in rows for r in l]
    return sents_flat, rows_flat


def average_embeddings(embeddings, weights=None, axis=0):
    """Calculates a (potentially weighted) average of an array of embeddings."""
    if weights is not None:
        embeddings = embeddings * weights
    return np.sum(embeddings) / embeddings.shape[axis]
