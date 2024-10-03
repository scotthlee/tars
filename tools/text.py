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
        self.text = docs
        self.metadata = metadata
        self.embeddings = embeddings
        self.reductions = {}

    def embed(self,
              model_name='ada-002',
              engine='text-embedding-ada-002'):
        """Embeds the object's text."""
        if model_name == 'ada-002':
            oai.load_openai_settings(mode='embeddings')
            with st.spinner('Fetching the embeddings...'):
                response = openai.Embedding.create(
                    input=self.text,
                    engine=engine,
                )
            embeddings = np.array([response['data'][i]['embedding']
                      for i in range(len(self.text))])
        elif model_type == 'huggingface':
            pass
        self.embeddings = pd.DataFrame(embeddings)
        self.precomputed_knn = data.compute_nn(self.embeddings)

    def reduce(self,
               method='UMAP',
               dimensions=3,
               main_kwargs={},
               aux_kwargs={}):
        reducer = data.EmbeddingReduction(method=method,
                                          dimensions=dimensions)
        reducer.fit(self.embeddings,
                    precomputed_knn=self.precomputed_knn,
                    main_kwargs=main_kwargs,
                    aux_kwargs=aux_kwargs)
        st.write(reducer.name)
        self.reductions.update({reducer.name: reducer})
        return

    def cluster(self,
                reduction,
                method,
                kwargs={}):
        """Runs a clustering algorithm on one of the object's reductions."""
        self.reductions[reduction].cluster(method=method, kwargs=kwargs)
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
    return


def fetch_embeddings():
    """Generates embeddings for the user's text, along with the associated \
    PCA reduction for initial visualization."""
    td = st.session_state.text_data_dict[st.session_state.embedding_type]
    td.embed(model_name=st.session_state.embedding_model)
    return


def average_embeddings(embeddings, weights=None, axis=0):
    """Calculates a (potentially weighted) average of an array of embeddings."""
    if weights is not None:
        embeddings = embeddings * weights
    return np.sum(embeddings) / embeddings.shape[axis]
