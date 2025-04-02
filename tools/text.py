import pandas as pd
import numpy as np
import streamlit as st
import io
import spacy
import tiktoken
import threading
import time

from matplotlib import pyplot as plt
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer

from tools import data


class TextData:
    def __init__(
        self,
        docs=None,
        metadata=None,
        embeddings=None
    ):
        self.docs = docs
        self.metadata = metadata
        self.embeddings = embeddings
        self.reductions = {}
        self.last_reduction = None

    def embed(
        self,
        model_name='all-MiniLM-L6-v2'
    ):
        """Embeds the object's text."""
        with st.spinner('Generating the embeddings...'):
            mod = SentenceTransformer(model_name)
            embeddings = mod.encode(self.docs)
        self.embeddings = pd.DataFrame(embeddings)
        self.precomputed_knn = data.compute_nn(self.embeddings)

    def reduce(
        self,
        method='PCA',
        dimensions=3,
        main_kwargs={},
        aux_kwargs={}
    ):
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

    def cluster(
        self,
        reduction,
        method,
        id_str=None,
        main_kwargs={},
        aux_kwargs={}
    ):
        """Runs a clustering algorithm on one of the object's reductions."""
        self.reductions[reduction].cluster(
            method=method,
            id_str=id_str,
            main_kwargs=main_kwargs,
            aux_kwargs=aux_kwargs
        )
        return

    def generate_cluster_keywords(
        self,
        reduction,
        id_str,
        docs=None,
        method='TF-IDF',
        top_k=10,
        norm='l1',
        main_kwargs={},
        aux_kwargs={}
    ):
        """Names clusters based on the text samples they contain. Uses one of
        two approaches: cluster TF-IDF (the last step of BERTopic), or direct
        labeling with ChatGPT.
        """
        if docs is None:
            docs = self.docs
        self.reductions[reduction].generate_cluster_keywords(
            id_str=id_str,
            method=method,
            top_k=top_k,
            norm=norm,
            docs=docs,
            main_kwargs=main_kwargs,
            aux_kwargs=aux_kwargs
        )
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


def docs_to_tokens(docs, scheme='cl100k_base'):
    """Converts a list of text chunks to a list of lists of tokens."""
    enc = tiktoken.get_encoding(scheme)
    encodings = [enc.encode(str(d)) for d in docs]
    return encodings


def tokens_to_docs(encodings, scheme='cl100k_base'):
    """Converts a list of tiktoken encodings back to their original text
    strings."""
    enc = tiktoken.get_encoding(scheme)
    docs = [enc.decode(l) for l in encodings]
    return docs


def truncate_text(docs, max_length, scheme='cl100k_base'):
    """Clips documents so that they don't exceed a given model's context
    window. Mainly for use with embedding models.
    """
    encodings = docs_to_tokens(docs, scheme)
    trimmed_encodings = []
    for e in encodings:
        if len(e) > max_length:
            trimmed_encodings.append(e[:max_length])
        else:
            trimmed_encodings.append(e)
    trimmed_text = tokens_to_docs(trimmed_encodings, scheme)
    return trimmed_text


def chunk_to_tpm(
    docs,
    max_docs=2000,
    max_doc_length=8192,
    tpm=120000,
    scheme='cl100k_base'
    ):
    """Breaks a list of documents into chunks to avoid triggering API TPM
    limits.
    """
    docs = truncate_text(docs, max_length=max_doc_length, scheme=scheme)
    doc_tokens = docs_to_tokens(docs, scheme=scheme)
    doc_blocks = []
    curr_block = []
    block_length = 0
    for doc_num, doc in enumerate(docs):
        block_length += len(doc_tokens[doc_num])
        if (block_length <= tpm) and (len(curr_block) < max_docs):
            curr_block.append(doc)
        else:
            doc_blocks.append(curr_block)
            block_length = len(doc_tokens[doc_num])
            curr_block = [doc]
    if curr_block:
        doc_blocks.append(curr_block)

    return doc_blocks

def split_list(lst, n):
    """Splits a list into sublists of size n."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]
