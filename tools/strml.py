"""Functions for working specifically with objects in the Streamlit session
state.
"""
import re
import logging
import os
import pandas as pd
import streamlit as st
import ast
import pymupdf

from tools.data import compute_nn
from tools.text import TextData


def keep(key):
    """Sets a session state variable's value based on the corresponding
    widget value.
    """
    st.session_state[key] = st.session_state['_' + key]
    return


def unkeep(key):
    """Sets a widget's value based on the corresponding session state
    variable's value.
    """
    st.session_state['_' + key] = st.session_state[key]
    return


def update_settings(keys, toast=True, toast_icon='👍'):
    """Wrapper for keep() that runs through a list of keys."""
    for key in keys:
        keep(key)
    if toast:
        st.toast('Settings updated!', icon=toast_icon)
    pass


def generate_disclaimer():
    """Generates a basic disclaimer that content was generated with ChatGPT."""
    disclaimer = '\n\n ### Generative AI Disclosure'
    disclaimer += '\nContent created with the help of '
    disclaimer += str(st.session_state.engine) + '. '
    return disclaimer


def reset_gpt():
    """Restores the default ChatGPT settings."""
    for key in st.session_state.gpt_keys:
        st.session_state[key] = st.session_state.gpt_defaults[key]
    return


def load_file():
    """Loads a source data file to be used as the basis for generating a data
    dictionary.
    """
    sf = st.session_state._source_file
    if sf is not None:
        data_type = st.session_state.data_type
        if data_type == 'Tabular data with text column':
            try:
                loaded_source = pd.read_csv(sf)
            except:
                loaded_source = pd.read_csv(sf, encoding='latin')
            st.session_state.source_file = loaded_source
            st.session_state.metadata = loaded_source
        elif data_type == 'Bulk documents':
            st.session_state.data_type_dict.update({'Metadata': ['csv']})
            docs = []
            for f in sf:
                if f.name[-3:] == 'pdf':
                    with pymupdf.open(f) as source_doc:
                        text_blobs = [d.get_text() for d in source_doc]
                        doc = ' '.join(text_blobs)
                else:
                    doc = f.read().decode('utf-8')
                docs.append(doc)
            td = TextData(docs=docs)
            st.session_state.source_file = docs[0]
            st.session_state.current_text_data = 'documents'
            st.session_state.text_data_dict.update({'documents': td})
        elif data_type == 'Premade embeddings':
            try:
                embeddings = pd.read_csv(sf)
            except:
                embeddings = pd.read_csv(sf, encoding='latin')
            td = TextData(embeddings=embeddings)
            td.precomputed_knn = compute_nn(embeddings=embeddings)
            td.reduce(method='UMAP')
            st.session_state.current_text_data = 'documents'
            st.session_state.text_data_dict.update({'documents': td})
            st.session_state.current_reduction = td.last_reduction
            st.session_state.data_type_dict.update({'Metadata': ['csv']})
        elif data_type == 'Metadata':
            try:
                metadata = pd.read_csv(sf)
            except:
                metadata = pd.read_csv(sf, encoding='latin')
            td = fetch_td(st.session_state.current_text_data)
            td.metadata = metadata
            st.session_state.text_data_dict.update({'documents': td})
    return


def fetch_td(td_name):
    """Wrapper to hide the session state dict call for selecting a TextData
    object to work with.
    """
    return st.session_state.text_data_dict[td_name]


def set_text():
    """Adds the text to be embedded to the session state. Only applies when
    tabular data is used for the input.
    """
    text_col = st.session_state._text_column
    sf = st.session_state.source_file.dropna(axis=0, subset=text_col)
    sf[text_col] = sf[text_col].astype(str)
    docs = [str(d) for d in sf[text_col]]
    text_type = st.session_state.embedding_type
    td = TextData(docs=docs, metadata=sf)
    st.session_state.current_text_data = text_type
    st.session_state.text_data_dict.update({text_type: td})
    st.session_state.hover_columns = [text_col]
    st.session_state.source_file = sf
    st.session_state.enable_generate_button = True
    return


def fetch_embeddings():
    """Generates embeddings for the user's text, along with the associated \
    PCA reduction for initial visualization."""
    td = fetch_td(st.session_state.current_text_data)
    if td is not None:
        td.embed(model_name=st.session_state.embedding_model)
        reduce_dimensions()
    else:
        st.error('Please specify a text column to embed.')
    return


def reduce_dimensions():
    """Reduces the dimensonality of the currently-chosen embeddings."""
    td = fetch_td(st.session_state.current_text_data)
    method = st.session_state.reduction_method
    rd = st.session_state.reduction_dict
    method_low = rd[method]['lower_name']
    param_names = [method_low + '_' + p for p in rd[method]['params']]
    update_settings(keys=param_names, toast=False)
    kwargs = {p: st.session_state[method_low + '_' + p]
              for p in rd[method]['params']}
    dimensions = int(st.session_state.reduce_to_3d) + 2
    td.reduce(method=method, dimensions=dimensions, main_kwargs=kwargs)
    st.session_state.current_reduction = td.last_reduction
    return


def run_clustering():
    """Runs a cluster analysis on the user's chosen reduction. Cluster IDs and \
    centers are saved to the reduction's dictionary entry for plotting.
    """
    algo = st.session_state.clustering_algorithm
    cd = st.session_state.cluster_dict
    mod_name = cd[algo]['sklearn_name']
    lower_name = cd[algo]['lower_name']
    param_names = [lower_name + '_' + p for p in cd[algo]['params']]
    update_settings(keys=param_names, toast=False)
    update_settings(keys=['cluster_kwargs'], toast=False)
    main_kwargs = {p.replace(lower_name + '_', ''):
        st.session_state[p] for p in param_names}
    aux_kwargs = ast.literal_eval(st.session_state.cluster_kwargs)
    td = fetch_td(st.session_state.current_text_data)
    td.cluster(reduction=st.session_state.current_reduction,
               method=algo,
               main_kwargs=main_kwargs,
               aux_kwargs=aux_kwargs)
    return


def switch_reduction():
    """Wrapper function for choosing a new data reduction. Kludgy, but
    apparenty necessary for resetting the variable used to color points in
    the main plot.
    """
    update_settings(['current_reduction'])
    st.session_state.color_column = None
    return


def name_clusters():
    """Generates cluster names as topic lists. Currently only one method,
    cluster-based TF-IDF, from BERTopic, is supported.
    """
    td = fetch_td(st.session_state.current_text_data)
    reduction = st.session_state.current_reduction
    model = 'DBSCAN'
    td.name_clusters(reduction=st.session_state.current_reduction,
                     method='TF-IDF',
                     model=model)
    return
