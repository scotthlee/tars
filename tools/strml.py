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
import markdown
import sklearn

from sklearn import metrics
from copy import deepcopy

from tools.data import compute_nn
from tools.text import TextData
from tools import text


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
        if '_' + key in st.session_state:
            keep(key)
        else:
            return
    if toast:
        st.toast('Settings updated!', icon=toast_icon)
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
            st.session_state.embedding_type_select = 'documents'
            st.session_state.text_data_dict.update({'documents': td})
        elif data_type == 'Premade embeddings':
            try:
                embeddings = pd.read_csv(sf)
            except:
                embeddings = pd.read_csv(sf, encoding='latin')
            st.session_state.premade_loaded = True
            td = TextData(embeddings=embeddings)
            td.precomputed_knn = compute_nn(embeddings=embeddings)
            td.reduce(method='UMAP')
            st.session_state.embedding_type_select = 'documents'
            st.session_state.text_data_dict.update({'documents': td})
            st.session_state.current_reduction = td.last_reduction
            st.session_state.data_type_dict.update({'Metadata': ['csv']})
        elif data_type == 'Metadata':
            try:
                metadata = pd.read_csv(sf)
            except:
                metadata = pd.read_csv(sf, encoding='latin')
            td = fetch_td(st.session_state.embedding_type_select)
            td.metadata = metadata
            st.session_state.source_file = metadata
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
    # Set the text column in the session state
    update_settings(['text_column'])
    text_col = st.session_state.text_column
    if text_col is not None:
        # Pull the text from the metadata file and prep
        sf = st.session_state.source_file.dropna(axis=0, subset=text_col)
        docs = [str(d) for d in sf[text_col]]

        # Create a new TextData object with the text as its docs
        data_type = st.session_state.data_type
        if data_type == 'Tabular data with text column':
            text_type = st.session_state.embedding_type
            td = TextData(docs=docs, metadata=sf)
            st.session_state.embedding_type_select = text_type
        elif data_type == 'Metadata':
            # Setting the docs for the current TextData object
            text_type = 'documents'
            td = fetch_td('documents')
            td.docs = docs

            # Running cluster keywords, if any cluster models have been run; this
            # is super kldugy right now and needs revision
            if bool(td.reductions):
                cr = st.session_state.current_reduction
                label_df = td.reductions[cr].label_df
                if label_df is not None:
                    id_strs = list(td.reductions[cr].cluster_models.keys())
                    with st.spinner('Generating cluster keywords...'):
                        for id_str in id_strs:
                            td.reductions[cr].generate_cluster_keywords(
                                docs=docs,
                                id_str=id_str
                            )

        # Set some other session state variables
        st.session_state.text_data_dict.update({text_type: td})
        st.session_state.hover_columns = [text_col]
        st.session_state.source_file = sf
        st.session_state.enable_generate_button = True
    return


def fetch_embeddings():
    """Generates embeddings for the user's text, along with the associated \
    PCA reduction for initial visualization."""
    # Get the current (embedding-less) TD object
    old_name = st.session_state.embedding_type_select
    td = deepcopy(fetch_td(old_name))

    # Fetch the embeddings and rename the current object, if it exists
    if td is not None:
        # Get things ready to rename the TextData object
        emb_type = st.session_state.embedding_type
        model_name = st.session_state.embedding_model
        new_name = emb_type + ' (' + model_name + ')'
        st.session_state.embedding_type_select = new_name

        # Add TD object to the dict in session state with the new, embedding-
        # specific name.
        st.session_state.text_data_dict.update({new_name: td})

        # Get the embeddings and running dimensionality reduction
        td.embed(model_name=model_name)
        reduce_dimensions()

        # Delete the generic TD object from the session state dict; only
        # triggered the first time embeddings are run
        if old_name == 'document':
            del st.session_state.text_data_dict[old_name]

    # Return an error if it doesn't exist
    else:
        st.error('Please specify a text column to embed.')
    return


def reduce_dimensions():
    """Reduces the dimensonality of the currently-chosen embeddings."""
    td = fetch_td(st.session_state.embedding_type_select)
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
    # Fetch the algorithm and hyperparameter choices from the main menu
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

    # Fetch and reformat the column name for the eventual cluster IDs
    col_name = st.session_state._cluster_column_name
    col_name = col_name.replace(' ', '_')

    # Fetch the current TextData object and run the clustering algorithm
    td = fetch_td(st.session_state.embedding_type_select)
    td.cluster(
        reduction=st.session_state.current_reduction,
        method=algo,
        id_str=col_name,
        main_kwargs=main_kwargs,
        aux_kwargs=aux_kwargs
    )

    # Optionally generate cluster-specific keywords, if metadata are available
    if td.docs is not None:
        td.generate_cluster_keywords(
            reduction=st.session_state.current_reduction,
            id_str=col_name
        )

    return


def switch_reduction():
    """Wrapper function for choosing a new data reduction. Kludgy, but
    apparenty necessary for resetting the variable used to color points in
    the main plot.
    """
    update_settings(['current_reduction'])
    return


def generate_cluster_keywords():
    """Generates cluster names as topic lists. Currently only one method,
    cluster-based TF-IDF, from BERTopic, is supported.
    """
    td = fetch_td(st.session_state.embedding_type_select)
    reduction = st.session_state.current_reduction
    model = st.session_state.clustering_algorithm
    td.generate_cluster_keywords(
        reduction=reduction,
        method='TF-IDF',
        model=model,
        id_str=st.session_state._cluster_column_name
    )
    return


def cluster_stats_to_text(cm):
    """Writes a short text report with quantitative clustering metrics."""
    counts = cm.counts
    ids = list(counts.keys())
    out = {}
    for id in ids:
        id_text = "Cluster size: " + str(int(cm.counts[id])) + '\n'
        id_text += "Size rank: " + str(int(cm.count_ranks[id]))
        id_text  += ' of ' + str(len(ids)) + '\n'
        id_text += "Cluster variance: " + str(cm.vars[id]) + '\n'
        id_text += "Variance rank: " + str(int(cm.var_ranks[id]))
        id_text += ' of ' + str(len(ids))
        out.update({id: id_text})
    return out


def reset_defaults(dict, main_key):
    """Resets a set of session state variables to their defaults values."""
    var_dict = dict[main_key]
    lower_name = var_dict['lower_name']
    for k in list(var_dict['defaults'].keys()):
        v = var_dict['defaults'][k]
        sess_var = lower_name + '_' + k
        st.session_state[sess_var] = v
    return


def run_auto_clustering(
    id_str=None,
    metric='silhouette_score'
    ):
    pass
