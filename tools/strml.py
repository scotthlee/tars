"""Functions for working specifically with objects in the Streamlit session
state.
"""
import re
import logging
import os
import pandas as pd
import streamlit as st

from tools.data import compute_nn, reduce_dimensions


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
            pass
        elif data_type == 'Premade embeddings':
            try:
                embeddings = pd.read_csv(sf)
            except:
                embeddings = pd.read_csv(sf, encoding='latin')
            st.session_state.embeddings = embeddings
            compute_nn()
            reduce_dimensions(reduction_method='PCA')
        elif data_type == 'Metadata':
            try:
                metadata = pd.read_csv(sf)
            except:
                metadata = pd.read_csv(sf, encoding='latin')
            st.session_state.metadata = metadata
    return


def set_text():
    """Adds the text to be embedded to the session state. Only applies when
    tabular data is used for the input.
    """
    text_col = st.session_state._text_column
    sf = st.session_state.source_file.dropna(axis=0, subset=text_col)
    sf[text_col] = sf[text_col].astype(str)
    docs = [str(d) for d in sf[text_col]]
    st.session_state.text = {'documents': docs, 'sentences': None}
    st.session_state.hover_columns = [text_col]
    st.session_state.source_file = sf
    return


def switch_reduction():
    """Wrapper function for choosing a new data reduction. Kludgy, but
    apparenty necessary for resetting the variable used to color points in
    the main plot.
    """
    update_settings(['current_reduction'])
    st.session_state.color_column = None
    return
