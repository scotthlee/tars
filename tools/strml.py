"""Functions for working specifically with objects in the Streamlit session
state.
"""
import re
import logging
import os
import pandas as pd
import streamlit as st


from tools.generic import compute_nn


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
    elif data_type == 'Metadata':
        try:
            metadata = pd.read_csv(sf)
        except:
            metadata = pd.read_csv(sf, encoding='latin')
        st.session_state.metadata = metadata
    return


def set_text():
    st.session_state.text = [
        d for d in st.session_state.source_file[st.session_state._text_column]
    ]
    return


def clear_source_files():
    st.session_state.source_file = None
    st.session_state.source_type = ''
    st.session_state.source_texxt = ''
    st.session_state.all_text = st.session_state.prompt
    return
