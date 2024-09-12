"""Functions for working specifically with objects in the Streamlit session
state.
"""
import re
import logging
import os
import pandas as pd
import streamlit as st


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
    st.session_state.source_file = pd.read_csv(sf, encoding='latin')
    return

def set_text():
    st.session_state.text = [
        d for d in st.session_state.source_file[st.session_state._text_column]
    ]
    return

def show_source_file():
    with st.session_state.source_pane:
        with st.expander('test'):
            if st.session_state.source_type == 'pdf':
                pdf_viewer(st.session_state.source_path)
            elif st.session_state.source_type == 'csv':
                st.dataframe(st.session_state.source_file)
    return


def clear_source_files():
    st.session_state.source_file = None
    st.session_state.source_type = ''
    st.session_state.source_texxt = ''
    st.session_state.all_text = st.session_state.prompt
    return
