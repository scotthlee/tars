import openai
import streamlit as st
import os

from dotenv import load_dotenv
from azure.identity import ClientSecretCredential


def load_openai_settings(mode='chat'):
    """Loads all of the OpenAI API settings for a page."""
    mode_choices = {
        'chat': st.session_state.chat_model,
        'embeddings': st.session_state.embedding_model
    }
    model = mode_choices[mode]
    options = st.session_state.openai_dict[mode][model]
    openai.api_base = options['url']
    openai.api_key = options['key']
    openai.api_type = st.session_state.api_type
    openai.api_version = st.session_state.api_version
    return
