import openai
import streamlit as st

from azure.identity import ClientSecretCredential

from tools.generic import reduce_dimensions


def load_openai_settings(mode='chat'):
    """Loads all of the OpenAI API settings for a page."""
    mod_choices = {
        'chat': st.session_state.chat_model,
        'embeddings': st.session_state.embedding_model
    }
    model = mod_choices[mode]
    options = st.session_state.openai_dict[mode][model]
    openai.api_base = options['url']
    openai.api_key = options['key']
    openai.api_type = st.session_state.api_type
    openai.api_version = st.session_state.api_version
    return


def fetch_embeddings():
    if st.session_state.embedding_type == 'openai':
        load_openai_settings(mode='embeddings')
        text = st.session_state.text[0:2000]
        response = openai.Embedding.create(
            input=text,
            engine=st.session_state.embedding_engine,
        )
        embeddings = [response['data'][i]['embedding']
                      for i in range(len(text))]
    if st.session_state.embedding_type == 'huggingface':
        pass
    st.session_state.embeddings = embeddings
    reduce_dimensions(embeddings)
    return
