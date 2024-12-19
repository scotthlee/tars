import openai
import streamlit as st
import os

from dotenv import load_dotenv
from azure.identity import ClientSecretCredential


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


# Fetch an API key
def load_api_key():
    """Get API Key using Azure Service Principal."""
    load_dotenv()

    # Set up credentials based on Azure Service Principal
    credential = ClientSecretCredential(
        tenant_id=os.environ["SP_TENANT_ID"],
        client_id=os.environ["SP_CLIENT_ID"],
        client_secret=os.environ["SP_CLIENT_SECRET"]
    )

    # Set the API_KEY to the token from the Azure credentials
    os.environ['OPENAI_API_KEY'] = credential.get_token(
        "https://cognitiveservices.azure.com/.default").token
