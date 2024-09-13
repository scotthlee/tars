import numpy as np
import pandas as pd
import streamlit as st
import openai
import os
import plotly.express as px

from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from tools import oai, generic, strml


# Fire up the page
st.set_page_config(page_title='NLP Tool',
                layout='wide',
                page_icon='📖')

# Fetch an API key
@st.cache_resource
def load_oai_api_key():
    """
    Get API Key using Azure Service Principal
    """
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

load_oai_api_key()

# Set up the OpenAI API settings
st.session_state.gpt_keys = [
    'engine', 'max_tokens', 'top_p', 'temperature',
    'frequency_penalty', 'presence_penalty'
]
openai_dict = {
    'chat':{
        'gpt-35-turbo': {
            'engine': 'GPT35-Turbo-0613',
            'url': os.environ['GPT35_URL'],
            'key': os.environ["OPENAI_API_KEY"],
            'tokens_in': 16385,
            'tokens_out': 4096
        },
        'gpt-4-turbo': {
            'engine': 'edav-api-share-gpt4-128k-tpm25plus-v1106-dfilter',
            'url': os.environ['GPT4_URL'],
            'key': os.environ["OPENAI_API_KEY"],
            'tokens_in': 128000,
            'tokens_out': 4096
        }
    },
    'embeddings': {
        'ada-002': {
            'engine': 'text-embedding-ada-002',
            'url': os.environ['GPT4_URL'],
            'key': os.environ['OPENAI_API_KEY'],
            'type': 'openai'
        }
    }
}
openai_defaults = {
    'chat': {
        'model': 'gpt-4-turbo',
        'engine': 'edav-api-share-gpt4-128k-tpm25plus-v1106-dfilter',
        'max_tokens': None,
        'top_p': 0.95,
        'temperature': 0.20,
        'presence_penalty': 0.0,
        'frequency_penalty': 0.0
    },
    'embeddings': {
        'model': 'ada-002',
        'engine': 'text-embedding-ada-002'
    }
}

if 'openai_dict' not in st.session_state:
    st.session_state.openai_dict = openai_dict
if 'chat_model' not in st.session_state:
    st.session_state.chat_model = openai_defaults['chat']['model']
if 'chat_engine' not in st.session_state:
    st.session_state.engine = openai_defaults['chat']['engine']
if 'chat_engine_choices' not in st.session_state:
    st.session_state.engine_choices = list(openai_dict['chat'].keys())

if 'embedding_engine' not in st.session_state:
    st.session_state.embedding_engine = openai_defaults['embeddings']['engine']
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = openai_defaults['embeddings']['model']
if 'embedding_type' not in st.session_state:
    st.session_state['embedding_type'] = 'openai'
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = pd.read_csv('test.csv')

if 'api_type' not in st.session_state:
    st.session_state.api_type = 'azure_ad'
if 'api_version' not in st.session_state:
    st.session_state.api_version = '2023-07-01-preview'
if 'base_url' not in st.session_state:
    st.session_state.base_url = openai_dict['chat'][st.session_state.chat_model]['url']
if 'temperature' not in st.session_state:
    st.session_state.temperature = openai_defaults['chat']['temperature']
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = openai_defaults['chat']['max_tokens']
if 'top_p' not in st.session_state:
    st.session_state.top_p = openai_defaults['chat']['top_p']
if 'presence_penalty' not in st.session_state:
    st.session_state.presence_penalty = openai_defaults['chat']['presence_penalty']
if 'frequency_penalty' not in st.session_state:
    st.session_state.frequency_penalty = openai_defaults['chat']['frequency_penalty']

# Setting up the I?O objects
if 'source_file' not in st.session_state:
    st.session_state.source_file = pd.read_csv('ifrc.csv', encoding='latin')
if 'text_column' not in st.session_state:
    st.session_state.text_column = None
if 'text' not in st.session_state:
    st.session_state.text = None

# Setting up the dimensionality reduction options
if 'map_in_3d' not in st.session_state:
    st.session_state.map_in_3d = True
if 'reduction_method' not in st.session_state:
    st.session_state.reduction_method = 'PCA'

# And finally the plotting options
if 'label_columns' not in st.session_state:
    st.session_state.label_columns = None
if 'color_column' not in st.session_state:
    st.session_state.color_column = None
if 'hover_columns' not in st.session_state:
    st.session_state.hover_columns = None
if 'plot_width' not in st.session_state:
    st.session_state.plot_width = 800
if 'plot_height' not in st.session_state:
    st.session_state.plot_height = 800
if 'marker_size' not in st.session_state:
    st.session_state.marker_size = 5
if 'marker_opacity' not in st.session_state:
    st.session_state.marker_opacity = 1.0
if 'show_grid' not in st.session_state:
    st.session_state.show_grid = True

# Loading the handful of variables that don't persist across pages
to_load = ['text_column']
for key in to_load:
    if st.session_state[key] is not None:
        strml.unkeep(key)

with st.sidebar:
    with st.expander('Load', expanded=True):
        st.file_uploader('Load your Data',
                          type='csv',
                          key='_source_file',
                          on_change=strml.load_file)
    with st.expander('Embed', expanded=False):
        if st.session_state.source_file is not None:
            st.selectbox('Text Column',
                         key='_text_column',
                         options=st.session_state.source_file.columns.values,
                         on_change=strml.set_text,
                         help="Choose the column in your dataset holding the \
                         text you'd like to embed.")
        st.selectbox(
            label='Type',
            key='_embedding_type',
            on_change=strml.update_settings,
            kwargs={'keys': ['embedding_type']},
            options=['document', 'word'],
            help="Whether you'd like to make embeddings for each 'document' \
            in your dataset (documents being text contained by a single \
            spreadhseet cell) or for the words in all the documents.",
            disabled=True
        )
        st.selectbox(
            label='Model',
            key='_embedding_model',
            on_change=strml.update_settings,
            kwargs={'keys': ['embedding_model']},
            options=['ada-002']
        )
        st.button('Generate Embeddings',
                  key='_embed_go',
                  on_click=oai.fetch_embeddings)
    with st.expander('Reduce', expanded=False):
        st.selectbox('Method',
                     options=['PCA', 'UMAP', 't-SNE'],
                     key='_reduction_method',
                     on_change=strml.update_settings,
                     kwargs={'keys': ['reduction_method']},
                     help='The algorithm used to reduce the dimensionality \
                     of the embeddings to make them viewable in 2- or 3-D.')
        st.toggle(label='3D',
                 key='_map_in_3d',
                 value=st.session_state.map_in_3d,
                 on_change=strml.update_settings,
                 kwargs={'keys': ['map_in_3d']},
                 help='Whether to reduce the embeddings to 3 dimensions \
                 (instead of 2).')
        st.button('Start Reduction',
                  on_click=generic.reduce_dimensions)
    with st.expander('Visualize', expanded=False):
        st.selectbox('Color points by',
                  options=st.session_state.source_file.columns.values,
                  key='_color_column',
                  on_change=strml.update_settings,
                  kwargs={'keys': ['color_column']})
        st.multiselect('Show on hover',
                       options=st.session_state.source_file.columns.values,
                       key='_hover_columns',
                       help="Choose the data you'd like to see for each point \
                        when you hover over the scatterplot.")
        st.slider('Marker size',
                  min_value=1,
                  max_value=20,
                  key='_marker_size',
                  on_change=strml.update_settings,
                  kwargs={'keys': ['marker_size']},
                  value=st.session_state.marker_size,
                  help='How large you want the points in the scatterplot to \
                  be.')
        st.slider('Marker opacity',
                  min_value=0.0,
                  max_value=1.0,
                  step=0.001,
                  value=st.session_state.marker_opacity,
                  key='_marker_opacity',
                  on_change=strml.update_settings,
                  kwargs={'keys': ['marker_opacity']},
                  help='How opaque you want the points in the scatterplot to \
                  be, with 1 being fully opaque, and 0 being transparent.')
        st.slider('Plot height',
                  min_value=100,
                  max_value=1200,
                  step=10,
                  key='_plot_height',
                  value=st.session_state.plot_height,
                  on_change=strml.update_settings,
                  kwargs={'keys': ['plot_height']},
                  help='How tall you want the scatterplot to be. It will fill \
                  the width of the screen by default, but the heigh is \
                  adjustable.')


    st.divider()
    st.write('Settings and Tools')
    with st.expander('ChatGPT', expanded=False):
        curr_model = st.session_state.chat_model
        st.selectbox(
            label='Engine',
            key='_model_name',
            on_change=strml.update_settings,
            kwargs={'keys': ['model_name']},
            index=st.session_state.engine_choices.index(curr_model),
            options=st.session_state.engine_choices)
        st.number_input(label='Max Tokens',
                        key='_max_tokens',
                        on_change=strml.update_settings,
                        kwargs={'keys': ['max_tokens']},
                        value=st.session_state.max_tokens)
        st.slider(label='Temperature',
                key='_temperature',
                on_change=strml.update_settings,
                kwargs={'keys': ['temperature']},
                value=st.session_state.temperature)
        st.slider(label='Top P',
                key='_top_p',
                on_change=strml.update_settings,
                kwargs={'keys': ['top_p']},
                value=st.session_state.top_p)
        st.slider(label='Presence Penalty',
                min_value=0.0,
                max_value=2.0,
                key='_presence_penalty',
                on_change=strml.update_settings,
                kwargs={'keys': ['presence_penalty']},
                value=st.session_state.presence_penalty)
        st.slider(label='Frequency Penalty',
                min_value=0.0,
                max_value=2.0,
                key='_frequency_penalty',
                on_change=strml.update_settings,
                kwargs={'keys': ['frequency_penalty']},
                value=st.session_state.frequency_penalty)
        st.button('Reset', on_click=strml.reset_gpt)

# Making the main visualization
with st.container(border=True):
    if st.session_state.source_file is not None:
        if st.session_state.map_in_3d:
            fig = px.scatter_3d(st.session_state.source_file,
                                x='d1', y='d2', z='d3',
                                hover_data=st.session_state.hover_columns,
                                color=st.session_state.color_column,
                                opacity=st.session_state.marker_opacity,
                                height=st.session_state.plot_height)
        else:
            fig = px.scatter(st.session_state.source_file,
                             x='d1', y='d2',
                             hover_data=st.session_state.hover_columns,
                             color=st.session_state.color_column,
                             opacity=st.session_state.marker_opacity,
                             height=st.session_state.plot_height)
        fig.update_traces(marker=dict(size=st.session_state.marker_size))
        st.plotly_chart(fig, use_container_width=True)
