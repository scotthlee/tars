import numpy as np
import pandas as pd
import streamlit as st
import openai
import os
import plotly.express as px
import zipfile

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

# load the API key
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
            'type': 'openai',
            'tokens_in': 8191
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
    st.session_state.embeddings = None

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
    st.session_state.source_file = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'text_column' not in st.session_state:
    st.session_state.text_column = None
if 'text' not in st.session_state:
    st.session_state.text = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = 'Tabular data with text column'
if 'data_type_dict' not in st.session_state:
    st.session_state.data_type_dict = {
        'Tabular data with text column': ['csv'],
        'Bulk documents': ['pdf', 'txt'],
        'Premade embeddings': ['csv', 'tsv'],
        'Metadata': ['csv']
    }

# Setting up the dimensionality reduction options
if 'map_in_3d' not in st.session_state:
    st.session_state.map_in_3d = True
if 'reduction' not in st.session_state:
    st.session_state.reduction = None
if 'reduction_method' not in st.session_state:
    st.session_state.reduction_method = 'UMAP'
if 'umap_n_neighbors' not in st.session_state:
    st.session_state.umap_n_neighbors = 15
if 'umap_min_dist' not in st.session_state:
    st.session_state.umap_min_dist = 0.1
if 'tsne_perplexity' not in st.session_state:
    st.session_state.tsne_perplexity = 30.0
if 'tsne_learning_rate' not in st.session_state:
    st.session_state.tsne_learning_rate = 1000.0
if 'tsne_n_iter' not in st.session_state:
    st.session_state.tsne_n_iter = 1000
if 'reduction_dict' not in st.session_state:
    st.session_state.reduction_dict = {}
if 'current_reduction' not in st.session_state:
    st.session_state.current_reduction = None

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
    st.session_state.marker_size = 3
if 'marker_opacity' not in st.session_state:
    st.session_state.marker_opacity = 1.0
if 'show_grid' not in st.session_state:
    st.session_state.show_grid = True
if 'hover_data' not in st.session_state:
    st.session_state.hover_data = {'d1': False, 'd2': False}
if st.session_state.map_in_3d:
    st.session_state.hover_data.update({'d3': False})

# Loading the handful of variables that don't persist across pages
to_load = ['text_column', 'data_type']
for key in to_load:
    if st.session_state[key] is not None:
        strml.unkeep(key)

with st.sidebar:
    with st.expander(label='Load',
                     expanded=st.session_state.embeddings is None):
        st.radio('What kind of data do you want to load?',
                 options=list(st.session_state.data_type_dict.keys()),
                 key='_data_type',
                 on_change=strml.update_settings,
                 kwargs={'keys': ['data_type']})
        st.file_uploader(label='Select the file(s)',
                         type=st.session_state.data_type_dict[st.session_state.data_type],
                         key='_source_file',
                         on_change=strml.load_file)
    with st.expander('Embed', expanded=st.session_state.embeddings is None):
        if st.session_state.source_file is not None:
            st.selectbox('Text Column',
                         key='_text_column',
                         index=None,
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
        st.button(label='Generate embeddings',
                  key='_embed_go',
                  on_click=oai.fetch_embeddings)
        if st.session_state.embeddings is not None:
            st.download_button(label='Download embeddings',
                               data=st.session_state.embeddings.to_csv(index=False),
                               file_name='embeddings.csv',
                               mime='text/csv',
                               key='_embed_save')
    with st.expander('Reduce', expanded=st.session_state.reduction is None):
        st.selectbox(label='Method',
                     options=['UMAP', 't-SNE', 'PCA'],
                     index=None,
                     key='_reduction_method',
                     placeholder=st.session_state.reduction_method,
                     on_change=strml.update_settings,
                     kwargs={'keys': ['reduction_method']},
                     help='The algorithm used to reduce the dimensionality \
                     of the embeddings to make them viewable in 2- or 3-D.')
        if st.session_state.reduction_method == 'UMAP':
            st.slider('Nearest neighbors',
                      min_value=2,
                      max_value=200,
                      value=st.session_state.umap_n_neighbors,
                      key='_umap_n_neighbors',
                      on_change=strml.update_settings,
                      kwargs={'keys': ['umap_n_neighbors']},
                      help='This parameter controls how UMAP balances \
                      local versus global structure. Low values will force \
                      UMAP to concentrate on very local structure, while \
                      large values will push UMAP to look at larger \
                      neighborhoods of each point when estimating \
                      the mainfold structure of the data.')
            st.slider('Minimum distance',
                      min_value=0.0,
                      max_value=1.0,
                      step=0.001,
                      value=st.session_state.umap_min_dist,
                      key='_umap_min_dist',
                      on_change=strml.update_settings,
                      kwargs={'keys': ['umap_min_dist']},
                      help='Controls how tightly UMAP is allowed to pack \
                      points together.')
        if st.session_state.reduction_method == 't-SNE':
            st.slider('Perplexity',
                      min_value=5.0,
                      max_value=50.0,
                      value=st.session_state.tsne_perplexity,
                      key='_tsne_perplexity',
                      on_change=strml.update_settings,
                      kwargs={'keys': ['tsne_perplexity']})
            st.slider('Learning rate',
                      min_value=100.00,
                      max_value=1000.00,
                      value=st.session_state.tsne_learning_rate,
                      key='_tsne_learning_rate',
                      on_change=strml.update_settings,
                      kwargs={'keys': ['tsne_learning_rate']})
            st.slider('Number of iterations',
                      min_value=200,
                      max_value=10000,
                      value=st.session_state.tsne_n_iter,
                      key='_tsne_n_iter',
                      on_change=strml.update_settings,
                      kwargs={'keys': ['tsne_n_iter']})
        st.toggle(label='3D',
                 key='_map_in_3d',
                 value=st.session_state.map_in_3d,
                 on_change=strml.update_settings,
                 kwargs={'keys': ['map_in_3d']},
                 help='Whether to reduce the embeddings to 3 dimensions \
                 (instead of 2).')
        st.button('Start Reduction',
                  on_click=generic.reduce_dimensions)
    with st.expander('Visualize',
                     expanded=st.session_state.reduction is not None):
        st.selectbox('Choose a reduction',
                     index=None,
                     key='_current_reduction',
                     placeholder=st.session_state.current_reduction,
                     options=list(st.session_state.reduction_dict.keys()),
                     on_change=strml.update_settings,
                     kwargs={'keys': ['current_reduction']})
        if st.session_state.metadata is not None:
            st.selectbox('Color points by',
                         index=None,
                         options=st.session_state.metadata.columns.values,
                         key='_color_column',
                         on_change=strml.update_settings,
                         kwargs={'keys': ['color_column']})
            st.multiselect('Show on hover',
                           options=st.session_state.metadata.columns.values,
                           key='_hover_columns',
                           on_change=strml.update_settings,
                           kwargs={'keys': ['hover_columns']},
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
                  be. 1 is fully opaque, and 0 is fully transparent.')
        st.slider('Plot height',
                  min_value=100,
                  max_value=1200,
                  step=10,
                  key='_plot_height',
                  value=st.session_state.plot_height,
                  on_change=strml.update_settings,
                  kwargs={'keys': ['plot_height']},
                  help='How tall you want the scatterplot to be. It will fill \
                  the width of the screen by default, but the height is \
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
    if st.session_state.embeddings is not None:
        # Construct the hover columns
        hover_data = st.session_state.hover_data
        if st.session_state.hover_columns is not None:
            hover_data.update(
                {col: True for col in st.session_state.hover_columns}
            )

        # Assemble any metadata
        display_data = st.session_state.reduction_dict[
            st.session_state.current_reduction
        ]
        if st.session_state.metadata is not None:
            display_data = pd.concat([display_data, st.session_state.metadata],
                                     axis=1)
        if st.session_state.map_in_3d:
            fig = px.scatter_3d(data_frame=display_data,
                                x='d1', y='d2', z='d3',
                                hover_data=hover_data,
                                color=st.session_state.color_column,
                                opacity=st.session_state.marker_opacity,
                                height=st.session_state.plot_height)
        else:
            fig = px.scatter(data_frame=display_data,
                             x='d1', y='d2',
                             hover_data=hover_data,
                             color=st.session_state.color_column,
                             opacity=st.session_state.marker_opacity,
                             height=st.session_state.plot_height)
        fig.update_traces(marker=dict(size=st.session_state.marker_size))
        st.plotly_chart(fig, use_container_width=True)
