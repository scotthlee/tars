import numpy as np
import pandas as pd
import streamlit as st
import openai
import os
import io
import plotly.express as px
import zipfile

from dotenv import load_dotenv
from matplotlib import pyplot as plt
from azure.identity import ClientSecretCredential
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from tools import oai, data, text, strml


# Setting the "about" section text
about_text = 'TARS is a Streamlit app for \
generating and analyzing text embeddings. Broadly, the app recreates \
the analytic flow of embeddings-based topic-modeling algorithms like \
BERTopic, allowing users to generate embeddings, reduce their \
dimensionality, and cluster them in the dimensionally-reduced space. \
Like BERTopic, the app can generate lists of potential topics using a \
cluster-based variant of TF-IDF, but, by way of LLM-based iterative \
summarization, it can also generate free-text summaries of the \
information in the clusters. The app makes these summaries, as well as \
any data artifacts generated during a session, available for download \
and further analysis offline. \n\n For more information, see the full \
README at https://github.com/scotthlee/tars/'

# Fire up the page
st.set_page_config(
    page_title='Embedding Projector',
    layout='wide',
    page_icon='📽',
    menu_items={
        'Report a Bug': 'https://github.com/scotthlee/nlp-tool/issues/new/choose',
        'About': about_text
    }
)

if 'about_text' not in st.session_state:
    st.session_state.about_text = about_text

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

load_api_key()

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
        'gpt-4': {
            'engine': 'edav-api-share-gpt4-api-nofilter',
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
            'tokens_in': 2048,
            'document_limit': None
        }
    }
}
openai_defaults = {
    'chat': {
        'model': 'gpt-4',
        'engine': 'edav-api-share-gpt4-api-nofilter',
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
if 'gpt_persona' not in st.session_state:
    st.session_state.gpt_persona = "You are a health communications specialist \
    with expertise in qualitative analysis."

if 'embedding_engine' not in st.session_state:
    st.session_state.embedding_engine = openai_defaults['embeddings']['engine']
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = openai_defaults['embeddings']['model']
if 'embedding_type' not in st.session_state:
    st.session_state.embedding_type = 'document'
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'enable_generate_button' not in st.session_state:
    st.session_state.enable_generate_button = False

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
if 'embedding_type_select' not in st.session_state:
    st.session_state.embedding_type_select = None
if 'reduction_select' not in st.session_state:
    st.session_state.reduction_select = None
if 'premade_loaded' not in st.session_state:
    st.session_state.premade_loaded = False
if 'text_data_dict' not in st.session_state:
    st.session_state.text_data_dict = {}
if 'current_text_data' not in st.session_state:
    st.session_state.current_text_data = None
if 'source_file' not in st.session_state:
    st.session_state.source_file = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'text_column' not in st.session_state:
    st.session_state.text_column = None
if 'data_type' not in st.session_state:
    st.session_state.data_type = 'Tabular data with text column'
if 'data_type_dict' not in st.session_state:
    st.session_state.data_type_dict = {
        'Tabular data with text column': ['csv'],
        'Premade embeddings': ['csv', 'tsv']
    }

# Setting up the dimensionality reduction options
reduction_dict = {
    'UMAP': {
        'lower_name': 'umap',
        'params': ['n_neighbors', 'min_dist'],
        'defaults': {
            'n_neighbors': 15,
            'min_dist': 0.1,
        }
    },
    't-SNE': {
        'lower_name': 'tsne',
        'params': ['perplexity', 'learning_rate', 'n_iter'],
        'defaults': {
            'perplexity': 30.0,
            'learning_rate': 1000.0,
            'n_iter': 1000
        }
    }
}

if 'reduction_dict' not in st.session_state:
    st.session_state.reduction_dict = reduction_dict
if 'reduce_to_3d' not in st.session_state:
    st.session_state.reduce_to_3d = True
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
if 'current_reduction' not in st.session_state:
    st.session_state.current_reduction = None

# Setting the clustering param defaults
cluster_defaults = {
    'DBSCAN': {
            'eps': 0.50,
            'min_samples': 5,
            'n_jobs': -1
    },
    'HDBSCAN': {
            'min_cluster_size': 5,
            'min_samples': None,
            'cluster_selection_epsilon': 0.0,
            'store_centers': 'both'
    },
    'k-means': {
            'n_clusters': 8,
            'max_iter': 300
    },
    'Agglomerative': {
            'n_clusters': None,
            'metric': 'euclidean',
            'linkage': 'ward',
            'compute_distances': True,
            'distance_threshold': 0.0
    }
}

# Setting up the clustering parameters
cluster_dict = {
    'DBSCAN': {
        'sklearn_name': 'DBSCAN',
        'lower_name': 'dbscan',
        'params': ['eps', 'min_samples'],
        'param_abbrevs': ['eps', 'min_samp']
    },
    'HDBSCAN': {
        'sklearn_name': 'HDBSCAN',
        'lower_name': 'hdbscan',
        'params': ['min_cluster_size', 'min_samples', 'store_centers']
    },
    'k-means': {
        'sklearn_name': 'KMeans',
        'lower_name': 'kmeans',
        'params': ['n_clusters', 'max_iter'],
        'param_abbrevs': ['k', 'max_iter']
    },
    'Agglomerative': {
        'sklearn_name': 'AgglomerativeClustering',
        'lower_name': 'aggl',
        'params': [
            'metric', 'linkage', 'n_clusters',
            'compute_distances', 'distance_threshold'
        ],
        'param_abbrevs': [
            'metric', 'link', 'comp_dist',
            'dist_thr'
        ]
    }
}

# Assigning initial session state values for the clustering models; the
# 'dbscan_eps' reference is arbitrary, as any session state key would work.
if 'dbscan_eps' not in st.session_state:
    for method in list(cluster_defaults.keys()):
        ln = cluster_dict[method]['lower_name']
        for param in cluster_dict[method]['params']:
            pn = ln + '_' + param
            p_def = cluster_defaults[method][param]
            st.session_state[pn] = p_def

# Setting the higher-level clusterinv variables
if 'cluster_dict' not in st.session_state:
    st.session_state.cluster_dict = cluster_dict
if 'clustering_algorithm' not in st.session_state:
    st.session_state.clustering_algorithm = 'DBSCAN'
if 'cluster_kwargs' not in st.session_state:
    st.session_state.cluster_kwargs = {}
if 'cluster_column_name' not in st.session_state:
    st.session_state.cluster_column_name = ''
if 'auto_clustering' not in st.session_state:
    st.session_state.auto_clustering = False
if 'cluster_metric_dict' not in st.session_state:
    st.session_state.cluster_metric_dict = {
        'Silhouette Score': 'silhouette_score',
        'Calinski-Harbasz Score': 'calinski_harabasz_score',
        'Davies-Bouldin Score': 'davies_bouldin_score'
    }

# Setting up the labeling options
if 'label_how' not in st.session_state:
    st.session_state.label_how = 'By keywords'
if 'keyword_type' not in st.session_state:
    st.session_state.keyword_type = 'TF-IDF'
if 'label_n_neighbors' not in st.session_state:
    st.session_state.label_n_neighbors = 10
if 'label_text_column' not in st.session_state:
    st.session_state.label_text_column = None

# Setting up the file download toggles
for k in ['original', 'embeddings', 'reduction', 'labels']:
    if 'dl_' + k not in st.session_state:
        st.session_state['dl_' + k] = False

# Setting up the plotting options
if 'map_in_3d' not in st.session_state:
    st.session_state.map_in_3d = True
if 'label_columns' not in st.session_state:
    st.session_state.label_columns = None
if 'color_column' not in st.session_state:
    st.session_state.color_column = None
if 'hover_columns' not in st.session_state:
    st.session_state.hover_columns = None
if 'font_size' not in st.session_state:
    st.session_state.font_size = 16
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
if 'show_legend' not in st.session_state:
    st.session_state.show_legend = True
if st.session_state.map_in_3d:
    st.session_state.hover_data.update({'d3': False})

# Setting up the summary report options
if 'summary_description' not in st.session_state:
    st.session_state.summary_description = ''
if 'summary_top_questions' not in st.session_state:
    st.session_state.summary_top_questions = 'Question 1\nQuestion 2\n...'
if 'summary_cluster_choice' not in st.session_state:
    st.session_state.cluster_choice = None
if 'summary_n_samles' not in st.session_state:
    st.session_state.summary_n_samples = 10
if 'summary_methods_section' not in st.session_state:
    st.session_state.summary_methods_section = False
if 'summary_report' not in st.session_state:
    st.session_state.summary_report = None
if 'summary_file_type' not in st.session_state:
    st.session_state.summary_file_type = 'html'

# Loading the handful of variables that don't persist across pages
to_load = ['text_column', 'data_type']
for key in to_load:
    if st.session_state[key] is not None:
        strml.unkeep(key)

# Specifying the current text data object for shorthand
td_name = st.session_state.embedding_type_select
has_data = td_name is not None
has_source = st.session_state.source_file is not None
has_report = st.session_state.summary_report is not None
tabular_source = st.session_state.data_type == 'Tabular data with text column'

# Some bools for controlling menu expansion and container rendering
if has_data:
    td = st.session_state.text_data_dict[td_name]
    has_embeddings = td.embeddings is not None
    has_reduction = bool(td.reductions)
    has_metadata = td.metadata is not None
    if has_metadata:
        st.session_state.metadata_columns = td.metadata.columns.values
    if has_reduction:
        cr = st.session_state.current_reduction
        has_clusters = td.reductions[cr].label_df is not None
        has_aggl = 'aggl' in list(td.reductions[cr].cluster_models.keys())
        if has_clusters:
            pass
    else:
        has_reduction = False
        has_clusters = False
        has_aggl = False
else:
    has_embeddings = False
    has_reduction = False
    has_metadata = False
    has_clusters = False
    has_aggl = False

with st.sidebar:
    st.subheader('I/O')
    with st.expander('Load', expanded=not (has_source or has_data)):
        st.radio(
            'What kind of data do you want to load?',
            options=list(st.session_state.data_type_dict.keys()),
            key='_data_type',
            on_change=strml.update_settings,
            kwargs={'keys': ['data_type']},
            help="The kind of data you want to load. If you don't have \
            embeddings made yet, choose tabular data to get started."
        )
        st.file_uploader(
            label='Select the file(s)',
            type=st.session_state.data_type_dict[st.session_state.data_type],
            key='_source_file',
            accept_multiple_files=False,
            on_change=strml.load_file
        )
        if has_source and (not tabular_source):
            st.selectbox(
                'Text Column',
                key='_text_column',
                index=st.session_state.text_column,
                options=st.session_state.source_file.columns.values,
                on_change=strml.set_text,
                kwargs={'col': 'text_column'},
                help="Choose the column in your dataset holding the \
                text you'd like to embed."
            )
    with st.expander('Download', expanded=False):
        if has_embeddings:
            st.download_button(
                label='Embeddings',
                data=td.embeddings.to_csv(index=False),
                file_name='embeddings.csv',
                mime='text/csv',
                key='_embed_save',
                help='Downloads the raw embeddings.'
            )
        if has_reduction:
            cr = st.session_state.current_reduction
            rd_df = td.reductions[cr].points.to_csv(index=False)
            st.download_button(
                label='Data reduction',
                data=rd_df,
                file_name=cr + '.csv',
                mime='text/csv',
                key='_reduc_save',
                help='Downloads the current set of reduced-dimension \
                embeddings, along with any cluster IDs that were generated \
                for them.'
            )
            if has_clusters:
                keyword_dfs = []
                mods = list(td.reductions[cr].cluster_models.values())
                for mod in mods:
                    keyword_dfs.append(pd.DataFrame(mod.keywords).transpose())
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer) as writer:
                    for i, df in enumerate(keyword_dfs):
                        df.to_excel(writer, sheet_name=mods[i].model_name)
                st.download_button(
                    label='Cluster keywords',
                    file_name='cluster_keywords.xlsx',
                    data=buffer.getvalue(),
                    mime='application/vnd.ms-excel',
                    key='_label_save'
                )
            if has_aggl:
                mod = td.reductions[cr]['cluster_models']['aggl']
                fig = data.make_dendrogram(mod)
                buf = io.BytesIO()
                fig.savefig(buf, format='svg')
                st.download_button(
                    label='Dendrogram',
                    data=buf,
                    mime='image/svg',
                    file_name='dendrogram.svg',
                    help='Downloads the clustering dendrogram.'
                )
            if has_report:
                st.download_button(
                    label='Summary Report',
                    file_name='summary_report.html',
                    data=st.session_state.summary_report,
                    mime='text/html'
                )
    st.divider()
    st.subheader('Analysis')
    if not st.session_state.premade_loaded:
        with st.expander('Embed', expanded=(not has_embeddings) and
                         (has_data or has_source)):
                if tabular_source and has_source:
                    st.selectbox(
                        'Text Column',
                        key='_text_column',
                        index=st.session_state.text_column,
                        options=st.session_state.source_file.columns.values,
                        on_change=strml.set_text,
                        kwargs={'col': 'text_column'},
                        help="Choose the column in your dataset holding the \
                        text you'd like to embed."
                    )
                st.selectbox(
                    label='Type',
                    key='_embedding_type',
                    on_change=strml.update_settings,
                    kwargs={'keys': ['embedding_type']},
                    options=['document'],
                    help="Whether you'd like to make embeddings for each 'document' \
                    in your dataset (documents being text contained by a single \
                    spreadhseet cell) or for the sentences in all the documents."
                )
                st.selectbox(
                    label='Model',
                    key='_embedding_model',
                    on_change=strml.update_settings,
                    kwargs={'keys': ['embedding_model']},
                    options=['ada-002'],
                    help='The model that will generate the embeddings. For more info \
                    about the different models, see the README.'
                )
                st.button(
                    label='Generate embeddings',
                    key='_embed_go',
                    disabled=(not st.session_state.enable_generate_button),
                    on_click=strml.fetch_embeddings
                )
    with st.expander('Shrink', expanded=(not has_reduction) and
                     has_embeddings):
        st.selectbox(
            label='Method',
            options=['UMAP', 't-SNE', 'PCA'],
            key='_reduction_method',
            placeholder=st.session_state.reduction_method,
            on_change=strml.update_settings,
            kwargs={'keys': ['reduction_method']},
            help='The algorithm used to reduce the dimensionality \
            of the embeddings to make them viewable in 2- or 3-D.'
        )
        with st.form('_reduce_param_form', border=False):
            if st.session_state.reduction_method == 'UMAP':
                st.slider(
                    label='Nearest neighbors',
                    min_value=2,
                    max_value=200,
                    key='_umap_n_neighbors',
                    value=st.session_state.umap_n_neighbors,
                    help='This parameter controls how UMAP balances \
                    local versus global structure. Low values will force \
                    UMAP to concentrate on very local structure, while \
                    large values will push UMAP to look at larger \
                    neighborhoods of each point when estimating \
                    the mainfold structure of the data.'
                )
                st.slider(
                    label='Minimum distance',
                    min_value=0.0,
                    max_value=1.0,
                    step=0.001,
                    value=st.session_state.umap_min_dist,
                    key='_umap_min_dist',
                    help='Controls how tightly UMAP is allowed to pack \
                          points together.'
                )
            if st.session_state.reduction_method == 't-SNE':
                st.slider(
                    label='Perplexity',
                    min_value=5.0,
                    max_value=50.0,
                    value=st.session_state.tsne_perplexity,
                    key='_tsne_perplexity'
                )
                st.slider(
                    label='Learning rate',
                    min_value=100.00,
                    max_value=1000.00,
                    value=st.session_state.tsne_learning_rate,
                    key='_tsne_learning_rate'
                )
                st.slider(
                    label='Number of iterations',
                    max_value=10000,
                    value=st.session_state.tsne_n_iter,
                    key='_tsne_n_iter'
                )
            st.toggle(
                label='3D',
                key='_reduce_to_3d',
                value=st.session_state.reduce_to_3d,
                help='Whether to reduce the embeddings to 3 dimensions \
                (instead of 2).'
            )
            st.form_submit_button(
                label='Start Reduction',
                on_click=strml.reduce_dimensions,
                disabled=not has_embeddings,
            )
        if st.button('Reset Default Values', disabled=not has_embeddings):
            strml.reset_defaults(
                dict=st.session_state.reduction_dict,
                main_key=st.session_state.reduction_method
            )
    with st.expander('Cluster', expanded=has_reduction):
        if st.toggle(
            label='Auto Mode',
            key='_auto_clustering',
            on_change=strml.update_settings,
            kwargs={'keys': ['auto_clustering']},
            value=st.session_state.auto_clustering
        ):
            with st.form('Auto clustering', border=False):
                st.text_input(
                    label='Cluster column name',
                    key='auto_cluster_column',
                    help='What to name the column holding the cluster IDs after \
                    the algorithm runs.'
                )
                st.selectbox(
                    label='Metric',
                    index=0,
                    options=list(st.session_state.cluster_metric_dict.keys()),
                    key='auto_cluster_metric',
                    help="Which metric to use for comparing clustering \
                    algorithms. If you're not sure which one to choose, try \
                    the default."
                )
                if st.form_submit_button('Go'):
                    strml.run_auto_clustering(
                        id_str=st.session_state.auto_cluster_column,
                        metric=st.session_state.auto_cluster_metric,
                    )
        else:
            st.selectbox(
                label='Algorithm',
                options=list(cluster_dict.keys()),
                key='_clustering_algorithm',
                index=None,
                placeholder=st.session_state.clustering_algorithm,
                on_change=strml.update_settings,
                kwargs={'keys': ['clustering_algorithm']},
                help='The algorithm to use for grouping the embeddings \
                into clusters.'
            )
            current_algorithm = st.session_state.clustering_algorithm
            default_name = cluster_dict[current_algorithm]['lower_name']
            with st.form(key='_cluster_param_form', border=False):
                if current_algorithm == 'DBSCAN':
                    st.number_input(
                        label='Epsilon',
                        min_value=0.001,
                        max_value=10.0,
                        value=st.session_state.dbscan_eps,
                        key='_dbscan_eps',
                        help='The maximum distance between two samples for one \
                        to be considered as in the neighborhood of the other.'
                    )
                    st.number_input(
                        label='Minimum samples',
                        min_value=1,
                        max_value=100,
                        value=st.session_state.dbscan_min_samples,
                        key='_dbscan_min_samples',
                        help='The number of samples in a neighborhood for a \
                        point to be considered as a core point. At higher \
                        values, the algorithm will find denser clusters, and \
                        at lower values, the clusters will be more sparser.'
                    )
                elif current_algorithm == 'HDBSCAN':
                    st.number_input(
                        label='Minimum cluster size',
                        min_value=2,
                        max_value=1000,
                        key='_hdbscan_min_cluster_size',
                        value=st.session_state.hdbscan_min_cluster_size,
                        help='The minimum number of samples in a group for the \
                        group to be considered a cluster. Groupings smaller than \
                        this size will be left as noise.'
                    )
                    st.number_input(
                        label='Minimum samples',
                        min_value=1,
                        max_value=1000,
                        key='_hdbscan_min_samples',
                        value=st.session_state.hdbscan_min_samples,
                        help='The parameter k used to calculate the distance \
                        between a point x_p and its k-th nearest neighbor. When \
                        None, defaults to the minimum cluster size.'
                    )
                elif current_algorithm == 'k-means':
                    st.number_input(
                        label='Number of clusters',
                        min_value=1,
                        max_value=100,
                        key='_kmeans_n_clusters',
                        value=st.session_state.kmeans_n_clusters,
                        help='The number of clusters to form, as well as the number\
                        of centroids to generate.'
                    )
                    st.number_input(
                        label='Max iterations',
                        min_value=1,
                        max_value=500,
                        key='_kmeans_max_iter',
                        value=st.session_state.kmeans_max_iter,
                        help='The maximum number of iterations for the algorithm \
                        to run.'
                    )
                elif current_algorithm == 'Agglomerative':
                    st.selectbox(
                        label='Metric',
                        options=['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
                        key='_aggl_metric',
                        index=None,
                        placeholder=st.session_state.aggl_metric,
                        kwargs={'keys': ['aggl_metric']}
                    )
                st.text_input(
                    label='Cluster column name',
                    key='_cluster_column_name',
                    value=cluster_dict[current_algorithm]['lower_name'] + '_id',
                    help='What to name the column holding the cluster IDs after \
                    the algorithm runs.'
                )
                st.text_input(
                    label='Keyword arguments',
                    key='_cluster_kwargs',
                    value=st.session_state.cluster_kwargs,
                    kwargs={'keys': ['cluster_kwargs']},
                    help="Extra arguments to pass to the scikit-learn \
                    clustering model. Should be formatted as a Python \
                    dictionary, e.g., {'kw': kw_value}. Note: the app will \
                    not check whether these are correct before attempting \
                    to run the algorithm, so incorrect entries may crash the \
                    current session."
                )
                if st.form_submit_button(
                    label='Run algorithm',
                    disabled=not has_reduction
                ):
                    strml.run_clustering()
                    st.rerun()
            if st.button(
                label='Reset Default Values',
                disabled=not has_reduction,
                key='reset_clustering'
            ):
                strml.reset_defaults(
                    dict=st.session_state.cluster_dict,
                    main_key=current_algorithm
                )
    with st.expander('Plot', expanded=has_reduction):
        if has_reduction:
            if (has_metadata) or (has_clusters):
                display_cols = []
                if has_clusters:
                    display_cols += list(td.reductions[
                        st.session_state.current_reduction
                    ].label_df.columns.values)
                if has_metadata:
                    display_cols += list(td.metadata.columns.values)
                st.selectbox('Color points by',
                             index=None,
                             options=display_cols,
                             key='_color_column',
                             on_change=strml.update_settings,
                             kwargs={'keys': ['color_column']},
                             help='The variable used to color the points in \
                             the plot. Continuous variables will generally \
                             produce a single color (graded by the value of the \
                             variable), and discrete variables will produce \
                             a palette of discrete colors, one for each level \
                             of the variable.')
                st.multiselect('Show on hover',
                               options=display_cols,
                               key='_hover_columns',
                               on_change=strml.update_settings,
                               kwargs={'keys': ['hover_columns']},
                               help="Choose the data you'd like to see for each \
                               point when you hover over the scatterplot.")
                st.toggle('Show legend',
                          key='_show_legend',
                          value=st.session_state.show_legend,
                          on_change=strml.update_settings,
                          kwargs={'keys': ['show_legend']},
                          help='Turns the plot legend on and off.')
        st.toggle(label='3D',
                  key='_map_in_3d',
                  value=st.session_state.map_in_3d,
                  on_change=strml.update_settings,
                  kwargs={'keys': ['map_in_3d']},
                  help='Whether to plot the embeddings in 3 dimensions \
                  (instead of 2).')
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
    if has_clusters:
        with st.expander('Summarize', expanded=False):
            with st.form(key='summary_form', border=False):
                dataset_description = st.text_area(
                    label='Dataset description',
                    key='_summary_description',
                    value=st.session_state.summary_description,
                    help="Generally, what is the text in your dataset about? For \
                    example, if they come from a scientific study, you might \
                    describe the setting and goals of the study. This will \
                    serve as context for ChatGPT as it summarizes the \
                    information in the dataset."
                )
                top_questions = st.text_area(
                    label='Top questions',
                    key='_summary_top_questions',
                    value=st.session_state.summary_top_questions,
                    help="What are the most important questions you'd like \
                    answered about the text in your dataset? Please write each \
                    question on its own line."
                )
                cluster_choice = st.selectbox(
                    label='Clustering choice',
                    key='_summary_cluster_choice',
                    options=td.reductions[cr].label_df.columns.values,
                    help="Which clustering result would you like to use to \
                    group the embeddings? Each cluster will be summarized on \
                    its own, and the then those summaries will be used to \
                    produce a top-level summary for the whole datset."
                )
                n_samples = st.number_input(
                    label='Number of samples',
                    min_value=1,
                    max_value=50,
                    value=st.session_state.summary_n_samples,
                    key='_summary_n_samples',
                    help="How many samples from each cluster you'd like to send \
                    to ChatGPT for it to use as a reference when summarizing \
                    the information is in the cluster."
                )
                file_type = st.radio(
                    label='File format',
                    options=['html', 'text'],
                    horizontal=True,
                    key='_summary_file_type',
                    help="The file format for the summary report. 'text' will \
                    render the report plain text (.txt), and 'html' will \
                    render it in Markdown (.html)."
                )
                methods_toggle = st.toggle(
                    label='Include methods section',
                    key='_summary_methods_section',
                    value=st.session_state.summary_methods_section,
                    disabled=True,
                    help="Whether to include a methods section in the summary \
                    report with information about your chosen embedding model, \
                    dimensionality-reduction algorithm, and clustering \
                    algorithm."
                )
                if st.form_submit_button('Generate report'):
                    strml.update_settings(
                        keys=[
                            'summary_description',
                            'summary_top_questions',
                            'summary_n_samples',
                            'summary_cluster_choice',
                            'summary_file_type',
                            'summary_methods_section'
                        ],
                        toast=False
                    )
                    strml.generate_report()
    st.divider()
    st.subheader('Options')
    if has_reduction:
        if st.button('Switch projection'):
            strml.switch_projection()

# Making the main visualization
with st.container(border=True):
    if has_reduction:
        # Construct the hover columns
        hover_data = st.session_state.hover_data
        if st.session_state.hover_columns is not None:
            hover_data.update(
                {col: True for col in st.session_state.hover_columns}
            )

        # Assemble any metadata
        current_reduc = td.reductions[st.session_state.current_reduction]
        display_data = current_reduc.points
        has_clusters = current_reduc.label_df is not None
        if has_clusters:
            algo = st.session_state.clustering_algorithm
            cluster_cols = current_reduc.label_df.columns.values
            display_data[cluster_cols] = current_reduc.label_df.values
        if has_metadata:
            display_data = pd.concat([display_data, td.metadata], axis=1)
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
        fig.update_layout(showlegend=st.session_state.show_legend,
                          hoverlabel=dict(font_size=st.session_state.font_size))
        st.plotly_chart(fig, use_container_width=True)
