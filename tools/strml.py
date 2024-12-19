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
import openai

from tools.data import compute_nn
from tools.text import TextData

from tools import oai


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


def set_text(col):
    """Adds the text to be embedded to the session state. Only applies when
    tabular data is used for the input.
    """
    # Pull the text from the metadata file and prep
    text_col = st.session_state['_' + col]
    sf = st.session_state.source_file.dropna(axis=0, subset=text_col)
    sf[text_col] = sf[text_col].astype(str)
    docs = [str(d) for d in sf[text_col]]

    # Create a new TextData object with the text as its docs
    data_type = st.session_state.data_type
    if data_type == 'Tabular data with text column':
        text_type = st.session_state.embedding_type
        td = TextData(docs=docs, metadata=sf)
        st.session_state.embedding_type_select = text_type
    elif data_type == 'Metadata':
        text_type = 'documents'
        td = fetch_td('documents')
        td.docs = docs
    st.session_state.text_data_dict.update({text_type: td})

    # Set some other session state variables
    st.session_state.hover_columns = [text_col]
    st.session_state.source_file = sf
    st.session_state.enable_generate_button = True
    return


def fetch_embeddings():
    """Generates embeddings for the user's text, along with the associated \
    PCA reduction for initial visualization."""
    td = fetch_td(st.session_state.embedding_type_select)
    if td is not None:
        td.embed(model_name=st.session_state.embedding_model)
        reduce_dimensions()
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
    td = fetch_td(st.session_state.embedding_type_select)
    td.cluster(reduction=st.session_state.current_reduction,
               method=algo,
               main_kwargs=main_kwargs,
               aux_kwargs=aux_kwargs)
    if td.docs is not None:
        td.generate_cluster_keywords(
            reduction=st.session_state.current_reduction,
            model=algo
        )
    return


def switch_reduction():
    """Wrapper function for choosing a new data reduction. Kludgy, but
    apparenty necessary for resetting the variable used to color points in
    the main plot.
    """
    update_settings(['current_reduction'])
    st.session_state.color_column = None
    return


def generate_cluster_keywords():
    """Generates cluster names as topic lists. Currently only one method,
    cluster-based TF-IDF, from BERTopic, is supported.
    """
    td = fetch_td(st.session_state.embedding_type_select)
    reduction = st.session_state.current_reduction
    model = st.session_state.clustering_algorithm
    td.generate_cluster_keywords(reduction=st.session_state.current_reduction,
                                 method='TF-IDF',
                                 model=model)
    return


def generate_report():
    """Generates a summary report of the information in the user's documents,
    organized by cluster.
    """
    # Prep the individual cluster samples
    id_col = st.session_state.summary_cluster_choice.replace('_id', '')
    cd = st.session_state.cluster_dict
    algo = [k for k in list(cd.keys()) if cd[k]['lower_name'] == id_col][0]
    samp_size = int(st.session_state.summary_n_samples)
    td = fetch_td(st.session_state.embedding_type_select)
    reduction = st.session_state.current_reduction
    cm = td.reductions[reduction].cluster_models[algo]
    docs = td.docs
    doc_samps = cm.sample_docs(docs=docs, max_count=samp_size)
    cluster_ids = list(doc_samps.keys())

    # Pull the quantitative measures for use later
    quant_reports = cluster_stats_to_text(cm)

    # Prep the LLM prompt
    report = ''
    spinner_text = 'Summarizing the clusters with ChatGPT. Please wait...'
    with st.spinner(spinner_text):
        for i, id in enumerate(cluster_ids):
            instructions = "I'm working on a qualitative analysis of a public health \
            dataset. Here's a brief description of the dataset itself: "
            instructions = '\n\n' + st.session_state.summary_description + '\n\n'
            instructions += "And for context, here's a sample of documents from the \
            dataset:\n\n"
            for doc in doc_samps[id]:
                instructions += doc + '\n'
            instructions += "\nBased on these samples, what one word or phrase would \
            you use to describe the information in the documents? Also, could you \
            a brief summary of the samples that would help someone answer the \
            following questions:\n\n"
            instructions += st.session_state.summary_top_questions
            instructions += "\n\nPlease format your answers using the following \
            template: \n\n??**Keywords**: [KEYWORDS/PHRASES GO HERE]\n\n \
            **Summary**: [BRIEF SUMMARY GOES HERE]??."

            # Generate the report
            message = [
                {
                    "role": "system",
                    "content": st.session_state.gpt_persona
                },
                {
                    "role": "user",
                    "content": instructions
                },
            ]
            completion = openai.ChatCompletion.create(
                engine=st.session_state.engine,
                messages=message,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                top_p=st.session_state.top_p,
                frequency_penalty=st.session_state.frequency_penalty,
                presence_penalty=st.session_state.presence_penalty,
                stop=None
            )
            res = completion['choices'][0]['message']['content']
            res += '\n\n'

            # Add the sample docs for reference
            res += '**Samples**: \n'
            for doc in doc_samps[id]:
                res += str(doc) + '\n'

            # Add the cluster-specific metrics to the top
            cl_report = quant_reports[id] + '\n' + res
            cl_report = 'Cluster ID: ' + str(id) + '\n' + cl_report + '\n\n'
            report += cl_report
        st.session_state.summary_report = report
    st.toast(
        body='Report generation finished! Head to "Download" in the I/O \
        section to download a copy.'
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


@st.dialog('Download Session Data')
def download_dialog():
    """Creates a modal dialog for downloading session data."""
    tdd = st.session_state.text_data_dict
    emb_types = list(tdd.keys())
    dl_select = st.form('dl_select_form')
    with dl_select:
        if len(emb_types) > 1:
            st.write('Which embeddings are you working with?')
            st.selectbox('', key='_dl_emb_typ',
                         options=emb_types)
        st.write('What files would you like to download?')
        dl_original = st.checkbox('Original data')
        dl_embeddings = st.checkbox('Raw embeddings')
        dl_reduction = st.checkbox('Reduced embeddings')
        dl_labels = st.checkbox('Cluster labels')
        st.form_submit_button('Save')
    st.download_button('Test')


@st.dialog('Switch Projection')
def switch_projection():
    emb_select = st.selectbox(
        label='Base embeddings',
        options=list(st.session_state.text_data_dict.keys()),
        help='Which embeddings would you like to work with?'
    )
    td_name = emb_select
    td = st.session_state.text_data_dict[td_name]
    reduc_select = st.selectbox('Reduction',
                                index=None,
                                options=list(td.reductions.keys()),
                                placeholder=st.session_state.current_reduction)
    if st.button('Save and Exit'):
        st.session_state.hover_data = {
            k: st.session_state.hover_data[k]
            for k in list(st.session_state.hover_data.keys())
            if '_id' not in k
        }
        st.session_state.embedding_type_select = emb_select
        st.session_state.current_reduction = reduc_select
        st.rerun()
