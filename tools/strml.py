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
import markdown
import sklearn
import time
import spacy

from sklearn import metrics
from copy import deepcopy

from tools.data import compute_nn
from tools.text import TextData
from tools import oai, text


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
    if key in ['current_reduction', 'embedding_type_select']:
        st.session_state['color_column'] = None
        st.session_state['hover_columns'] = None
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
            st.session_state.embedding_type_select = 'Document'
            st.session_state.text_data_dict.update({'Document': td})
        elif data_type == 'Premade embeddings':
            try:
                embeddings = pd.read_csv(sf)
            except:
                embeddings = pd.read_csv(sf, encoding='latin')
            st.session_state.premade_loaded = True
            td = TextData(embeddings=embeddings)
            td.precomputed_knn = compute_nn(embeddings=embeddings)
            td.reduce(method='UMAP')
            st.session_state.embedding_type_select = 'Document'
            st.session_state.text_data_dict.update({'Document': td})
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
            st.session_state.text_data_dict.update({'Document': td})
    return


def fetch_td(td_name):
    """Wrapper to hide the session state dict call for selecting a TextData
    object to work with.
    """
    return st.session_state.text_data_dict[td_name]


def set_text():
    """Adds the text to be embedded to the session state. Only applies when
    tabular data is used for the input.
    """
    # Set the text column in the session state
    update_settings(['text_column'])
    text_col = st.session_state.text_column
    if text_col is not None:
        # Pull the text from the metadata file and prep
        sf = st.session_state.source_file.dropna(axis=0, subset=text_col)
        docs = [str(d) for d in sf[text_col]]

        # Create a new TextData object with the text as its docs
        data_type = st.session_state.data_type
        if data_type == 'Tabular data with text column':
            td = TextData(docs=docs, metadata=sf)
            td_name = 'Base'
        elif data_type == 'Metadata':
            # Setting the docs for the current TextData object
            td_name = 'Document'
            td = fetch_td(td_name)
            td.docs = docs

            # Running cluster keywords, if any cluster models have been run; this
            # is super kldugy right now and needs revision
            if bool(td.reductions):
                cr = st.session_state.current_reduction
                label_df = td.reductions[cr].label_df
                if label_df is not None:
                    id_strs = list(td.reductions[cr].cluster_models.keys())
                    with st.spinner('Generating cluster keywords...'):
                        for id_str in id_strs:
                            td.reductions[cr].generate_cluster_keywords(
                                docs=docs,
                                id_str=id_str
                            )

        # Set some other session state variables
        st.session_state.text_data_dict.update({td_name: td})
        st.session_state.hover_columns = [text_col]
        st.session_state.source_file = sf
        st.session_state.enable_generate_button = True
    return


def fetch_embeddings():
    """Generates embeddings for the user's text, along with the associated \
    PCA reduction for initial visualization."""
    # Get the current (embedding-less) TD object
    emb_type = st.session_state.embedding_type
    td = deepcopy(fetch_td('Base'))

    # Fetch the embeddings and rename the current object, if it exists
    if td is not None:
        # Get things ready to rename the TextData object
        model_name = st.session_state.embedding_model
        new_name = emb_type + ' (' + model_name + ')'

        # Splitting docs into smaller chunks, if specified
        if emb_type != 'Document':
            docs = td.docs
            doc_ids = range(len(docs))
            if emb_type == 'Sentence':
                docs, doc_ids = split_docs(
                    docs=docs,
                    doc_ids=doc_ids,
                    split_by='sentence'
                )
            elif emb_type == 'Idea':
                pass
            td.docs = docs
            doc_ids = [int(id) for id in doc_ids]
            td.metadata = td.metadata.iloc[doc_ids, :].reset_index(drop=True)
            td.metadata[st.session_state.text_column] = docs

        # Add TD object to the dict in session state with the new, embedding-
        # specific name.
        st.session_state.text_data_dict.update({new_name: td})
        st.session_state.embedding_type_select = new_name

        # Get the embeddings and running dimensionality reduction
        td.embed(model_name=model_name)
        reduce_dimensions()
    
    # Return an error if it doesn't exist
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
    # Fetch the algorithm and hyperparameter choices from the main menu
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

    # Fetch and reformat the column name for the eventual cluster IDs
    col_name = st.session_state._cluster_column_name
    col_name = col_name.replace(' ', '_')

    # Fetch the current TextData object and run the clustering algorithm
    td = fetch_td(st.session_state.embedding_type_select)
    td.cluster(
        reduction=st.session_state.current_reduction,
        method=algo,
        id_str=col_name,
        main_kwargs=main_kwargs,
        aux_kwargs=aux_kwargs
    )

    # Optionally generate cluster-specific keywords, if metadata are available
    if td.docs is not None:
        td.generate_cluster_keywords(
            reduction=st.session_state.current_reduction,
            id_str=col_name
        )

    return


def switch_reduction():
    """Wrapper function for choosing a new data reduction. Kludgy, but
    apparenty necessary for resetting the variable used to color points in
    the main plot.
    """
    update_settings(['current_reduction'])
    return


def generate_cluster_keywords():
    """Generates cluster names as topic lists. Currently only one method,
    cluster-based TF-IDF, from BERTopic, is supported.
    """
    td = fetch_td(st.session_state.embedding_type_select)
    reduction = st.session_state.current_reduction
    model = st.session_state.clustering_algorithm
    td.generate_cluster_keywords(
        reduction=reduction,
        method='TF-IDF',
        model=model,
        id_str=st.session_state._cluster_column_name
    )
    return


def generate_report():
    """Generates a summary report of the information in the user's documents,
    organized by cluster.
    """
    # Load API settings
    oai.load_openai_settings(mode='chat')

    # Fetch the correct cluster model
    td = fetch_td(st.session_state.embedding_type_select)
    reduction = st.session_state.current_reduction
    id_col = st.session_state.summary_cluster_choice
    cm = td.reductions[reduction].cluster_models[id_col]
    alg = cm.model_name

    # Prep the individual cluster samples
    samp_size = int(st.session_state.summary_n_samples)
    docs = td.docs
    doc_samps = cm.sample_docs(docs=docs, max_count=samp_size)
    cluster_ids = list(doc_samps.keys())

    # Pull the quantitative measures for use later
    quant_reports = cluster_stats_to_text(cm)

    # Prep the LLM prompt
    report = ''
    completions = ''
    spinner_text = 'Summarizing the clusters with ChatGPT. Please wait...'
    with st.spinner(spinner_text):
        for i, id in enumerate(cluster_ids):
            doc_samp = doc_samps[id]
            num_docs = len(doc_samp)
            max_doc_length = int(8000/num_docs) - 1
            doc_samp = text.truncate_text(
                doc_samp,
                max_length=max_doc_length
            )
            instructions = "I'm working on a qualitative analysis of a public health \
            dataset. Here's a brief description of the dataset itself: "
            instructions = '\n\n' + st.session_state.summary_description + '\n\n'
            instructions += "And for context, here's a sample of documents from the \
            dataset:\n\n"
            for doc in doc_samp:
                instructions += doc + '\n'
            instructions += "\nBased on these samples, what one word or phrase would \
            you use to describe the information in the documents? Also, could you \
            a brief summary of the samples that would help someone answer the \
            following questions:\n\n"
            instructions += st.session_state.summary_top_questions
            instructions += "\n\nPlease format your answers using the following \
            template: \n\n###Keywords\n[KEYWORDS/PHRASES GO HERE]\n\n \
            ###Summary\n[BRIEF SUMMARY GOES HERE]."

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
            res += '###Samples\n'
            for doc in doc_samp:
                res += str(doc) + '\n'

            # Save the completions for writing the full summary
            completions += res

            # Add the cluster-specific metrics to the top
            cl_report = quant_reports[id] + '\n\n' + res
            cl_report = '##Cluster ID: ' + str(id) + '\n' + cl_report + '\n\n'
            report += cl_report

        # Write the summary report based on the cluster results
        # Generate the report
        instructions = "I'm working on a qualitative analysis for a public \
        health dataset. I've grouped the documents by keyword and theme and \
        included samples from each group for you to review below:"
        instructions += '\n\n' + completions + '\n\n'
        instructions += "Based on these samples, please write a medium-length \
        summary (300 to 500 words) of the whole dataset that addresses the \
        following questions for me:\n\n"
        instructions += st.session_state.summary_top_questions + '\n\n'
        instructions += "Please only provide your summary--do not provide any \
        of the information in the samples I have provided."
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
        res = "#Overall Summary\n\n" + res + "\n\n\n"
        res += "#Cluster-Specific Summaries\n\n"

        # Adding the LLM disclaimer
        disclaimer = "\n\n#Generative AI Disclaimer\n"
        disclaimer += "This report generated by " + st.session_state.chat_model + "."

        # Save the report to the session state
        full_report = res + report + disclaimer
        report_text = full_report.replace('#', '')
        report_html = markdown.markdown(
            text=full_report,
            output_format='html',
            extensions=['nl2br']
        )
        report_dict = {
            'text': report_text,
            'html': report_html
        }
        file_type = st.session_state.summary_file_type
        st.session_state.summary_report = report_dict[file_type]
    st.toast(
        body='Report generation finished! Head to "Download" in the I/O \
        section to download a copy.'
    )
    st.rerun()
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


def run_auto_clustering(
    id_str=None,
    metric='silhouette_score'
    ):
    pass


def split_docs(docs, doc_ids, split_by='idea'):
    """Converts a set of documents to a set of sentences."""
    new_docs = []
    new_doc_ids = []
    if split_by == 'sentence':
        nlp = spacy.load(
            name='en_core_web_sm',
            enable='senter',
            config={'nlp': {'disabled': []}}
        )
        new_docs = [[s for s in nlp(d).sents] for d in docs]
        new_doc_ids = [[doc_ids[i]] * len(l) for i, l in enumerate(new_docs)]
    elif split_by == 'idea':
        oai.load_openai_settings(mode='chat')
        for i, doc in enumerate(docs):
            print(i)
            wait = 2
            success = False
            while not success:
                try:
                    message = [
                        {
                            "role": "system",
                            "content": "The following text may reflect more than one \
                            coherent idea. If that's the case (there's more than one \
                            idea), please break it up into smaller pieces, where each \
                            piece reflects a single idea, and return the pieces as a \
                            Python list (e.g. ['idea one text', 'idea two text'], \
                            etc.). Leave no text behind--each part of the original \
                            must be in the returned list--and do not split text if \
                            it's only a sentence long or shorter."
                        },
                        {
                            "role": "user",
                            "content": doc
                        },
                    ]
                    completion = openai.ChatCompletion.create(
                        engine='edav-api-share-gpt4-api-nofilter',
                        messages=message,
                        temperature=0,
                        max_tokens=4096,
                        stop=None
                    )
                    res = eval(completion['choices'][0]['message']['content'])
                    new_docs.append(res)
                    new_doc_ids.append([doc_ids[i]] * len(res))
                    success = True
                except:
                    time.sleep(wait)
                    wait *= 2
    # Flattening the lists
    new_docs = [str(s) for l in new_docs for s in l]
    new_doc_ids = [str(r) for l in new_doc_ids for r in l]
    return new_docs, new_doc_ids
