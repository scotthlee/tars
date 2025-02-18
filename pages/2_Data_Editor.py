import streamlit as st
import pandas as pd

from copy import deepcopy

from tools import strml, data


# Fire up the page
st.set_page_config(
    page_title='Data Editor',
    layout='wide',
    page_icon='💾👓',
    menu_items={
        'Report a Bug': 'https://github.com/scotthlee/nlp-tool/issues/new/choose',
        'About': st.session_state.about_text
    }
)

# Set the viewer defaults
if 'data_editor_height' not in st.session_state:
    st.session_state.data_editor_height = 800
if 'display_filter_column' not in st.session_state:
    st.session_state.display_filter_column = None
if 'display_filter_values' not in st.session_state:
    st.session_state.display_filter_values = []
if 'display_column_select' not in st.session_state:
    st.session_state.display_column_select = []

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
    if has_reduction:
        # Setting the basic components
        cr = st.session_state.current_reduction
        has_clusters = td.reductions[cr].label_df is not None
        rd_df = td.reductions[cr].points

        # Assembling the current version of the full dataset (reduction,
        # metadata, and then cluster labels)
        full_dataset = rd_df
        if has_metadata:
            full_dataset = pd.concat([td.metadata, rd_df], axis=1)
        if has_clusters:
            cluster_labels = td.reductions[cr].label_df
            cluster_cols = cluster_labels.columns.values
            full_dataset = pd.concat([full_dataset, cluster_labels])

        # Applying filters; code is hideous; will revise later
        filtering_on = st.session_state.display_filter_values != []
        if filtering_on:
            to_display = full_dataset[
                full_dataset[
                    st.session_state.display_filter_column
                ].isin(st.session_state.display_filter_values)
            ]
        else:
            to_display = full_dataset

        # Applying column selection
        columns_chosen = st.session_state.display_column_select != []
        if columns_chosen:
            to_display = to_display[st.session_state.display_column_select]

        # Record the indices from the full dataset we're currently showing so
        # that we can update the correct row when changes are made
        active_ids = to_display.index.values

        # Form for choosing which columns to display
        with st.sidebar:
            with st.expander('Select', expanded=True):
                st.multiselect(
                    label='Column Choice',
                    default=st.session_state.display_column_select,
                    options=full_dataset.columns.values,
                    key='_display_column_select',
                    on_change=strml.update_settings,
                    kwargs={'keys': ['display_column_select'], 'toast': False},
                    help='Which columns would you like to display? Note: this \
                    will not change the columns in the dataset itself--it will \
                    only change which ones you can see and edit here.'
                )
            with st.expander('Filter', expanded=True):
                st.selectbox(
                    label='Filter On',
                    key='_display_filter_column',
                    options=st.session_state._display_column_select,
                    on_change=strml.update_settings,
                    kwargs={'keys': ['display_filter_column']},
                    help='If you would like ot filter the data by one of \
                    the columns, choose the column here.'
                )
                if st.session_state.display_filter_column is not None:
                    filter_values = full_dataset[
                        st.session_state.display_filter_column
                    ].unique()
                else:
                    filter_values = []
                st.multiselect(
                    label='Filter Values',
                    key='_display_filter_values',
                    options=filter_values,
                    on_change=strml.update_settings,
                    kwargs={'keys': ['display_filter_values']},
                    help='Only rows with this value of your chosen filter \
                    column will be shown.'
                )

            with st.expander('Options', expanded=True):
                st.number_input(
                    label='Viewer Height',
                    key='_data_editor_height',
                    value=st.session_state.data_editor_height,
                    on_change=strml.update_settings,
                    kwargs={'keys': ['data_editor_height'], 'toast': False},
                    help='How tall the data viewer to the right should be, \
                    measured in pixels.'
                )

            def change_data():
                '''Merges data from the editor into the corresponding TextData
                object's metadata and cluster labels.
                '''
                current_view = st.session_state.data_display
                changed_dict = current_view['edited_rows']
                changed_row = list(current_view['edited_rows'].keys())[0]
                changed_index = active_ids[int(changed_row)]
                changed_column = str(list(changed_dict[changed_row].keys())[0])
                new_value = changed_dict[changed_row][changed_column]
                if has_metadata:
                    if changed_column in td.metadata.columns.values:
                        td.metadata.loc[changed_index, changed_column] = new_value
                if has_clusters:
                    if changed_column in td.reductions[cr].label_df:
                        td.reductions[cr].label_df.loc[changed_index, changed_column] = new_value

        # Render the dataframe, hiding the dim redux columns to avoid botching
        # the current projection
        st.data_editor(
            data=pd.DataFrame(to_display),
            key='data_display',
            on_change=change_data,
            height=st.session_state.data_editor_height
        )
