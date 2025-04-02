import streamlit as st
import pandas as pd
import numpy as np

from copy import deepcopy

from tools import strml, data


def change_data(mode='edit', column_name=None, dtype=None):
    '''Merges data from the editor into the corresponding TextData
    object's metadata and cluster labels.
    '''
    if mode == 'edit':
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
    elif mode == 'add':
        dtype_dict = {
            'Text': str,
            'Integers': int,
            'Floating-Point Numbers': float
        }
        dtype = dtype_dict[dtype]
        num_rows = td.metadata.shape[0]
        td.metadata[column_name] = np.empty(
            shape=num_rows,
            dtype=dtype
        )
    elif mode == 'remove':
        for col in column_name:
            if col in td.metadata.columns.values:
                td.metadata.drop(col, axis=1, inplace=True)
            elif has_clusters:
                if col in td.reductions[cr].label_df.columns.values:
                    td.reductions[cr].label_df.drop(col, axis=1, inplace=True)
    return

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
if 'display_column_show' not in st.session_state:
    st.session_state.display_column_show = []
if 'display_column_hide' not in st.session_state:
    st.session_state.display_column_hide = []

# Specifying the current text data object for shorthand
td_name = st.session_state.embedding_type_select
has_data = td_name is not None
has_source = st.session_state.source_file is not None
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
        columns_chosen = st.session_state.display_column_show != []
        if columns_chosen:
            to_display = to_display[st.session_state.display_column_show]
        columns_hidden = st.session_state.display_column_hide != []
        if columns_hidden:
            to_display = to_display.drop(
                            st.session_state.display_column_hide,
                            axis=1)

        # Record the indices from the full dataset we're currently showing so
        # that we can update the correct row when changes are made
        active_ids = to_display.index.values

        # Form for choosing which columns to display
        with st.sidebar:
            st.subheader('Edit')
            with st.expander('Show or Hide', expanded=False):
                st.multiselect(
                    label='Columns to Show',
                    default=st.session_state.display_column_show,
                    options=full_dataset.columns.values,
                    key='_display_column_show',
                    on_change=strml.update_settings,
                    kwargs={'keys': ['display_column_show'], 'toast': False},
                    help='Only these columns will be displayed.'
                )
                if st.session_state.display_column_show != []:
                    hideable = st.session_state.display_column_show
                else:
                    hideable = full_dataset.columns.values
                st.multiselect(
                    label='Columns to Hide',
                    default=st.session_state.display_column_hide,
                    options=hideable,
                    key='_display_column_hide',
                    on_change=strml.update_settings,
                    kwargs={'keys': ['display_column_hide'], 'toast': False},
                    help='These columns will be hidden.'
                )
            with st.expander('Add or Remove', expanded=False):
                with st.form('New Column', border=False):
                    st.text_input(
                        label='New Column Name',
                        key='new_column_name',
                        help='What would you like to name your new column?'
                    )
                    st.radio(
                        label='Data Type',
                        options=['Text', 'Integers', 'Floating-Point Numbers'],
                        key='new_column_type',
                        help='What kind of data will the new column hold?'
                    )
                    if st.form_submit_button('Add'):
                        change_data(
                            mode='add',
                            column_name=st.session_state.new_column_name,
                            dtype=st.session_state.new_column_type
                        )
                        st.rerun ()
                st.divider()
                with st.form('Remove Columns', border=False):
                    st.multiselect(
                        label='Columns to Remove',
                        options=to_display.columns.values,
                        key='columns_to_remove',
                        help='Which columns to remove. NOTE: This will \
                        permanently delete the columns from the \
                        current session data. Proceed with caution!'
                    )
                    if st.form_submit_button('Remove'):
                        change_data(
                            mode='remove',
                            column_name=st.session_state.columns_to_remove
                        )
                        st.rerun()
            with st.expander('Filter', expanded=False):
                st.selectbox(
                    label='Filter On',
                    key='_display_filter_column',
                    options=to_display.columns.values,
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
            st.divider()
            st.subheader('Options')
            st.number_input(
                label='Viewer Height',
                key='_data_editor_height',
                value=st.session_state.data_editor_height,
                on_change=strml.update_settings,
                kwargs={'keys': ['data_editor_height'], 'toast': False},
                help='How tall the data viewer to the right should be, \
                measured in pixels.'
            )

        # Render the dataframe, hiding the dim redux columns to avoid botching
        # the current projection
        st.data_editor(
            data=pd.DataFrame(to_display),
            key='data_display',
            on_change=change_data,
            height=st.session_state.data_editor_height
        )
