import pandas as pd
import numpy as np
import openai
import streamlit as st

from sklearn.decomposition import PCA


def reduce_dimensions():
    reduction_method = st.session_state.reduction_method
    dims = st.session_state.reduction_dims
    if reduction_method == 'PCA':
        pca = PCA(n_components=dims)
        pca.fit(st.session_state.embeddings)
        reduction = pca.transform(st.session_state.embeddings)
    elif reduction_method == 'UMAP':
        pass
    elif reduction_method == 't-SNE':
        pass
    colnames = ['d' + str(i + 1) for i in range(dims)]
    reduction = pd.DataFrame(reduction, columns=colnames)
    st.session_state.reduction = reduction
    st.session_state.source_file = pd.concat(st.session_state.source_file,
                                             reduction,
                                             axis=1)
    return
