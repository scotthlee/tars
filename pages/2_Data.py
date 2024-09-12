import numpy as np
import pandas as pd
import streamlit as st
import openai
import os

from dotenv import load_dotenv
from azure.identity import ClientSecretCredential

from tools import oai, generic, strml

st.set_page_config(page_title='Editor',
                layout='wide',
                page_icon='📖')

st.data_editor(data=st.session_state.source_file)
