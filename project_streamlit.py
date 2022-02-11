import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json as js
import streamlit as st


oscar_df = pd.read_csv('the_oscar_award.csv')
metadata_df = pd.read_csv('movies_metadata.csv')

oscar_df = oscar_df.drop(['year_ceremony','ceremony'], axis=1)
oscar_df = oscar_df.dropna(subset=['film'])

st.header('The Oscars analysis')

a = st.sidebar.radio('Sections:', ['Data cleaning', 'Academy Award fun facts', 'Other plots', 'Predictive model'])

if a == 'Predictive model':
       st.subheader('Predictive Model')
       st.write('')