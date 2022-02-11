import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json as js
import streamlit as st


oscar_df = pd.read_csv('the_oscar_award.csv')
metadata_df = pd.read_csv('movies_metadata.csv')

oscar_df = oscar_df.drop(['year_ceremony','ceremony'], axis=1)

mask = (oscar_df.category == 'HONORARY AWARD') | \
       (oscar_df.category == 'SPECIAL AWARD') | \
       (oscar_df.category == 'IRVING G. THALBERG MEMORIAL AWARD') | \
       (oscar_df.category == 'JEAN HERSHOLT HUMANITARIAN AWARD') | \
       (oscar_df.category == 'SPECIAL ACHIEVEMENT AWARD') | \
       (oscar_df.category == 'HONORARY FOREIGN LANGUAGE FILM AWARD') | \
       (oscar_df.category == 'SPECIAL FOREIGN LANGUAGE FILM AWARD')

oscar_df = oscar_df[~mask]

oscar_df = oscar_df.dropna(subset=['film'])

st.text('Fixed width text')
st.text('Fixed width text')