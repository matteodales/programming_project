from tkinter import N
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json as js
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


oscar_df = pd.read_csv('the_oscar_award.csv')
metadata_df = pd.read_csv('movies_metadata.csv')
model_df = pd.read_csv('model_df.csv')

oscar_df = oscar_df.drop(['year_ceremony','ceremony'], axis=1)
oscar_df = oscar_df.dropna(subset=['film'])

st.header('The Oscars analysis')

sec = st.sidebar.radio('Sections:', ['Data cleaning', 'Academy Award fun facts', 'Other plots', 'Predictive model'])

if sec == 'Predictive model':
       st.subheader('Predictive Model')
       st.write('')

       runtime = st.checkbox('Runtime')
       popularity = st.checkbox('Popularity')
       budget = st.checkbox('Budget')
       revenue = st.checkbox('Revenue')
       genre = st.checkbox('Genre')
       language = st.checkbox('Language')
       prod_country = st.checkbox('Producting Country')
       prod_company = st.checkbox('Producting Company')

       features = []
       if runtime: features.append('runtime')
       if popularity: features.extend(['popularity','vote_average','vote_count'])
       if budget: features.append('budget')
       if revenue: features.append('revenue')
       if genre: features.extend(['genre Animation', 'genre Comedy','genre Family', 'genre Adventure', 'genre Fantasy', \
       'genre Romance','genre Drama', 'genre Action', 'genre Crime', 'genre Thriller','genre Horror', 'genre History', 'genre Science Fiction', \
       'genre Mystery', 'genre War', 'genre Foreign', 'genre Music','genre Documentary', 'genre Western', 'genre TV Movie'])
       if language: features.extend(['language_en', 'language_fr', 'language_it','language_ja', 'language_de', 'language_es', 'language_ru'])
       if prod_country: features.extend(['country_us', 'country_uk', 'country_fr', 'country_ge', 'country_it','country_ca', 'country_ja'])
       if prod_company: features.extend(['prod_warner','prod_mgm', 'prod_paramount', 'prod_20centuryfox', 'prod_universal'])
       
       x_oscar = model_df[features]
       y_oscar = model_df['nominated_at_least_once']

       x_train, x_test, y_train, y_test = train_test_split(x_oscar, y_oscar, test_size=0.1, random_state=1)

       model = RandomForestClassifier()
       model.fit(x_train, y_train)
       y_pred = model.predict(x_test)

       st.write('General accuracy: ', sum(y_pred == y_test) / len(y_test))
       st.write('Accuracy on nominated films: ', sum((y_pred == y_test) & (y_test == 1))/sum(y_test==1))
       st.write('Accuracy on not nominated films: ', sum((y_pred == y_test) & (y_test == 0))/sum(y_test==0))

       st.write(y_test)
       st.write(y_pred)

