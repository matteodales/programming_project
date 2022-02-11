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

if sec == 'Academy Award fun facts':
       st.subheader('Academy Award fun facts')

       st.write('Exploration tool for the Academy Awards history.')
       st.write()
       sel_year = st.selectbox('Choose a year:', [''] + list(range(1927,2021)))
       if sel_year != '':
              list_cat = list(oscar_df[oscar_df.year_film == sel_year-1].category.unique())
              sel_cat = st.selectbox('Choose a category:', [''] + list_cat)
              if sel_cat != '':
                     st.dataframe(oscar_df[(oscar_df.year_film == sel_year-1) & (oscar_df.category == sel_cat)].drop(['year_film','category'],axis=1))


       st.write('Write the name of an actor/actress/director to see their list of nominations and wins.')
       name_string = st.text_input('Name: ')
       
       if name_string != '':
              nom_film = list(oscar_df[oscar_df.name == name_string]['film'])
              nom_cat = list(oscar_df[oscar_df.name == name_string]['category'])
              nom_year = list(oscar_df[oscar_df.name == name_string]['year_film']+1)
              nom_won = list(oscar_df[oscar_df.name == name_string]['winner'])

              for i in range(len(nom_film)):
                     if nom_won[i] == False:
                            st.write(nom_year[i], ': ',nom_cat[i],'for ', nom_film[i], ', Nomination.')
                     else:
                            st.write(nom_year[i], ': ',nom_cat[i],'for ', nom_film[i], ', Won.')
              
              st.write('Total number of nominations: ', len(nom_film))
              st.write('Total number of wins: ', sum(nom_won))


       #most wins and most nominations actors, actresses and film
       #andamento temporale di numero di nominees, numero di vincitori, percentuale di vittoria

       st.write('Write the title of a film to see its list of nominations and wins.')
       title_string = st.text_input('Title: ')

       if title_string != '':

              nom_year = list(oscar_df[oscar_df.film == title_string]['year_film']+1)
              

              if len(set(nom_year))>1:
                     sel_year = st.selectbox('Choose a year:', [''] + list(set(nom_year)))
              if len(set(nom_year))==1:
                     sel_year = list(set(nom_year))[0]

              if sel_year != '':
                     nom_name = list(oscar_df[(oscar_df.film == title_string) & (oscar_df.year_film == sel_year-1)]['name'])
                     nom_cat = list(oscar_df[(oscar_df.film == title_string) & (oscar_df.year_film == sel_year-1)]['category'])
                     nom_won = list(oscar_df[(oscar_df.film == title_string) & (oscar_df.year_film == sel_year-1)]['winner'])

                     for i in range(len(nom_name)):
                            if nom_won[i] == False:
                                   st.write(nom_cat[i],' ', nom_name[i], ', Nomination.')
                            else:
                                   st.write(nom_cat[i],' ', nom_name[i], ', Won.')
              
                     st.write('Total number of nominations: ', len(nom_name))
                     st.write('Total number of wins: ', sum(nom_won))

       st.write('This graphs present the record holders for number of Academy Awards nominations and wins.')

       acting_categories_m = ['ACTOR IN A SUPPORTING ROLE','ACTOR','ACTOR IN A LEADING ROLE']
       acting_categories_f = ['ACTRESS IN A SUPPORTING ROLE', 'ACTRESS','ACTRESS IN A LEADING ROLE']
       nominated_f = oscar_df[oscar_df.category.isin(acting_categories_f)]
       nominated_m = oscar_df[oscar_df.category.isin(acting_categories_m)]

       most_nominated_f = nominated_f.groupby('name')['winner'].agg(['sum','count']).sort_values('count', ascending = False).head(5)
       fig1 = plt.figure(figsize=(2,2))
       plt.bar(most_nominated_f.index, most_nominated_f['count'], label='Nominations')
       plt.xticks(rotation=45)
       plt.bar(most_nominated_f.index, most_nominated_f['sum'], color='red', label='Wins')
       plt.xticks(rotation=45, fontsize=6)
       plt.yticks(list(range(0,21,5)),fontsize=6)
       plt.legend(loc='upper right', fontsize=6)
       st.pyplot(fig1)

       fig2 = plt.figure(figsize=(3,3))
       most_nominated_m = nominated_m.groupby('name')['winner'].agg(['sum','count']).sort_values('count', ascending = False).head(5)
       plt.bar(most_nominated_m.index, most_nominated_m['count'], label='Nominations')
       plt.xticks(rotation=45)
       plt.bar(most_nominated_m.index, most_nominated_m['sum'], color='red', label='Wins')
       plt.xticks(rotation=45)
       plt.yticks(list(range(0,16,5)))
       plt.legend(loc='upper right')
       st.pyplot(fig2)

       fig3 = plt.figure(figsize=(3,3))
       most_winning_f = nominated_f.groupby('name')['winner'].agg(['sum','count']).sort_values('sum', ascending = False).head(5)
       plt.bar(most_winning_f.index, most_winning_f['sum'], label='Wins')
       plt.xticks(rotation=45)
       plt.yticks(list(range(5)))
       plt.legend(loc='upper right')
       st.pyplot(fig3)

       fig4 = plt.figure(figsize=(3,3))
       most_winning_m = nominated_m.groupby('name')['winner'].agg(['sum','count']).sort_values('sum', ascending = False).head(5)
       plt.bar(most_winning_m.index, most_winning_m['sum'], label='Wins')
       plt.xticks(rotation=45)
       plt.yticks(list(range(5)))
       plt.legend(loc='upper right')
       st.pyplot(fig4)





if sec == 'Predictive model':
       st.subheader('Predictive Model')

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
       
       if features != []:
              x_oscar = model_df[features]
              y_oscar = model_df['nominated_at_least_once']

              x_train, x_test, y_train, y_test = train_test_split(x_oscar, y_oscar, test_size=0.1, random_state=1)

              model = RandomForestClassifier()
              model.fit(x_train, y_train)
              y_pred = model.predict(x_test)

              st.write('General accuracy: ', sum(y_pred == y_test) / len(y_test))
              st.write('Accuracy on nominated films: ', sum((y_pred == y_test) & (y_test == 1))/sum(y_test==1))
              st.write('Accuracy on not nominated films: ', sum((y_pred == y_test) & (y_test == 0))/sum(y_test==0))
       else:
              st.write('Choose the features to train the model on.')

