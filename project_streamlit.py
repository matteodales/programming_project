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

       with st.expander('Exploration tool'):
              st.write('Exploration tool for the Academy Awards history.')
              st.write()
              sel_year = st.selectbox('Choose a year:', [''] + list(range(1928,2021)))
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


       with st.expander('Nominations and wins record holders'):

              st.write('This graphs present the record holders for number of Academy Awards nominations and wins.')

              values = st.slider('Select a range of years',1927, 2020 ,(1927,2020))

              min_year = values[0]
              max_year = values[1]

              acting_categories_m = ['ACTOR IN A SUPPORTING ROLE','ACTOR','ACTOR IN A LEADING ROLE']
              acting_categories_f = ['ACTRESS IN A SUPPORTING ROLE', 'ACTRESS','ACTRESS IN A LEADING ROLE']
              nominated_f = oscar_df[(oscar_df.category.isin(acting_categories_f)) & (oscar_df.year_film >= min_year-1) & (oscar_df.year_film <= max_year-1)]
              nominated_m = oscar_df[(oscar_df.category.isin(acting_categories_m)) & (oscar_df.year_film >= min_year-1) & (oscar_df.year_film <= max_year-1)]

              nom_check = st.checkbox('Most nominations')
              win_check = st.checkbox('Most wins')

              film_df = oscar_df
              film_df.drop(['category','name'],axis=1)
              film_df['nominations'] = 1
              film_df['winner'] = film_df['winner'].apply(lambda x:int(x))
              film_df['identifier'] = [str(film_df.year_film.iloc[i]) + film_df.film.iloc[i] for i in range(len(film_df.film))]
              film_df = film_df.groupby('identifier').sum()
              film_df['film'] = [film_df.index[i][4:] for i in range(len(film_df.year_film))]
              film_df['year_film'] = [int(film_df.index[i][0:4]) for i in range(len(film_df.year_film))]
              film_df = film_df.set_index(pd.Series(range(len(film_df.film))))

              if nom_check:
                     fig1, axs = plt.subplots(3, 1, figsize=(10,10))

                     most_nominated_f = nominated_f.groupby('name')['winner'].agg(['sum','count']).sort_values('count', ascending = False).head(5)
                     p1 = axs[0].bar(most_nominated_f.index, most_nominated_f['count'], label='Nominations')
                     axs[0].bar(most_nominated_f.index, most_nominated_f['sum'], color='red', label='Wins')
                     axs[0].bar_label(p1, label_type = 'center')
                     axs[0].tick_params(labelsize = 9)
                     axs[0].legend(loc='upper right', fontsize=9)
                     axs[0].set_title('Most nominated female actress')

                     most_nominated_m = nominated_m.groupby('name')['winner'].agg(['sum','count']).sort_values('count', ascending = False).head(5)
                     p1 = axs[1].bar(most_nominated_m.index, most_nominated_m['count'], label='Nominations')
                     axs[1].bar(most_nominated_m.index, most_nominated_m['sum'], color='red', label='Wins')
                     axs[1].bar_label(p1, label_type = 'center')
                     axs[1].tick_params(labelsize = 9)
                     axs[1].legend(loc='upper right', fontsize=9)
                     axs[1].set_title('Most nominated male actor')

                     most_nominated_film = film_df.sort_values('nominations',ascending=False).head(5)
                     p1 = axs[2].bar(most_nominated_film.film, most_nominated_film.nominations, label='Nominations')
                     axs[2].bar(most_nominated_film.film, most_nominated_film.winner, color='red', label='Wins')
                     axs[2].bar_label(p1, label_type = 'center')
                     axs[2].tick_params(labelsize = 9)
                     axs[2].legend(loc='upper right', fontsize=9)
                     axs[2].set_title('Most nominated film')

                     fig1.tight_layout(pad=2.0)
                     st.pyplot(fig1)
              
              if win_check:
                     fig2, axs = plt.subplots(3, 1, figsize=(10,10))

                     most_winning_f = nominated_f.groupby('name')['winner'].agg(['sum','count']).sort_values('sum', ascending = False).head(5)
                     p1 = axs[0].bar(most_winning_f.index, most_winning_f['sum'], color='red', label='Wins')
                     axs[0].bar_label(p1, label_type = 'center')
                     axs[0].tick_params(labelsize = 9)
                     axs[0].legend(loc='upper right', fontsize=9)
                     axs[0].set_title('Most winning female actress')

                     most_winning_m = nominated_m.groupby('name')['winner'].agg(['sum','count']).sort_values('sum', ascending = False).head(5)
                     p1 = axs[1].bar(most_winning_m.index, most_winning_m['sum'], color='red', label='Wins')
                     axs[1].bar_label(p1, label_type = 'center')
                     axs[1].tick_params(labelsize = 9)
                     axs[1].legend(loc='upper right', fontsize=9)
                     axs[1].set_title('Most winning male actor')

                     most_nominated_film = film_df.sort_values('winner',ascending=False).head(5)
                     p1 = axs[2].bar(most_nominated_film.film, most_nominated_film.winner, color='red', label='Wins')
                     axs[2].bar_label(p1, label_type = 'center')
                     axs[2].tick_params(labelsize = 9)
                     axs[2].legend(loc='upper right', fontsize=9)
                     axs[2].set_title('Most winning film')

                     fig2.tight_layout(pad=2.0)

                     st.pyplot(fig2)

       with st.expander('Categories evolution'):
              num_categories = oscar_df.groupby(['year_film'])
              num_categories = num_categories.agg({"category": "nunique"})

              sl_year = st.slider('Choose a year: ', 1928,2020)

              fig3, axs = plt.subplots(2, 1, figsize=(10,10))
              axs[0].plot(num_categories)
              axs[0].plot(sl_year-1, num_categories.category.loc[sl_year-1],marker='o', markersize=10, color='red')

              st.pyplot(fig3)
             





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

