from ctypes import alignment
from tkinter import N
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skm
import time


oscar_df = pd.read_csv('the_oscar_award.csv')
metadata_df = pd.read_csv('movies_metadata.csv')
model_df = pd.read_csv('model_df.csv')

oscar_df = oscar_df.drop(['year_ceremony','ceremony'], axis=1)
oscar_df = oscar_df.dropna(subset=['film'])

corr_df = model_df.corr().stack().rename_axis(('feat1', 'feat2')).reset_index(name='value')
corr_df = corr_df[corr_df.feat1 != corr_df.feat2]

st.title('The Oscars analysis')

sec = st.sidebar.radio('Sections:', ['Data cleaning', 'Academy Award exploration', 'Movie features analysis', 'Predictive model'])

if sec == 'Data cleaning':
       st.header('Data cleaning')

       #mettere spiegazione piuttosto discorsiva dei passaggi fatti, magari divisa in pulizia oscar, metadata e unione
       #possibilità di scaricare i raw data e il dataset finale

if sec == 'Academy Award exploration':
       st.header('Academy Award exploration')

       st.subheader('Exploration tool')
       st.write('Find out information about the Academy Awards history.')
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

       st.subheader('Record holders')
       st.write('This graphs present the record holders for number of Academy Awards nominations and wins.')

       values = st.slider('Select a range of years',1927, 2020 ,(1950,2000))

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

       st.subheader('Categories evolution')
       st.write('This graphs explore the evolution of the number of given awards and nominees during the years.')

       num_categories = oscar_df.groupby(['year_film'])
       num_categories = num_categories.agg({"category": "nunique"})

       sl_year = st.slider('Choose a year: ', 1928,2020, 1950)

       fig3, axs = plt.subplots(3, 1, figsize=(10,10))
       axs[0].plot(num_categories)
       axs[0].plot(sl_year-1, num_categories.category.loc[sl_year-1],marker='o', markersize=10, color='red')
       axs[0].set_title('Number of different categories')
       st.write("In ", sl_year, " the number of different categories was ", num_categories.category.loc[sl_year-1])

       num_nominees = oscar_df.groupby('year_film').count()
       axs[1].plot(num_nominees.index,num_nominees.category)
       axs[1].plot(sl_year-1, num_nominees.category.loc[sl_year-1],marker='o', markersize=10, color='red')
       axs[1].set_title('Total number of nominations')
       st.write("In ", sl_year, " the total number of nominations ", num_nominees.category.loc[sl_year-1])

       win_prob = num_categories.category/num_nominees.category
       axs[2].plot(win_prob)
       axs[2].plot(sl_year-1, win_prob.loc[sl_year-1],marker='o', markersize=10, color='red')
       axs[2].set_title('Winning probability for a nominated film')
       st.write("In ", sl_year, " the winning probability for a nominated film was ", win_prob.loc[sl_year-1])
       
       fig3.tight_layout(pad=2.0)
       st.pyplot(fig3)

if sec == 'Movie features analysis':

       st.header('Movie features analysis')

       not_df = model_df[model_df.nominated_at_least_once == 0]
       nom_df = model_df[model_df.nominated_at_least_once == 1]
       win_df = model_df[model_df.won_at_least_once == 1]

       st.subheader('Comparison')
       st.write("Make a comparison between these features on not nominated, nominated and winning films.")

       confront = st.multiselect('Pick the features you want to compare', ['Budget', 'Revenue', 'Popularity', 'Runtime', 'Vote average', 'Vote count'])
       confront = [item.lower() for item in confront]

       if 'vote average' in confront: 
              confront.append('vote_average')
              confront.remove('vote average')
       if 'vote count' in confront: 
              confront.append('vote_count')
              confront.remove('vote count')

       for feat in ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count']:
              if feat in confront:
                     fig, axs = plt.subplots(1, 3, figsize=(8,4))
                     axs[0].hist(not_df[not_df[feat] > 0][feat],50)
                     axs[0].title.set_text('Not nominated')
                     axs[1].hist(nom_df[nom_df[feat] > 0][feat],50)
                     axs[1].set_title('Nominated')
                     axs[2].hist(win_df[win_df[feat] > 0][feat],50)
                     axs[2].set_title('Won')

                     fig.tight_layout(pad=1.0)
                     fig.suptitle(feat, fontsize=15, ha='center', y = 1.05)
                     st.pyplot(fig)

                     st.write('Average ', feat ,'for not nominated films is: ', not_df[not_df[feat] > 0][feat].mean())
                     st.write('Average ', feat ,'for nominated films is: ', nom_df[nom_df[feat] > 0][feat].mean())
                     st.write('Average ', feat ,'for winning films is: ', win_df[win_df[feat] > 0][feat].mean())
       
       st.subheader('Correlation')
       st.write("Find out the features mostly correlated with being nominated and winning an Academy Award.")

       select = st.selectbox('Select a feature', ['', 'Genre', 'Original Language','Producting Country', 'Producting Company','All Features'])
       
       if select != '':
              
              if select == 'Genre':
                     features = ['genre Animation', 'genre Comedy','genre Family', 'genre Adventure', 'genre Fantasy', 'genre Romance','genre Drama',\
                             'genre Action', 'genre Crime', 'genre Thriller','genre Horror', 'genre History', 'genre Science Fiction','genre Mystery', \
                                    'genre War', 'genre Foreign', 'genre Music', 'genre Documentary', 'genre Western', 'genre TV Movie']
              if select == 'Original Language':
                     features = ['language_en', 'language_fr', 'language_it','language_ja', 'language_de', 'language_es', 'language_ru']
              if select == 'Producting Country': 
                     features = ['country_us', 'country_uk', 'country_fr', 'country_ge', 'country_it','country_ca', 'country_ja']
              if select == 'Producting Company':
                     features = ['prod_warner','prod_mgm', 'prod_paramount', 'prod_20centuryfox', 'prod_universal']
              if select == 'All Features':
                     features = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average',\
                                   'vote_count', 'genre Animation', 'genre Comedy',\
                                   'genre Family', 'genre Adventure', 'genre Fantasy', 'genre Romance',\
                                   'genre Drama', 'genre Action', 'genre Crime', 'genre Thriller',\
                                   'genre Horror', 'genre History', 'genre Science Fiction',\
                                   'genre Mystery', 'genre War', 'genre Foreign', 'genre Music',\
                                   'genre Documentary', 'genre Western', 'genre TV Movie', 'prod_warner',\
                                   'prod_mgm', 'prod_paramount', 'prod_20centuryfox', 'prod_universal',\
                                   'country_us', 'country_uk', 'country_fr', 'country_ge', 'country_it',\
                                   'country_ca', 'country_ja', 'language_en', 'language_fr', 'language_it',\
                                   'language_ja', 'language_de', 'language_es', 'language_ru']

              col1, col2 = st.columns(2)
              with col1:
                     st.write("Correlation with being nominated:")
                     f_table_nom = pd.concat([pd.Series(features), pd.Series(list(corr_df[(corr_df['feat1']=='nominated_at_least_once') & (corr_df['feat2'].apply(lambda x: x in features))].value))], axis=1, keys=['Feature','Correlation'])
                     st.dataframe(f_table_nom)
              with col2:
                     st.write("Correlation with winning:")
                     f_table_win = pd.concat([pd.Series(features), pd.Series(list(corr_df[(corr_df['feat1']=='won_at_least_once') & (corr_df['feat2'].apply(lambda x: x in features))].value))], axis=1, keys=['Feature','Correlation'])
                     st.dataframe(f_table_win)

       with st.expander('Release date distribution'):
              st.write('The general consensus in the film industry is that the later in the year a film is published, the higher chances of awards it has: this has been\
                     attributed to the fact that awards season falls in the beginning of the successive year, and movies that have just been released leave a more vivid mark\
                     in the memories of awards voters.')

              fig, axs = plt.subplots(1, 3, figsize=(8,4))
              axs[0].hist(model_df[(model_df.nominated_at_least_once == 0) & (model_df.day>1)]['day'],50)
              axs[0].title.set_text('Not nominated')
              axs[1].hist(model_df[(model_df.nominated_at_least_once == 1) & (model_df.day>1)]['day'],50)
              axs[1].set_title('Nominated')
              axs[2].hist(model_df[(model_df.won_at_least_once == 1) & (model_df.day>1)]['day'],50)
              axs[2].set_title('Won')

              fig.tight_layout(pad=1.0)
              fig.suptitle('day', fontsize=15, ha='center', y = 1.05)
              st.pyplot(fig)

              st.write('Average ', 'day' ,'for not nominated films is: ', model_df[(model_df.nominated_at_least_once == 0) & (model_df.day>1)]['day'].mean())
              st.write('Average ', 'day' ,'for nominated films is: ', model_df[(model_df.nominated_at_least_once == 1) & (model_df.day>1)]['day'].mean())
              st.write('Average ', 'day' ,'for winning films is: ', model_df[(model_df.won_at_least_once == 1) & (model_df.day>1)]['day'].mean())

              day_rate = [len(model_df[(model_df.nominated_at_least_once == 1) & (model_df.day==x)]['day'])/len(model_df[model_df.day==x]['day']) for x in range(2,365)]
              fig, ax = plt.subplots(figsize=(8,2))
              ax.contourf([day_rate,day_rate],cmap='jet',vmin=0, vmax=max(day_rate))
              st.pyplot(fig)

              st.write('The heatmap shows how the rate of nominated films released in a day varies\
                     throughout the year. The data seems to prove the rumors right! ')
              
if sec == 'Predictive model':
       st.header('Predictive Model')

       st.write('Using this tool you can run a Decision Tree Classifier on a chosen list of\
               features to determine wheter a film has received at least a nomination or not.')

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

              x_train, x_test, y_train, y_test = train_test_split(x_oscar, y_oscar, test_size=0.1, random_state=10)

              model = DecisionTreeClassifier()
              model.fit(x_train, y_train)
              y_pred = model.predict(x_test)

              with st.spinner(text='In progress'):
                     time.sleep(2)

              st.subheader("Model's confusion matrix")
              st.table(skm.confusion_matrix(y_test,y_pred))

              #grafici che fanno vedere che ne indovina tot, quanti dice che sono nominati...
              
              st.subheader("Some performance metrics")
              st.write("F1 score: ", skm.f1_score(y_test,y_pred))
              st.write("Precision score: ", skm.precision_score(y_test,y_pred))
              st.write("Recall score: ", skm.recall_score(y_test,y_pred))
              st.write("Accuracy score: ", skm.accuracy_score(y_test,y_pred))

              with st.expander('Learn more'):
                     st.markdown("""

                     The dataset we are basing this model on is **unbalanced**, in the sense that a lot more films don't get nominated
                     for Academy Awards then the ones that do (here exactly 40416 films are in class 0, meaning they haven't been nominated and 2832 are in class 1, meaning they have).
                     
                     This means that the accuracy score is not an accurate representation of the performance of the model. To study this we can look at the **confusion matrix** for the model, that allows for a better visualization of what is happening. 
                     
                     In a confusion matrix, each row represents the instances in an actual class while each column represents the instances in a predicted class. 
                     
                     Instead of accuracy it is more informative to look at one of these metrics, that are of course intended for class 1: 
                     - Precision: it's intuitively the ability of the classifier not to label as positive a sample that is negative. It is a value that goes from 0 to 1.
                     - Recall: it's intuitively the ability of the classifier to find all the positive samples. It is a value that goes from 0 to 1.
                     - F1 score: as we can imagine, there exists some kind of tradeoff between precision and recall, the F1 score is an harmonic mean between the two values that represents a more balanced metric. 
                     
                     To find out more information about these tools:
                     - https://en.wikipedia.org/wiki/Precision_and_recall#Recall
                     - https://en.wikipedia.org/wiki/Confusion_matrix,
                     - https://en.wikipedia.org/wiki/F-score

                     """)



       else:
              st.write('Choose the features to train the model on.')

