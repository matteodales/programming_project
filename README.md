# programming_project
Project for the Programming 2021/2022 class

The scope of the project is the exploration of two datasets:
- **Oscar Award dataset** (*the_oscar_award.csv*), about the Awards and nominations given between the first ceremony of 1928 and 2020.
- **Movies metadata dataset** (*movies_metadata.csv*): which contains data about around 45000 movies.

The **data exploration and cleaning** part of the project was carried out in the data_exploration_and_cleaning.ipynb file, both because I felt more comfortable using a notebook to explore the datasets and to separate the first processing from the actual file that creates the streamlit app, to be able to run it faster.
The result of the cleaning process and the merging of the two datasets has been exported as the *model_df.csv* file.

The **Academy Award exploration** section of the streamlit app is developed on the raw Oscar dataset and presents simple informations and graphs about the awards (for example the list of actors and movies for most nominations and most wins or the progressive variation of the categories of the award).

The **movie features analysis** section integrates the data from the second dataset to highlight the difference in budget, runtime, etc. for movies that have been nominated and have won or not, and in general the features most correlated with a nomination.

Finally, in the **predictive model** section I used a Decision Tree Classifier to determine wheter a film has received at least a nomination or not, basing it on a variable list of features. Some performance metrics are presented to describe how the model behaves.

