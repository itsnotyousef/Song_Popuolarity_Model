import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import re
import ast
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_classif
import xgboost as xgb
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error , r2_score,accuracy_score,mean_absolute_error
import plotly.express as px
from sklearn.feature_selection import SelectKBest, f_regression, chi2, mutual_info_regression,f_classif
from scipy.stats import pearsonr, spearmanr
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
import ast
import pickle
import json
from scipy.stats import shapiro
pd.set_option('display.max_colwidth', None)
df=pd.read_csv(r"C:\Users\yousef\Downloads\SongPopularity_Milestone2.csv")
pd.set_option('display.max_columns', None)
df.head()
# **Hot100 Ranking Year:** The year in which the song achieved its ranking on the Billboard Hot 100 chart.
#
# **Hot100 Rank:** The specific ranking of the song on the Billboard Hot 100 chart during a particular year.
#
# **Acousticness:** tells us how much of a song is made with real instruments versus electronic ones: higher numbers mean more real instruments, while lower numbers mean more electronic sounds.
#
# **Danceability:** measures how easy it is to dance to a song: higher values mean it's easier to dance to, while lower values mean it might be harder to dance to.
#
# **Energy:** A measure of the song's intensity and activity, often associated with loudness and speed.
#
# **Instrumentalness:** Indicates the presence of vocals vs. instrumental elements in the song.
#
# **Liveness:** Reflects the likelihood of the song being performed live, based on audience noises and crowd sounds.
#
# **Speechiness:** Measures the presence of spoken words or speech-like elements in the song.
#
# **Tempo:** The speed or pace of the song, typically measured in beats per minute (BPM).
#
# **Valence:** Describes the musical positiveness conveyed by the song, such as happiness or cheerfulness.
#
# **Key:** The musical key or tonality of the song, which influences its mood and sound.
#
# **Time Signature:** Specifies the number of beats in each bar and the type of note that receives one beat, defining the song's rhythmic structure.
# **Data Preprocessing**
df.shape
list(df.columns)
df.info()
# All datatypes are correct
df.describe().round(2)
# Hot100 Ranking Year: The ranking years range between 1946 and 2022
#
# Hot100 Rank: The average rank is 48.32
#
# Popularity: The average popularity score is 54.12
#
# Energy: The minimum energy value is 0.01, the maximum energy value is 1.00.
#
# Instrumentalness: The average instrumentalness value is 0.05.
#
# Liveness: The liveness value ranges from 0.02 to 0.98.
#
# Loudness: The minimum loudness value is -37.84, the maximum loudness value is -0.81.
#
# Acousticness: The most acoustic songs tend to have an acousticness score of 1.00.
#
# Danceability: The average Danceability is 0.62
#
# Key: The average of key scores 5.24.
#
# Speechiness: The average Speechines is 5.24.
#
# Tempo: The fastest songs tend to have a tempo score of 232.47.
#
# Valence: The minimum valence score is 0.00, the maximum valence score is 0.99.
#
# Time Signature: The average time signature is 3.94.

# Mode: There are two modes in the dataset: "Major" and "Minor"and the most common mode is "Major" in the dataset.
df.isnull().sum()
df.duplicated().sum()
# No null or duplicated values
### Exploring unique values of some features
df['Mode'].unique()
df['Hot100 Ranking Year'].unique()
df['Key'].unique()
df['Time Signature'].unique()
df['Artist Names'].value_counts()
# In the dataset, many artists have multiple songs, and the artist with the highest contribution is 'The Karaoke Channel' with 42 songs. Following them are 'Madonna' and 'Janet Jackson' with their respective song counts.
album_counts = df['Album'].value_counts()
multiple_occurrences = album_counts[album_counts > 1]
print(multiple_occurrences)
# For albums, the most prolific is 'Greatest Hits' with 48 entries. This is followed by 'Super Hits' and '16 Most Requested Songs', each having their own respective counts.
song_counts = df['Song'].value_counts()
multiple_occurrences = song_counts[song_counts > 1]
print(multiple_occurrences)
# **Many song names are duplicated!**
df1=df[df['Song']=='I Like It']
df1
# Song names are duplicated but not all links are
df1['Spotify Link'].unique()
link_counts = df['Spotify Link'].value_counts()
link_multiple_occurrences = link_counts[link_counts > 1]
print(link_multiple_occurrences)
duplicate_links = df[df.duplicated(subset=['Spotify Link'], keep=False)]
duplicate_links
### If a link appears more than once but with different top 100 ranking years, it may not be an issue since the song could be featured in multiple significant song lists across different years. However, if the top 100 ranking years are identical for duplicated links, it presents a conflict for the model. This is uncommon because a song typically shouldn't have multiple rankings in the same year. Therefore, I will investigate rows where both the link and top 100 ranking year are duplicated. ###

duplicate_links = df[df.duplicated(subset=['Spotify Link', 'Hot100 Ranking Year'], keep=False)]
duplicate_links.shape[0]
duplicate_links
# Since there are only 12 rows where both the top 100 ranking year and the link are duplicated, I will drop these rows from the dataset.


df.drop_duplicates(subset=['Spotify Link', 'Hot100 Ranking Year'], keep=False, inplace=True)

df.reset_index(drop=True, inplace=True)
duplicate_links = df[df.duplicated(subset=['Spotify Link'], keep=False)]
duplicate_links.shape[0]
link_counts = df['Spotify URI'].value_counts()
link_multiple_occurrences = link_counts[link_counts > 1]
print(link_multiple_occurrences)
df['Hot100 Ranking Year'].value_counts()
df['PopularityLevel'].value_counts()
df['Album'].value_counts()
# The majority of songs appear to have rankings in the mid to high 90s, particularly in 2017 and 1974. This suggests that these years might have had a higher number of popular or significant songs. On the other hand, earlier years like 1955, 1950, and 1952 show lower rankings, indicating fewer popular songs or possibly a smaller dataset for those years.

### Handling Some Hidden Nulls
# Nulls in Artists' Genres are represented as "[]" in the dataset.
df[df['Artist(s) Genres'].isin(["[]"])].head()
df['Artist(s) Genres'].isin(["[]"]).sum()
# Filling null values in the Artists' Genres, which is categorical data, with the mode is a suitable approach. This will replace missing values with the most frequently occurring genre in the dataset.
df['Artist(s) Genres'] = df['Artist(s) Genres'].apply(lambda x: np.nan if x == "[]" else x)
mode_value = df['Artist(s) Genres'].mode()[0]
df['Artist(s) Genres'].fillna(mode_value, inplace=True)
print(df['Artist(s) Genres'].isin(["[]"]).sum())
# **Checking if there any other hidden Nulls**
pattern = re.compile(r'[^a-zA-Z\s]')
df[df['Song'].str.contains(pattern, na=False)].head()
pattern = r'^[^\w\s]+$'
HiddenNulls=df[df['Album'].str.match(pattern, na=False)]
HiddenNulls
# **There are hidden nulls in album column in form of "?" !!!**
HiddenNulls.shape[0]
rows_to_delete = df[df.isin(HiddenNulls.to_dict('list')).all(axis=1)].index

df.drop(rows_to_delete, inplace=True)

df.reset_index(drop=True, inplace=True)

pattern = r'^[^\w\s]+$'
HiddenNulls=df[df['Album'].str.match(pattern, na=False)]
HiddenNulls.shape[0]
pattern = r'^[^\w\s]+$'
HiddenNulls=df[df['PopularityLevel'].str.match(pattern, na=False)]
HiddenNulls.shape[0]
# I will convert the 'Album Release Date' column to only display the 'Year' since it is the most relevant information for our analysis.

df['Year'] = df['Album Release Date'].apply(lambda x: x.split('/')[-1].split('-')[0])
df.head()
# A tempo of 0 is not possible for any song, so I will remove entries with a tempo value of 0 from the dataset.

df[df['Tempo']==0].shape[0]
'''
rows_to_delete = df[df['Tempo'] <= 0].index

df.drop(rows_to_delete, inplace=True)

df.reset_index(drop=True, inplace=True)
'''
# When an attempt was made to exclude instances where Tempo was equal to 0, it resulted in a decrease in accuracy. Consequently, these instances were retained in the dataset.
# **EDA**
cat=df[['Song','Album','Album Release Date','Artist Names','Artist(s) Genres','Spotify Link','Song Image','Spotify URI','PopularityLevel','Year']]
cont=df.drop(columns=cat.columns)
# **Boxplot**
plt.figure(figsize = (15,25))
for idx, i in enumerate(cont):
    plt.subplot(12, 2, idx + 1)
    sns.boxplot(x = i, data = df,palette="mako")
    plt.title(i,color='black',fontsize=15)
    plt.xlabel(i, size = 12)
plt.tight_layout()
plt.show()
# There are outliers but we will handle them later.
# **Histplots and Kdeplots**
# Check distributions
fig, axs = plt.subplots(len(cont.columns), 2, figsize=(20, 60))

axs = axs.flatten()

for i, column in enumerate(cont.columns):

    sns.histplot(cont[column], bins=50, ax=axs[2*i])
    axs[2*i].set_title(f'Histogram of {column}')
    axs[2*i].set_xlabel(column)
    axs[2*i].set_ylabel('Frequency')

    sns.kdeplot(cont[column], ax=axs[2*i+1], fill=True)
    axs[2*i+1].set_title(f'KDE Plot of {column}')
    axs[2*i+1].set_xlabel(column)
    axs[2*i+1].set_ylabel('Density')
plt.tight_layout()
plt.show()
# The 'popularity' feature has over 250 zeros. We will further investigate to determine if these zeros represent null values or if they are

def check_distribution(data):
    skewness = stats.skew(data)
    _, shapiro_p_value = stats.shapiro(data)

    if shapiro_p_value > 0.05:
        if skewness > 0:
            return "Right-skewed"
        elif skewness < 0:
            return "Left-skewed"
        else:
            return "Normally distributed"
    else:
        return "Not normally distributed"

results = cont.apply(check_distribution)

print(results)
# Some features, like 'danceability', appear to have a distribution close to normal. However, most of the other features seem to be right-skewed.


# rows_to_delete = df[df['Tempo'] > 230].index
# df.drop(rows_to_delete, inplace=True)

# rows_to_delete = df[df['Loudness'] < -35].index
# df.drop(rows_to_delete, inplace=True)

# rows_to_delete = df[df['Speechiness'] > 0.8].index
# df.drop(rows_to_delete, inplace=True)

# df.reset_index(drop=True, inplace=True)

list(cat.columns)
# **Countplot**
# Most artists have multiple songs in the dataset.

duplicatedArtists=df[df['Artist Names'].duplicated()]
artist_counts = df['Artist Names'].value_counts()
top_30_artists = artist_counts.head(30)
plt.figure(figsize=(15, 10))
sns.countplot(y='Artist Names', data=df[df['Artist Names'].isin(top_30_artists.index)], order=top_30_artists.index,palette='mako')
plt.title('Count of Duplicated Artists (Top 30)')
plt.xlabel('Count')
plt.ylabel('Artist Names')
plt.show()
# Average popularity in these top 30 artists.
# Popularity Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='PopularityLevel', data=df,palette='mako')
plt.title('Distribution of Popularity Levels')
plt.xlabel('Popularity Level')
plt.ylabel('Count')
plt.show()
# Most songs have average popularity
# Decades VS Popularity
year_ranges = [(1950, 1960), (1960, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 2010), (2010, 2020)]

df['Year Range'] = pd.cut(df['Hot100 Ranking Year'], bins=[x[0] for x in year_ranges] + [year_ranges[-1][-1]], labels=[f"{x[0]}-{x[1]}" for x in year_ranges])

sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Year Range', hue='PopularityLevel', palette='mako')
plt.title('Count of Entries by Year Range and Popularity Level')
plt.xlabel('Year Range')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Popularity Level')
plt.grid(True)
plt.show()
# Features VS Popularity
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='PopularityLevel', y='Valence', palette='mako')
plt.title('Popularity Level vs. Valence')
plt.xlabel('Popularity Level')
plt.ylabel('Valence')
plt.grid(True)
plt.show()
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='PopularityLevel', y='Loudness', palette='mako')
plt.title('Popularity Level vs. Loudness')
plt.xlabel('Popularity Level')
plt.ylabel('Loudness')
plt.grid(axis='y')
plt.show()
popularity_levels = ['Not Popular', 'Average', 'popular']
avg_danceability = []
for level in popularity_levels:
    subset = df[df['PopularityLevel'] == level]
    avg_danceability.append(subset['Danceability'].mean())

plt.figure(figsize=(10, 6))
plt.bar(popularity_levels, avg_danceability)
plt.xlabel('Popularity Level')
plt.ylabel('Average Danceability')
plt.title('Average Danceability by Popularity Level')
plt.grid(axis='y')
plt.show()

df['minutes_length'] = df['Song Length(ms)'].apply(lambda x: x / 60000)
df.head()
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='PopularityLevel', y='minutes_length', hue='PopularityLevel', palette='mako', dodge=False)
plt.title('Average Song Length by Popularity Level')
plt.xlabel('Popularity Level')
plt.ylabel('Average Song minutes length')
plt.grid(axis='y')
plt.legend(title='Popularity Level', loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Energy', y='PopularityLevel', palette='mako')
plt.title('Average Energy by Popularity Level')
plt.xlabel('Popularity Level')
plt.ylabel('Average Energy')
plt.grid(axis='y')
plt.show()
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Hot100 Rank', y='PopularityLevel', palette='mako')
plt.title('Average Energy by Popularity Level')
plt.xlabel('Popularity Level')
plt.ylabel('Hot 100 Rank')
plt.grid(axis='y')
plt.show()
# The artists with the most popular songs are typically the most successful and famous in the industry.
#
# Most repeated years in the dataset
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Year', palette='mako')
plt.title('Count Plot for Album Release Date')
plt.xlabel('Album Release Date')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Hot100 Ranking Year', palette='mako')
plt.title('Count Plot for Album Release Date')
plt.xlabel('Album Release Date')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()
df['Year'] = df['Year'].astype(int)
# Albums with most columns
album_counts = df['Album'].value_counts().reset_index()
album_counts.columns = ['Album', 'Count']
top_30_albums = album_counts.head(50)


plt.figure(figsize=(15, 8))
sns.barplot(data=top_30_albums, x='Album', y='Count', palette='mako')
plt.title('Top 30 Albums by Number of Songs')
plt.xlabel('Album')
plt.ylabel('Number of Songs')
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()
# **Scatter plots**
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Acousticness', x='Energy', hue='PopularityLevel', palette='mako')
plt.title('Acousticness vs Energy')
plt.xlabel('Energy')
plt.ylabel('Acousticness')
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Energy',x='Loudness',hue='PopularityLevel', palette='mako')
plt.title('Energy vs Loudness')
plt.xlabel('Loudness')
plt.ylabel('Energy')
plt.grid(True)
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, y='Danceability',x='Valence',hue='PopularityLevel', palette='mako')
plt.title('Scatter Plot of Popularity vs Year')
plt.xlabel('Valence')
plt.ylabel('Danceability')
plt.grid(True)
plt.show()
popularity_values = {'Not Popular': 0, 'Average': 1, 'popular': 2}

df['PopularityNumeric'] = df['PopularityLevel'].map(popularity_values)

plt.figure(figsize=(10, 6))
plt.scatter(df['Year'], df['PopularityNumeric'], c=df['PopularityNumeric'], cmap='mako', s=100, alpha=0.8)
plt.colorbar(label='Popularity Level')
plt.xlabel('Year')
plt.ylabel('Popularity Level')
plt.title('Popularity Level vs. Year of Release')
plt.xticks(range(df['Year'].min(), df['Year'].max()+1, 10))
plt.grid(True)
plt.show()

# **Pie Charts**
def popularity_level(x):
    if x == "Average":
        return "Average"
    elif x == "Not Popular":
        return "Not Popular"
    else:
        return "popular"
df['PopularityLevel'] = df['PopularityLevel'].apply(popularity_level)
n_songs_per_category = df.groupby('PopularityLevel').size()
fig=px.pie(df,names=n_songs_per_category.index ,values=n_songs_per_category.values)
fig.update_layout(title='level of popularity')
fig.show()
original_speechiness_values = df['Speechiness'].copy()
def type_of_Song(x):
    if x >= 0.0 and x < 0.1:
        return "very low"
    elif x >= 0.1 and x < 0.3:
        return "low"
    elif x >= 0.3 and x < 0.5:
        return "medium"
    elif x >= 0.5 and x < 0.7:
        return "high"
    else:
        return "very high"
df['Speechiness'] = df['Speechiness'].apply(type_of_Song)
n_songs_per_category = df.groupby('Speechiness').size()
fig=px.pie(df,names=n_songs_per_category.index ,values=n_songs_per_category.values)
fig.update_layout(title='type of songs')
fig.show()
df['Speechiness']=original_speechiness_values
Tempo_original_value=df['Tempo'].copy()
def classify_tempo(bpm):
    if bpm < 90:
        return "Slow"
    elif 90 <= bpm <= 130:
        return "Moderate"
    else:
        return "Fast"
df['Tempo'] = df['Tempo'].apply(classify_tempo)
n_songs_per_category = df.groupby('Tempo').size()

fig = px.pie(names=n_songs_per_category.index,
             values=n_songs_per_category.values,
             title='Types of Tempo')

fig.update_layout(title='Types of Tempo')
fig.show()
df['Tempo']=Tempo_original_value
valence_original_value=df['Valence'].copy()
def valence_type(x):
    if x >= 0.0 and x < 0.5:
        return "Happy|Positive"
    elif x >= 0.5 and x < 1:
        return "Sad|Negative"
df['Valence']=df['Valence'].apply(valence_type)
n_songs_per_category = df.groupby('Valence').size()
fig=px.pie(df,names=n_songs_per_category.index ,values=n_songs_per_category.values)
fig.update_layout(title='types of valence (Happy or sad)')

fig.show()
df['Valence']=valence_original_value
## Insights from EDA Visualizations


# **Valence & Danceability**: Valence and danceability are somewhat directly proportional. Songs with higher danceability tend to have higher popularity.
#
#
# **Energy, Loudness & Popularity**: Energy and loudness are directly proportional to each other. Higher loudness and energy levels correlate with higher popularity.
#
#
# **Acousticness & Energy**: Acousticness and energy show an inverse relationship. Popularity tends to be higher with lower acousticness.
#
#
# **Speechiness**: Most of the dataset has very low speechiness.
#
#
# **Tempo & Speed**: Over 54% of the data indicates songs that are neither too fast nor too slow. Fast songs with high tempo are twice as common as slow songs.
#
#
# **Mood & Popularity**: More than 64% of the dataset comprises sad songs. Interestingly, sad songs seem to have higher popularity.

# **Feature Engineering**
df.head()
df.drop(['Album Release Date', 'minutes_length'], axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)

df=df[['Song', 'Album','Year','Artist Names', 'Artist(s) Genres',
       'Hot100 Ranking Year', 'Hot100 Rank', 'Song Length(ms)', 'Spotify Link',
       'Song Image', 'Spotify URI', 'Acousticness',
       'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness',
       'Speechiness', 'Tempo', 'Valence', 'Key', 'Mode', 'Time Signature','PopularityLevel']]
## Encoding
### **Since the 'artist name' and 'artist genres' columns contain lists of strings, I will split each list into multiple rows, with each element in its own row. I will perform this transformation for both the 'artist name' and 'genre' columns.**
df['Artist Names'] = df['Artist Names'].apply(ast.literal_eval)
df['Artist(s) Genres'] = df['Artist(s) Genres'].apply(ast.literal_eval)


df_exploded = df.explode('Artist Names')
df_exploded = df_exploded.explode('Artist(s) Genres')

df_exploded.info()
# Ordinal Encoding of Popularity Level Feature
df_exploded["PopularityLevel"].replace("popular",2,inplace=True)
df_exploded["PopularityLevel"].replace("Average",1,inplace=True)
df_exploded["PopularityLevel"].replace("Not Popular",0,inplace=True)
df_exploded
# Average Popularity for top 30 Artists
top_30_artists = df_exploded['Artist Names'].value_counts().head(30).index

average_popularity = df_exploded[df_exploded['Artist Names'].isin(top_30_artists)].groupby('Artist Names')['PopularityLevel'].mean()

plt.figure(figsize=(15, 10))
average_popularity.plot(kind='bar')
plt.title('Average Popularity of Songs by Top 30 Artists')
plt.xlabel('Artist Names')
plt.ylabel('Average Popularity')
plt.xticks(rotation=90)
plt.show()
### After splitting the 'artist name' and 'artist genres' columns into individual rows, I applied both target and label encoding to these features. Subsequently, I combined the encoded values back into a single cell for lists with more than one string.

# Target Encoding for Artist Genres
encoder = TargetEncoder(cols=['Artist(s) Genres'])
encoder.fit(df_exploded, df_exploded['PopularityLevel'])
df_encoded = encoder.transform(df_exploded)



with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
# Label Encoding for Artist Genres
le = LabelEncoder()
df_encoded['Artist Names'] = le.fit_transform(df_encoded['Artist Names'])



with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
df_encoded
def aggregate_rows(group):
    sum_artists = sum(group['Artist Names'].unique())
    sum_genres = sum(group['Artist(s) Genres'].unique())

    return pd.Series({
        'Artist Names Encoded': sum_artists,
        'Artist(s) Genres Encoded': sum_genres
    })

aggregated_df = df_encoded.groupby(df_encoded.index).apply(aggregate_rows)

aggregated_df = aggregated_df.reset_index(drop=True)


aggregated_df.info()

aggregated_df

original_indices = set(df.index)
aggregated_indices = set(aggregated_df.index)

missing_indices = original_indices - aggregated_indices
extra_indices = aggregated_indices - original_indices

print("Missing indices:", missing_indices)
print("Extra indices:", extra_indices)
df=pd.concat([aggregated_df,df], axis=1)
df
df["PopularityLevel"].replace("popular",2,inplace=True)
df["PopularityLevel"].replace("Average",1,inplace=True)
df["PopularityLevel"].replace("Not Popular",0,inplace=True)

columns_to_drop = ['Artist Names', 'Artist(s) Genres', 'Song', 'Album', 'Spotify Link', 'Song Image', 'Spotify URI']

df.drop(columns_to_drop, axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
column_names = {
    'Artist Names Encoded': 'Artist Names',
    'Artist(s) Genres Encoded': 'Artist(s) Genres'
}

df = df.rename(columns=column_names)
df.head()
## Creating new features
# The 'Hype' feature is calculated as the sum of 'Loudness' and 'Energy'.
#
# The 'Happiness' feature represents the sum of 'Danceability' and 'Valence'.
df['Hype']=df['Loudness']+df['Energy']
df['Happiness']=df['Danceability']+df['Valence']
# **How songs popularity increases over time**
df_sorted = df.sort_values(by='Year')

bins = range(1900, 2030, 10)
labels = [f"{i}-{i+9}" for i in range(1900, 2020, 10)]

df_sorted['Year Group'] = pd.cut(df_sorted['Year'], bins=bins, labels=labels, right=False)

avg_popularity_by_year = df_sorted.groupby('Year Group')['PopularityLevel'].mean()

plt.figure(figsize=(15, 5))
plt.plot(avg_popularity_by_year.index, avg_popularity_by_year.values, marker='o', linestyle='-')
plt.title('Average Popularity Over 10-Year Intervals (1900-2019)')
plt.xlabel('Year Interval')
plt.ylabel('Average Popularity')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
# Higher popularities tend to be in more recent years. As we look further back in time, the popularities of songs generally decrease.

## Data Splitting
X_feature=df.drop(['PopularityLevel'],axis=1)
Y_feature=df['PopularityLevel']
X_train, X_test, y_train, y_test = train_test_split(X_feature, Y_feature, test_size = 0.20,shuffle=True,random_state=10)
y_train = pd.DataFrame(y_train, columns=['PopularityLevel'])
y_test = pd.DataFrame(y_test, columns=['PopularityLevel'])
y_train
## Transformations ##
fig, axs = plt.subplots(len(df.columns), 2, figsize=(20, 60))

axs = axs.flatten()

for i, column in enumerate(df.columns):

    sns.histplot(df[column], bins=50, ax=axs[2*i])
    axs[2*i].set_title(f'Histogram of {column}')
    axs[2*i].set_xlabel(column)
    axs[2*i].set_ylabel('Frequency')

    sns.kdeplot(df[column], ax=axs[2*i+1], fill=True)
    axs[2*i+1].set_title(f'KDE Plot of {column}')
    axs[2*i+1].set_xlabel(column)
    axs[2*i+1].set_ylabel('Density')
plt.tight_layout()
plt.show()
## To address the skewed distributions, I will apply both a log transformation and a square root transformation to the relevant features.

offset = 1e-10

X_train['Liveness_log'] = np.log(X_train['Liveness'] + offset)

sns.kdeplot(X_train['Liveness_log'], fill=True)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
X_test['Liveness_log'] = np.log(X_test['Liveness'] + offset)

sns.kdeplot(X_test['Liveness_log'], fill=True)
plt.title('Kernel Density Estimation Plot for x_test')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
offset = 1e-10

X_train['Acousticness_sqrt'] = np.sqrt(X_train['Acousticness'] + offset)

sns.kdeplot(X_train['Acousticness_sqrt'], fill=True)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
X_test['Acousticness_sqrt'] = np.sqrt(X_test['Acousticness'] + offset)

sns.kdeplot(X_test['Acousticness_sqrt'], fill=True)
plt.title('Kernel Density Estimation Plot for x_test')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
X_train['Speechiness_sqrt'] = np.sqrt(X_train['Speechiness'])

sns.kdeplot(X_train['Speechiness_sqrt'], fill=True, warn_singular=False)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
X_test['Speechiness_sqrt'] = np.sqrt(X_test['Speechiness'])
sns.kdeplot(X_test['Speechiness_sqrt'], fill=True, warn_singular=False)
plt.title('Kernel Density Estimation Plot for x_test')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
X_train.head()
columns_to_drop = ['Liveness','Acousticness','Speechiness']

X_train.drop(columns_to_drop, axis=1, inplace=True)
X_train.reset_index(drop=True, inplace=True)
X_train.head()
column_names = {
    'Liveness_log': 'Liveness',
    'Acousticness_sqrt': 'Acousticness',
    'Speechiness_sqrt':'Speechiness'
}

X_train = X_train.rename(columns=column_names)
X_train.head()
X_test.head()
columns_to_drop = ['Liveness','Acousticness','Speechiness']

X_test.drop(columns_to_drop, axis=1, inplace=True)
X_test.reset_index(drop=True, inplace=True)
X_test.head()
column_names = {
    'Liveness_log': 'Liveness',
    'Acousticness_sqrt': 'Acousticness',
    'Speechiness_sqrt':'Speechiness'
}

X_test = X_test.rename(columns=column_names)
X_test.head()
## Scaling
### ###################################################################Normalization######################################################## ###
# During the scaling process, the dataset was initially partitioned to avoid data leakage. Min-Max scaling was employed as it useful when data is not Gaussian, while Standard scaling was not utilized. Standard scaling is typically applied to normally distributed data, making Min-Max scaling the preferred choice for this dataset.
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

#y_scaler = MinMaxScaler()

#y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
#y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

#y_train_scaled_df = pd.DataFrame(y_train_scaled, columns=['Scaled Popularity'])
#y_test_scaled_df = pd.DataFrame(y_test_scaled, columns=['Scaled Popularity'])

plt.hist(X_train_scaled_df.iloc[:, 0])
plt.title('Histogram of First Scaled Feature in Training Set')
plt.xlabel('Scaled Values')
plt.ylabel('Frequency')
plt.show()




with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

#with open('y_scaler.pkl', 'wb') as f:
 #   pickle.dump(y_scaler, f)

with open('X_train_scaled_df.pkl', 'wb') as f:
    pickle.dump(X_train, f)
X_train_scaled_df.head()
X_test_scaled_df.head()
X_train=X_train_scaled_df
X_test =X_test_scaled_df
#y_train=y_train_scaled_df
#y_test=y_test_scaled_df
################################################# Feature Selection###############################################################
df[df['Year']>df['Hot100 Ranking Year']]
df[df['Year']>df['Hot100 Ranking Year']].shape[0]
# There are over 1760 rows, accounting for more than 28% of the data, with a 'rank year' of 0 even before the song's release year. This suggests that the 'rank year' column is largely inaccurate.
df['Time Signature'].value_counts()/df.shape[0]
sns.violinplot(X_train['Time Signature'], fill=True)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
df['Instrumentalness'].value_counts()/df.shape[0]
sns.violinplot(X_train['Instrumentalness'], fill=True)
plt.title('Kernel Density Estimation Plot for x_train')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
# Instrumentalness and Time Signature will not be useddue to their negligible variance, indicating that they would not significantly influence the model. Specifically, over 99% of the Time Signature values were '4', and nearly half of the Instrumentalness values were zeros.
# ##  Correlation heatmap
# Since the correlation matrix is most informative with continuous data, I will exclude categorical data when generating the correlation heatmap.

excluded_columns = [ 'Artist Names', 'Artist(s) Genres','Year','Hot100 Rank','Key','Mode','Time Signature']


X_correlation = df.drop(excluded_columns, axis=1)
y = df['PopularityLevel']


numerical_features = X_correlation.select_dtypes(include=['int64', 'float64'])
categorical_features = X_correlation.select_dtypes(exclude=['int64', 'float64'])

numerical_correlations = numerical_features.corrwith(y)

sorted_correlations = numerical_correlations.abs().sort_values(ascending=False)
top_n_features = 18
top_features = sorted_correlations.index[:top_n_features]

top_feature_corr = X_correlation[top_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(top_feature_corr, annot=True, cmap='mako', fmt=".2f")
plt.title(f"Correlation Heatmap of Top {top_n_features} Features with Popularity")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()

##  Filter methods (SelectKBest)
### I will use SelectKBest to select some features and reduce the dimensionality of the model. First, I'll split the features into continuous and categorical. For the categorical features, I will select a subset using Chi_squared, as the input is categorical and the output is categorical. For the continuous features, I will select a subset using ANOVA , as these is appropriate when the input is numerical and output are categorical.

cat_columns = ['Artist Names', 'Artist(s) Genres', 'Year', 'Hot100 Rank', 'Key', 'Mode', 'Time Signature','Hot100 Ranking Year']
X_train_cat = X_train[cat_columns]
X_train_cat
y_train_array = y_train['PopularityLevel'].values

best_cat_features = SelectKBest(score_func=chi2, k=len(cat_columns))

fit = best_cat_features.fit(X_train_cat, y_train_array)

df_scores = pd.DataFrame(fit.scores_, columns=['Score'])
df_columns = pd.DataFrame(X_train_cat.columns, columns=['Feature'])

featureScores = pd.concat([df_columns, df_scores], axis=1)

print(featureScores.nlargest(len(cat_columns), 'Score'))
cont_columns = df.drop(['Artist Names', 'Artist(s) Genres', 'Year', 'Hot100 Rank', 'Key', 'Mode', 'Time Signature','PopularityLevel','Hot100 Ranking Year'], axis=1).columns

X_train_cont = X_train[cont_columns]

best_cont_features = SelectKBest(score_func=f_classif, k='all')

fit = best_cont_features.fit(X_train_cont, y_train)

df_scores = pd.DataFrame(fit.scores_, columns=['Score'])
df_columns = pd.DataFrame(X_train_cont.columns, columns=['Feature'])

featureScores = pd.concat([df_columns, df_scores], axis=1)

print(featureScores.nlargest(33, 'Score'))
selected_features =['Acousticness','Artist(s) Genres','Artist Names','Danceability','Hot100 Ranking Year','Mode','Year','Hot100 Rank','Song Length(ms)','Hype','Loudness','Energy','Instrumentalness']

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
X_train=X_train_selected
X_test =X_test_selected
X_train.head()
X_test.head()
y_train.head()
# Modeling & Hyperparameter tuning
# **Hyperparameter Tuning**
#
# Hyperparameter tuning was conducted using grid search, which systematically combined all possible combinations of hyperparameters provided. This approach aimed to identify the optimal hyperparameter values that yielded the best-performing models in terms of accuracy. The tuning process significantly enhanced the accuracy of the models by optimizing the hyperparameters.
#
# **Cross-Validation**
#
# Cross-validation was employed to assess the generalization capability of the models and mitigate overfitting. This provided a more reliable estimate of a model's performance by averaging the evaluation results across multiple validation sets, thereby offering a more robust evaluation metric compared to a single train-test split.
# # Logistic Regression

# y_train_1d = np.ravel(y_train)
#
# logistic_regression_model = LogisticRegression(max_iter=1000)
#
# cv_scores = cross_val_score(logistic_regression_model, X_train, y_train_1d, cv=5)
# mean_mse = np.mean(cv_scores)
# print("Mean Cross-Validated MSE:", mean_mse)
#
# logistic_regression_model.fit(X_train, y_train_1d)
#
# y_pred = logistic_regression_model.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
#
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="mako")
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.show()
#
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# SVM (SVC)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

svm = SVC()

param_dist = {
    'C': uniform(0.1, 3),  # Decreased range
    'gamma': uniform(0.1, 0.5),  # Decreased range
    'kernel': ['linear', 'rbf'],  # Removed 'poly' and 'sigmoid'
    'degree': [2,3]  # Fixed degree to 2
}

random_search = RandomizedSearchCV(svm, param_dist, n_iter=100, cv=kf, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

best_svm = SVC(**best_params)
best_svm.fit(X_train, y_train)

y_pred = best_svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Decision Tree Classifier
param_grid_dt = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_model = DecisionTreeClassifier()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search_dt = GridSearchCV(estimator=dt_model, param_grid=param_grid_dt, cv=kf, scoring='neg_mean_squared_error')
grid_search_dt.fit(X_train, y_train)

best_params_dt = grid_search_dt.best_params_
best_score_dt = -grid_search_dt.best_score_

dt_model_best = DecisionTreeClassifier(**best_params_dt)
dt_model_best.fit(X_train, y_train)

y_pred_dt = dt_model_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_dt)
mse = mean_squared_error(y_test, y_pred_dt)

print(f"Accuracy: {accuracy}")
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_regressor = RandomForestClassifier(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error', verbose=0)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = -grid_search.best_score_

rf_best = RandomForestClassifier(**best_params, random_state=42)
rf_best.fit(X_train, y_train)

y_pred = rf_best.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))
# KNN
knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_knn = KNeighborsClassifier(**best_params)

cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5)

mean_cv_accuracy = cv_scores.mean()
print("Mean Cross-validation Accuracy:", mean_cv_accuracy)

best_knn.fit(X_train, y_train)

y_pred = best_knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Test Set Accuracy: {accuracy}")
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Gradient Boosting Classifier
# gbm = GradientBoostingClassifier()

# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=kf, scoring='accuracy')
# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_

# best_gbm = GradientBoostingClassifier(**best_params)

# cv_scores = cross_val_score(best_gbm, X_train, y_train, cv=kf)

# mean_cv_accuracy = cv_scores.mean()

# best_gbm.fit(X_train, y_train)

# y_pred = best_gbm.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)

# print(f"Test Set Accuracy: {accuracy}")
# print(f"Mean Squared Error: {mse}")

# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()

# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# XGBoost Classifier

xgb_model = XGBClassifier()

param_grid = {
   'n_estimators': [50, 100, 150],
   'learning_rate': [0.01, 0.1, 0.2],
   'max_depth': [3, 5, 7],
   'min_child_weight': [1, 3, 5],
   'gamma': [0, 0.1, 0.2]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=kf, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

best_xgb_model = XGBClassifier(**best_params)

cv_scores = cross_val_score(best_xgb_model, X_train, y_train, cv=kf)

mean_cv_accuracy = cv_scores.mean()

best_xgb_model.fit(X_train, y_train)

y_pred = best_xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Test Set Accuracy: {accuracy}")
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))
with open('xgb.pkl', 'wb') as f:
    pickle.dump(best_xgb_model, f)
# AdaBoost
# from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import randint

ada_model = AdaBoostClassifier()

param_dist = {
    'n_estimators': randint(50, 150),
    'learning_rate': [0.01, 0.1, 0.2]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(estimator=ada_model, param_distributions=param_dist, n_iter=10, cv=kf, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

best_params = random_search.best_params_

best_ada_model = AdaBoostClassifier(**best_params)

cv_scores = cross_val_score(best_ada_model, X_train, y_train, cv=kf)

mean_cv_accuracy = cv_scores.mean()

best_ada_model.fit(X_train, y_train)

y_pred = best_ada_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Test Set Accuracy: {accuracy}")
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))
with open('Ada_Boost_Classfier.pkl', 'wb') as f:
    pickle.dump(best_ada_model, f)
# Naive Bayes classifier

# nb_model = GaussianNB()

# nb_model.fit(X_train, y_train)

# y_pred = nb_model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)

# print(f"Test Set Accuracy: {accuracy}")
# print(f"Mean Squared Error: {mse}")

# plt.figure(figsize=(8, 6))
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()

# print("Classification Report:")
# print(classification_report(y_test, y_pred))
# Ensemble modeling (Voting Classifier)
ens = VotingClassifier(estimators=[
    ('xgb', best_xgb_model),
    ('ada', best_ada_model),
    ('knn',best_knn),
], voting='hard')

cv_scores = cross_val_score(ens, X_train, y_train, cv=5)

ens.fit(X_train, y_train)

y_pred = ens.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())
print("Accuracy on test set:", accuracy)


with open('Ensemble_Classfier.pkl', 'wb') as f:
    pickle.dump(ens, f)
# **Summary of Findings**
# After comprehensive exploration and analysis of the song dataset, the top-performing model was identified as the **The ensemble model** utilizing a **voting classifier** composed of **XGBoost**, **AdaBoost**, and **KNN**, which achieved a notable accuracy of 0.719.