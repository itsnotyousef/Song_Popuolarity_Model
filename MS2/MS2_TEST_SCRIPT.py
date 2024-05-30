import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
from sklearn.metrics import mean_squared_error , r2_score,accuracy_score,mean_absolute_error
import ast
import pickle
import json
from scipy.stats import shapiro
pd.set_option('display.max_colwidth', None)



# **Test Script**
test_df=pd.read_csv(r"C:\Users\yousef\Downloads\SongPopularity_Milestone2.csv")
pd.set_option('display.max_columns', None)
test_df.head()
orig_df=pd.read_csv(r"C:\Users\yousef\Downloads\SongPopularity_Milestone2.csv")
mean_of_SongLength = orig_df['Song Length(ms)'].mean()
median_of_SongLength = orig_df['Song Length(ms)'].median()

mean_of_Acousticness=orig_df['Acousticness'].mean()
median_of_Acousticness=orig_df['Acousticness'].median()

mean_of_Danceability=orig_df['Danceability'].mean()
median_of_Danceability=orig_df['Danceability'].median()

mean_of_Energy=orig_df['Energy'].mean()
median_of_Energy=orig_df['Energy'].median()

mean_of_Instrumentalness=orig_df['Instrumentalness'].mean()
median_of_Instrumentalness=orig_df['Instrumentalness'].median()

mean_of_Liveness=orig_df['Liveness'].mean()
median_of_Liveness=orig_df['Liveness'].median()

mean_of_Loudness=orig_df['Loudness'].mean()
median_of_Loudness=orig_df['Loudness'].median()

mean_of_Speechiness=orig_df['Speechiness'].mean()
median_of_Speechiness=orig_df['Speechiness'].median()

mean_of_Tempo=orig_df['Tempo'].mean()
median_of_Tempo=orig_df['Tempo'].median()

mean_of_Valence=orig_df['Valence'].mean()
median_of_Valence=orig_df['Valence'].median()
Song_mod = orig_df['Song'].mode()[0]
Album_mod = orig_df['Album'].mode()[0]
Album_Release_Date_mod = orig_df['Album Release Date'].mode()[0]
Artist_Names_mod = orig_df['Artist Names'].mode()[0]
Artist_Genres_mod = orig_df['Artist(s) Genres'].mode()[0]
Hot100_Ranking_Year_mod = orig_df['Hot100 Ranking Year'].mode()[0]
Hot100_Rank_mod = orig_df['Hot100 Rank'].mode()[0]
Spotify_Link_mod = orig_df['Spotify Link'].mode()[0]
Song_Image_mod = orig_df['Song Image'].mode()[0]
Spotify_URI_mod = orig_df['Spotify URI'].mode()[0]
Key_mod = orig_df['Key'].mode()[0]
Mode_mod = orig_df['Mode'].mode()[0]
Time_Signature_mod = orig_df['Time Signature'].mode()[0]
categorical_modes = {
    'Song': orig_df['Song'].mode()[0],
    'Album': orig_df['Album'].mode()[0],
    'Album Release Date': orig_df['Album Release Date'].mode()[0],
    'Artist Names': orig_df['Artist Names'].mode()[0],
    'Artist(s) Genres': orig_df['Artist(s) Genres'].mode()[0],
    'Hot100 Ranking Year': orig_df['Hot100 Ranking Year'].mode()[0],
    'Hot100 Rank': orig_df['Hot100 Rank'].mode()[0],
    'Spotify Link': orig_df['Spotify Link'].mode()[0],
    'Song Image': orig_df['Song Image'].mode()[0],
    'Spotify URI': orig_df['Spotify URI'].mode()[0],
    'Key': orig_df['Key'].mode()[0],
    'Mode': orig_df['Mode'].mode()[0],
    'Time Signature': orig_df['Time Signature'].mode()[0]
}


def convert_to_builtin_type(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj)))

with open('categorical_modes.json', 'w') as f:
    json.dump(categorical_modes, f, default=convert_to_builtin_type)

with open('categorical_modes.json', 'r') as f:
    categorical_modes = json.load(f)
def fillna_by_mod(feature,mode_value):
    test_df[feature].fillna(mode_value,inplace=True)
fillna_by_mod("Song",Song_mod)
fillna_by_mod("Album",Album_mod)
fillna_by_mod("Album Release Date",Album_Release_Date_mod)
fillna_by_mod("Artist Names",Artist_Names_mod)
fillna_by_mod("Artist(s) Genres",Artist_Genres_mod)
fillna_by_mod("Hot100 Ranking Year",Hot100_Ranking_Year_mod)
fillna_by_mod("Hot100 Rank",Hot100_Rank_mod)
fillna_by_mod("Spotify Link",Spotify_Link_mod)
fillna_by_mod("Song Image",Song_Image_mod)
fillna_by_mod("Spotify URI",Spotify_URI_mod)
fillna_by_mod("Key",Key_mod)
fillna_by_mod("Mode",Mode_mod)
fillna_by_mod("Time Signature",Time_Signature_mod)
mean_median_values = {
    'Song Length(ms)': {'mean': mean_of_SongLength, 'median': median_of_SongLength},
    'Acousticness': {'mean': mean_of_Acousticness, 'median': median_of_Acousticness},
    'Danceability': {'mean': mean_of_Danceability, 'median': median_of_Danceability},
    'Energy': {'mean': mean_of_Energy, 'median': median_of_Energy},
    'Instrumentalness': {'mean': mean_of_Instrumentalness, 'median': median_of_Instrumentalness},
    'Liveness': {'mean': mean_of_Liveness, 'median': median_of_Liveness},
    'Loudness': {'mean': mean_of_Loudness, 'median': median_of_Loudness},
    'Speechiness': {'mean': mean_of_Speechiness, 'median': median_of_Speechiness},
    'Tempo': {'mean': mean_of_Tempo, 'median': median_of_Tempo},
    'Valence': {'mean': mean_of_Valence, 'median': median_of_Valence}
}


with open('mean_median_values.json', 'w') as json_file:
    json.dump(mean_median_values, json_file)
with open('mean_median_values.json', 'r') as json_file:
    mean_median_values = json.load(json_file)
def fillna_numberical(column,mean_value,median_value):
    stat, p_value = shapiro(test_df[column].dropna())
    print("Column '{}':".format(column))
    print("   Shapiro-Wilk Test Statistic:", stat)
    print("   p-value:", p_value)

    if p_value < 0.05:
        test_df[column].fillna(median_value, inplace=True)
        print("   Nulls replaced by median:", median_value)
    else:
        print("   Distribution is normal.")
        mean_value = test_df[column].mean()
        test_df[column].fillna(mean_value, inplace=True)
        print("   Nulls replaced by mean:", mean_value)


def Hidden_Nulls(column, mod_value):
    pattern = r'^[^\w\s]+$'
    hidden_nulls = test_df[test_df[column].str.match(pattern, na=False)]
    test_df[column] = test_df[column].apply(lambda x: np.nan if x in hidden_nulls.values else x)

    test_df[column].fillna(mod_value, inplace=True)

Hidden_Nulls("Song",Song_mod)
Hidden_Nulls("Album",Album_mod)
Hidden_Nulls("Album Release Date",Album_Release_Date_mod)
Hidden_Nulls("Artist Names",Artist_Names_mod)
Hidden_Nulls("Artist(s) Genres",Artist_Genres_mod)
Hidden_Nulls("Spotify Link",Spotify_Link_mod)
Hidden_Nulls("Song Image",Song_Image_mod)
Hidden_Nulls("Spotify URI",Spotify_URI_mod)
test_df['Year'] = test_df['Album Release Date'].apply(lambda x: x.split('/')[-1].split('-')[0])
test_df.drop(['Album Release Date'],axis=1,inplace=True)
test_df.head(3)
test_df['Artist Names'] = test_df['Artist Names'].apply(ast.literal_eval)
test_df['Artist(s) Genres'] = test_df['Artist(s) Genres'].apply(ast.literal_eval)


df_exploded = test_df.explode('Artist Names')
df_exploded = df_exploded.explode('Artist(s) Genres')


test_df=df_exploded.copy()
test_df
# test_df["PopularityLevel"].replace("popular",2,inplace=True)
test_df["PopularityLevel"].replace("Popular",2,inplace=True)
test_df["PopularityLevel"].replace("Average",1,inplace=True)
test_df["PopularityLevel"].replace("Not Popular",0,inplace=True)
test_df.head()
with open('target_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

test_data_encoded = encoder.transform(test_df)
test_data_encoded
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
train_unique_labels = set(le.classes_)
test_data_encoded['Artist Names'] = test_data_encoded['Artist Names'].apply(lambda x: le.transform([x])[0] if x in train_unique_labels else -1)

test_df = test_data_encoded.copy()
test_df
def aggregate_rows(group):
    sum_artists = sum(group['Artist Names'].unique())
    sum_genres = sum(group['Artist(s) Genres'].unique())

    return pd.Series({
        'Artist Names Encoded': sum_artists,
        'Artist(s) Genres Encoded': sum_genres
    })

aggregated_df = test_df.groupby(test_df.index).apply(aggregate_rows)

aggregated_df = aggregated_df.reset_index(drop=True)


aggregated_df.info()

original_indices = set(test_df.index)
aggregated_indices = set(aggregated_df.index)

missing_indices = original_indices - aggregated_indices
extra_indices = aggregated_indices - original_indices

print("Missing indices:", missing_indices)
print("Extra indices:", extra_indices)
test_df=pd.concat([aggregated_df,test_df], axis=1)
test_df
columns_to_drop = ['Artist Names', 'Artist(s) Genres', 'Song', 'Album', 'Spotify Link', 'Song Image', 'Spotify URI']

test_df.drop(columns_to_drop, axis=1, inplace=True)
test_df.reset_index(drop=True, inplace=True)
column_names = {
    'Artist Names Encoded': 'Artist Names',
    'Artist(s) Genres Encoded': 'Artist(s) Genres'
}

test_df = test_df.rename(columns=column_names)
test_df.head(3)
test_df['Hype']=test_df['Loudness']+test_df['Energy']
test_df['Happiness']=test_df['Danceability']+test_df['Valence']
offset = 1e-10

test_df['Liveness_log'] = np.log(test_df['Liveness'] + offset)
test_df['Acousticness_sqrt'] = np.sqrt(test_df['Acousticness'] + offset)
test_df['Speechiness_sqrt'] = np.sqrt(test_df['Speechiness'])
columns_to_drop = ['Liveness','Acousticness','Speechiness']

test_df.drop(columns_to_drop, axis=1, inplace=True)
test_df.reset_index(drop=True, inplace=True)
test_df.head()
column_names = {
    'Liveness_log': 'Liveness',
    'Acousticness_sqrt': 'Acousticness',
    'Speechiness_sqrt':'Speechiness'
}

test_df = test_df.rename(columns=column_names)
test_df.head()
test_features=test_df.drop(['PopularityLevel'],axis=1)
test_target=test_df['PopularityLevel']
test_features.columns
test_features_df=test_features[['Artist Names', 'Artist(s) Genres', 'Year', 'Hot100 Ranking Year',
       'Hot100 Rank', 'Song Length(ms)', 'Danceability', 'Energy',
       'Instrumentalness', 'Loudness', 'Tempo', 'Valence', 'Key', 'Mode',
       'Time Signature', 'Hype', 'Happiness', 'Liveness',
       'Acousticness', 'Speechiness']]
test_features_df
test_features_df[test_features_df['Artist Names']==-1]
test_target_df = pd.DataFrame({'Target': test_target})
test_target_df
test_features_df
with open('X_train_scaled_df.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

#with open('y_scaler.pkl', 'rb') as f:
 #   y_scaler = pickle.load(f)

test_features_scaled = scaler.transform(test_features_df)
#test_target_scaled = y_scaler.transform(test_target_df.values.resh

test_features_scaled_df = pd.DataFrame(test_features_scaled, columns=test_features_df.columns)

test_features_scaled_df
test_target_df
selected_features =['Acousticness','Artist(s) Genres','Artist Names','Danceability','Hot100 Ranking Year','Mode','Year','Hot100 Rank','Song Length(ms)','Hype','Loudness','Energy','Instrumentalness']
test_features_scaled_df_selected = test_features_scaled_df[selected_features]
test_features=test_features_scaled_df_selected.copy()
test_target=test_target_df.copy()
test_features.head()
test_target
test_features
with open('Ensemble_Classfier.pkl', 'rb') as f:
    loaded_ens = pickle.load(f)


y_pred = loaded_ens.predict(test_features)

accuracy = accuracy_score(test_target, y_pred)
mse = mean_squared_error(test_target, y_pred)

print(f"Test Set Accuracy: {accuracy}")
print(f"Mean Squared Error: {mse}")