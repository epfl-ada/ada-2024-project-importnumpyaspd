import networkx as nx
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def get_most_represented_language(Movie, number_language):
    Movie_copy = Movie.copy()  # Créer une copie du dataset Movie pour éviter la modification du original
    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.replace(r'\s*,\s*', ',', regex=True)
    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.split(',')
    Movie_copy = Movie_copy.explode('Movie_languages')

    Movie_wo_nan_language = Movie_copy[Movie_copy["Movie_languages"] != ""]
    language_to_keep = Movie_wo_nan_language["Movie_languages"].value_counts().head(number_language).index

    return language_to_keep

def create_actor_language_dataset(Movie, Actor, number_language):
    Movie_copy = Movie.copy()  # Créer une copie du dataset Movie pour éviter la modification du original
    Actor_copy = Actor.copy()  # Créer une copie du dataset Actor pour éviter la modification du original

    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.replace(r'\s*,\s*', ',', regex=True)
    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.split(',')
    Movie_copy = Movie_copy.explode('Movie_languages')

    Movie_wo_nan_language = Movie_copy[Movie_copy["Movie_languages"] != ""]
    print(f"We lose {(Movie_copy.shape[0] - Movie_wo_nan_language.shape[0]) / Movie_copy.shape[0] * 100:.2f}% of the dataset of movies with this operation.")

    language_to_keep = Movie_wo_nan_language["Movie_languages"].value_counts().head(number_language).index
    Movie_selected_languages = Movie_copy[Movie_copy.Movie_languages.isin(language_to_keep)]

    Actor_movie = Actor_copy["Freebase_movie_ID"]
    Actor_movie.index = Actor_copy.Freebase_actor_ID
    dict_Actor_movie = Actor_movie.to_dict()

    dict_movie_actor = {}
    for key, values in dict_Actor_movie.items():
        for value in values:
            if value not in dict_movie_actor:
                dict_movie_actor[value] = []
            dict_movie_actor[value].append(key)

    Movie_copy = Movie_copy.reset_index()
    Movie_copy["list_actor"] = Movie_copy["Freebase_movie_ID"].map(dict_movie_actor)

    Movie_copy = Movie_copy[~Movie_copy.list_actor.isna()]

    movie_dict = {}
    for language in language_to_keep:
        movie_dict[language] = Movie_copy[Movie_copy.Movie_languages == language]

    actor_per_language = {language: {} for language in language_to_keep}

    for language in language_to_keep:
        df = movie_dict[language]

        for index, row in df.iterrows():
            actor_list = row["list_actor"]
            for actor in actor_list:
                if actor in actor_per_language[language]:
                    actor_per_language[language][actor] += 1
                else:
                    actor_per_language[language][actor] = 1

    actor_lists = []
    for language in language_to_keep:
        actor_df = pd.DataFrame.from_dict(actor_per_language[language], orient='index', columns=[language])
        actor_lists.append(actor_df)

    total = pd.concat(actor_lists, axis=1, join='outer').fillna(0)
    total.columns = language_to_keep

    return total

def create_cross_language(df):
    df_copy = df.copy()  # Créer une copie du dataframe
    result = {}
    n_language = df_copy.shape[1]

    for index in range(n_language):
        newdf = df_copy.copy()

        filtered_column = df_copy.iloc[:, index][df_copy.iloc[:, index] != 0]
        newdf = newdf.loc[filtered_column.index, :]
        newdf[newdf>0] = 1
        newdf = newdf.T
        newdf["sum"] = newdf.sum(axis=1)

        result[df_copy.columns[index]] = newdf

    return result
    
def create_cross_language_count(df):
    df_copy = df.copy()  # Créer une copie du dataframe
    result = {}

    n_language = df_copy.shape[1]
    for index in range(n_language):
        newdf = df_copy.copy()

        filtered_column = df_copy.iloc[:, index][df_copy.iloc[:, index] != 0]
        newdf = newdf.loc[filtered_column.index, :]
        newdf["nb_movie"] = newdf.sum(axis=1)

        result[df_copy.columns[index]] = newdf

    return result

def plot_language_histograms(result):
    for key, value in result.items():
        data = value
        
        plt.figure(figsize=(10, 6))
        plt.bar(data.index, data['sum'])
        plt.title(f"Histogramme des films pour {key}")
        plt.xlabel("Langues")
        plt.ylabel("Nombre de films")
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.show()

def custom_create_actor_network(Actors, Movie, min_movies=50, min_releasedate=0, add_attributes=False):
    Actors_copy = Actors.copy()  # Créer une copie du dataset Actors
    Movie_copy = Movie.copy()  # Créer une copie du dataset Movie

    actors_with_min_x_movies = Actors_copy[Actors_copy['actor_age_atmovierelease'].apply(len) >= min_movies]
    actors_df = actors_with_min_x_movies.explode('Freebase_movie_ID')
    movie_releasedates = Movie_copy.set_index('Freebase_movie_ID')['release_date'].to_dict()

    G = nx.Graph()

    for movie_id, group in tqdm(actors_df.groupby('Freebase_movie_ID'), desc="Creating network"):
        if movie_releasedates.get(movie_id, 3000) <= min_releasedate:
            actor_ids = group['Freebase_actor_ID'].tolist()
            
            for actor1, actor2 in combinations(actor_ids, 2):
                if actor1 != actor2:
                    if G.has_edge(actor1, actor2):
                        G[actor1][actor2]['weight'] += 1
                    else:
                        movie_language = Movie_copy.loc[Movie_copy['Freebase_movie_ID'] == movie_id, 'Movie_languages']
                        if not movie_language.empty:
                            movie_language = movie_language.iloc[0]
                            # Si movie_language est une liste, on peut simplement utiliser le premier élément
                            if isinstance(movie_language, list):
                                movie_language = movie_language[0].strip()  # On prend la première langue
                            else:
                                # Si ce n'est pas une liste, appliquer split
                                movie_language = movie_language.split(',')[0].strip()
                        else:
                            movie_language = "Unknown"
                        if movie_language == "":
                            movie_language = "Unknown"
                        G.add_edge(actor1, actor2, weight=1, langue=movie_language)

    if add_attributes:
        for _, row in tqdm(actors_df.iterrows(), desc="Adding attributes"):
            actor_id = row['Freebase_actor_ID']
            if actor_id in G:
                G.nodes[actor_id].update({
                    'name': row['actor_name'],
                    'gender': row.get('actor_gender', None),
                    'ethnicity': row.get('ethnicity', None),
                    'height': row.get('actor_height', None)
                })
    
    return G
