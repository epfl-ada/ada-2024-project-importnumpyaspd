import networkx as nx
from scipy.stats import chi2_contingency
from itertools import combinations
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

def performChi2(df):
    """
    perform Chi2 on matrix df (it should have already the right format)
    """
    chi2, p_value, dof, expected = chi2_contingency(df)

    return chi2, p_value, dof, expected

def explode_movie_dataset(Movie):
    """
    explodes the 'Movie_languages' column of the dataset into multiple rows 
    removing nan values.
    """
    Movie_copy = Movie.copy()
    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.replace(r'\s*,\s*', ',', regex=True)
    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.split(',')# split the 'Movie_languages' into a list
    Movie_copy = Movie_copy.explode('Movie_languages') #explode dataset

    Movie_wo_nan_language = Movie_copy[Movie_copy["Movie_languages"] != ""]  # Remove rows with empty language

    return Movie_wo_nan_language

def get_most_represented_language(Movie, number_language):
    """
    get the "number_language" most represented languages in the dataset
    """
    Movie_copy = Movie.copy()
    Movie_wo_nan_language = explode_movie_dataset(Movie_copy) #explode dataset

    #get most frequent languages
    language_to_keep = Movie_wo_nan_language["Movie_languages"].value_counts().head(number_language).index

    return language_to_keep

def create_actor_language_dataset(Movie, Actor, number_language):
    """
    creates a dataset of actor with languages movie they played in.
    """  
    Movie_copy = Movie.copy()
    Actor_copy = Actor.copy()

    # explode
    Movie_copy = explode_movie_dataset(Movie_copy)

    # keep movie with most represetnative languages
    language_to_keep = get_most_represented_language(Movie, number_language)
    Movie_selected_languages = Movie_copy[Movie_copy.Movie_languages.isin(language_to_keep)]

    # dict to map actors to movies
    Actor_movie = Actor_copy["Freebase_movie_ID"]
    Actor_movie.index = Actor_copy.Freebase_actor_ID
    dict_Actor_movie = Actor_movie.to_dict()

    dict_movie_actor = {}
    for key, values in dict_Actor_movie.items():
        for value in values:
            if value not in dict_movie_actor:
                dict_movie_actor[value] = []
            dict_movie_actor[value].append(key)

    # do the mapping
    Movie_copy = Movie_copy.reset_index()
    Movie_copy["list_actor"] = Movie_copy["Freebase_movie_ID"].map(dict_movie_actor)

    #drop movies twithout associated actors
    Movie_copy = Movie_copy[~Movie_copy.list_actor.isna()]

    movie_dict = {}
    for language in language_to_keep:
        movie_dict[language] = Movie_copy[Movie_copy.Movie_languages == language]

    actor_per_language = {language: {} for language in language_to_keep}

    # Counter appearandce for actor in each language's movies
    for language in language_to_keep:
        df = movie_dict[language]
        for index, row in df.iterrows():
            actor_list = row["list_actor"]
            for actor in actor_list:
                if actor in actor_per_language[language]:
                    actor_per_language[language][actor] += 1
                else:
                    actor_per_language[language][actor] = 1

    # Create a DataFrame for each language's actor counts and combine them into a single DataFrame
    actor_lists = []
    for language in language_to_keep:
        actor_df = pd.DataFrame.from_dict(actor_per_language[language], orient='index', columns=[language])
        actor_lists.append(actor_df)

    # Concatenate actor counts for each language
    total = pd.concat(actor_lists, axis=1, join='outer').fillna(0)
    total.columns = language_to_keep

    return total
    
def create_cross_language(df):
    """
     create a dict per languages kept. each value are df only with actor that have played in this specific language (key)
    """
    
    df_copy = df.copy()
    result = {}  # Init result dict
    n_language = df_copy.shape[1]  #get nb languages

    for index in range(n_language):
        newdf = df_copy.copy()
        
        # filter column to have only when there is participation for the given language(indeex)
        filtered_column = df_copy.iloc[:, index][df_copy.iloc[:, index] != 0]
        
        # Keep only the rows that are in the filtered column
        newdf = newdf.loc[filtered_column.index, :]
        
        # Set all positive values to 1, indicating participation in that language
        newdf[newdf > 0] = 1
        
        newdf = newdf.T
        
        # Create a 'sum' column that counts the total participation across all languages for each actor
        newdf["sum"] = newdf.sum(axis=1)
        
        # add to dict result
        result[df_copy.columns[index]] = newdf

    return result
    
def create_cross_language_count(df):
    """
    create dict containing a dict with a key per movie language.
    for each key : there is the actor that have played in this language with all other movie languages played in (with the count of movie per languages)
    """
    
    df_copy = df.copy()
    result = {} #init result dict

    n_language = df_copy.shape[1] #get nb coloumnm
    
    for index in range(n_language):
        newdf = df_copy.copy()
        
        # Filter column with participation
        filtered_column = df_copy.iloc[:, index][df_copy.iloc[:, index] != 0]
        
        # Keep only rows for actors that are present in the filtered column
        newdf = newdf.loc[filtered_column.index, :]
        
        # Add a count of nb movie played in for each actor
        newdf["nb_movie"] = newdf.sum(axis=1)
        
        # add to dict
        result[df_copy.columns[index]] = newdf

    return result

def custom_create_actor_network(Actors, Movie, min_movies=50):
    """
    create network with connection "play together" between actor
    identify the language of movie in the edge of the network
    """
    start_time = time.time()
    
    Actors_copy = Actors.copy()
    Movie_copy = Movie.copy()
    
    # Filter actors who have acted in at least 'min_movies' movies
    actors_with_min_x_movies = Actors_copy[Actors_copy['actor_age_atmovierelease'].apply(len) >= min_movies]
    
    # Explode the list of movies to create a DataFrame where each row corresponds to a unique actor-movie pair
    actors_df = actors_with_min_x_movies.explode('Freebase_movie_ID')
    
    # Create a dictionary of movie release dates using movie IDs as keys
    movie_releasedates = Movie_copy.set_index('Freebase_movie_ID')['release_date'].to_dict()

    # use networkx to construct the netwrok
    G = nx.Graph()

    # for each movie : get list of actor id 
    for movie_id, group in tqdm(actors_df.groupby('Freebase_movie_ID'), desc="Creating network"):
        actor_ids = group['Freebase_actor_ID'].tolist()
        
        # iterate over each pair of actors who play in the same movie and add a connecttion
        # it's also looking at the movie language in order to later apply the same color for the plot 
        for actor1, actor2 in combinations(actor_ids, 2):
            if actor1 != actor2:
                if G.has_edge(actor1, actor2):
                    # If the edge exists, att 2 to weight
                    G[actor1][actor2]['weight'] += 1
                else:
                    # get the language of the movie
                    movie_language = Movie_copy.loc[Movie_copy['Freebase_movie_ID'] == movie_id, 'Movie_languages']
                    if not movie_language.empty:
                        movie_language = movie_language.iloc[0]
                        if isinstance(movie_language, list):
                            movie_language = movie_language[0].strip() # take only the first language if it's a list
                        else:
                            # If not a list, split by comma and take the first language (reason : can't represent both color, there is less than 10% woith many language, it won't really change the gloabl aspect pf the plot)
                            movie_language = movie_language.split(',')[0].strip()
                    else:
                        movie_language = "Unknown"
                    if movie_language == "":
                        movie_language = "Unknown"
                    
                    # add a new edge between two actors with weight 1 and the language of the movie !
                    G.add_edge(actor1, actor2, weight=1, langue=movie_language)
    
    end_time = time.time()
    print(f"time to compute: {end_time - start_time:.1f} seconds")
    
    return G

def create_distribution_for_each_group(dict_group, number_group):
    """
    create distribution of movie languages for different group. 
    """

    dict_total = {}

    # Iterate through each group
    for g in range(0, number_group):
        group = dict_group[g]
    
        # Iterate on languages
        for i in group.columns:
            if i != "nb_movie":
                ratio = float(group[i].sum()) #compute ratio to show only %
                if i in dict_total:
                    dict_total[i].append(ratio)
                else:
                    dict_total[i] = [ratio]

    return dict_total

def create_group_success(actor_count_movie_language):
    """
    create specific group describe in notebook. (based on first part : label of career profile)
    """
    dict_group = {}
    
    group1 = actor_count_movie_language[actor_count_movie_language["Labels"]==0]
    dict_group[1] = group1
    group2 = actor_count_movie_language[actor_count_movie_language["Labels"]==1]
    dict_group[2] = group2
    group3 = actor_count_movie_language[actor_count_movie_language["Labels"]==2]
    dict_group[3] = group3

    #check that the nb of actor per group is not that small
    print(f"Size of groups : {group1.shape[0],group2.shape[0],group3.shape[0]}")

    return dict_group

def print_result_chi2(chi2, p_value, dof, expected):
    # show result of chi2 test
    print("Chi-2 :", chi2)
    print("P-value :", p_value)
    print("degree of freedom :", dof)

    if p_value < 0.05:
        print("Distribution are significantly statistically different (p < 0.05).")
    else:
        print("No significance statistical differences (p >= 0.05).")
