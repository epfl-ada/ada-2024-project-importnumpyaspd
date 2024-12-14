import pickle 
import numpy as np 
import pandas as pd
import networkx as nx
from tqdm import tqdm
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import community
import collections
import scipy.stats

# j'ai enlevé les bails à propos des realease date car je n'en avais pas besoin
def create_actor_network_2(Actors, Movie, min_movies=50):
    """
    Create the actor network for actors that played in at least `min_movies` movies. 
    Each node is an actor with name, gender, ethnicity, and height as attributes.
    The weight of the edges corresponds to the number of times they played together.

    Parameters:
    Actors (pd.DataFrame): A dataframe containing all the information of the actors.
    Movie (pd.DataFrame): A dataframe containing all the information of the movies.
    min_movies (int): A threshold to use the actors that played in at least `min_movies`.
    min_releasedate (int): The minimum release date to consider for filtering movies.

    Returns:
    NetworkX Graph: A graph representing the connections between actors.
    """
    
    # Filter actors who have played in at least `min_movies` films
    actors_with_min_x_movies = Actors[Actors['actor_age_atmovierelease'].apply(len) >= min_movies]
    
    # Explode the actor's movie list into separate rows and filter by movie release date
    actors_df = actors_with_min_x_movies.explode('Freebase_movie_ID')

    G = nx.Graph()

    for movie_id, group in tqdm(actors_df.groupby('Freebase_movie_ID'), desc="Creating network"):
        actor_ids = group['Freebase_actor_ID'].tolist()
        for actor1, actor2 in combinations(actor_ids, 2):
            if actor1 != actor2:
                if G.has_edge(actor1, actor2):
                    G[actor1][actor2]['weight'] += 1
                else:
                    G.add_edge(actor1, actor2, weight=1)

    # Add attributes to each actor node
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


def Actor_age_start(Actor):
    """
    Adds a column to the actor profile containing the starting age of their carreer
    """
    Actor["actor_age_atmovierelease"] = Actor["actor_age_atmovierelease"].apply(
    lambda x: [-1 if pd.isna(age) or age < 0 else int(age) for age in x])
    
    Actor["Career_Start_age"] = Actor.apply(
    lambda row: min((val for val in row["actor_age_atmovierelease"] if val > 0), default=np.nan),axis=1,)

    return Actor 


def Director_age_start(Director):
    """
    Adds a column to the director profile containing the starting age of their carreer
    """
    Director["age_at_movie_release"] = Director["age_at_movie_release"].apply(
    lambda x: [-1 if pd.isna(age) or age < 0 else int(age) for age in x])
    
    Director["Career_Start_age"] = Director.apply(
    lambda row: min((val for val in row["age_at_movie_release"] if val > 0), default=np.nan),axis=1,)

    return Director 


def Actor_ratings(Actor,Movie):
    """
    Create a list containing the ratings of each movie per actor
    """
    df_explode = Actor.explode('Freebase_movie_ID')
    df_explode = df_explode.merge(Movie[['Freebase_movie_ID','Average rating']], on='Freebase_movie_ID', how='left')
    ratings = df_explode.groupby('Freebase_actor_ID')['Average rating'].apply(list)
    Actor = Actor.merge(ratings.rename('Ratings'), on='Freebase_actor_ID', how='left')

    return Actor 


def Director_ratings(Director,Movie):
    """
    Create a list containing the ratings of each movie per director
    """
    df_explode = Director.explode('Freebase_movie_ID')
    df_explode = df_explode.merge(Movie[['Freebase_movie_ID','Average rating']], on='Freebase_movie_ID', how='left')
    ratings = df_explode.groupby('IMDb_director_ID')['Average rating'].apply(list)
    Director = Director.merge(ratings.rename('Ratings'), on='IMDb_director_ID', how='left')

    return Director 


def Actor_mean_rating(Actor):
    """
    Compute the mean rating of the movie in which the actor played
    """
    Actor['Mean_Rating'] = Actor['Ratings'].apply(
    lambda ratings: sum(ratings) / len(ratings) if ratings and all(pd.notna(ratings)) else None)

    return Actor

    
def Director_mean_rating(Director):
    """
    Compute the mean rating of the movie in which the actor directed
    """ 
    Director['Mean_Rating'] = Director['Ratings'].apply(
    lambda ratings: sum(ratings) / len(ratings) if ratings and all(pd.notna(ratings)) else None)

    return Director


def Number_per_genre(Movie, Min_movie=1000):
    """
    Compute the number of movie per genre with a minimal value per movie to be taken into account
    """ 
    # Splitting the genre
    Movie_bis = Movie.copy()
    Movie_bis['Movie_genres'] = Movie_bis['Movie_genres'].str.split(', ')

    # Counting the number of movies per genre
    all_genres = Movie_bis['Movie_genres']
    list_all_genres = [genre for sublist in all_genres for genre in sublist]
  
    genre_counts = pd.DataFrame(list_all_genres, columns=['Movie_genres']).value_counts().reset_index()
    genre_counts = genre_counts[genre_counts["Movie_genres"]!='']
    genre_counts.columns = ['Movie_genres', 'Count']

    # Computing the mean per genre
    exploded_movies = Movie_bis.explode('Movie_genres')
    Mean_per_genre = exploded_movies.groupby('Movie_genres')['Average rating'].mean().reset_index()
    Mean_per_genre = Mean_per_genre.iloc[1:]

    # Merging both dataframe
    genre_counts = genre_counts.merge(Mean_per_genre[['Movie_genres','Average rating']], on='Movie_genres', how='left')

    genre_counts['Average rating'] = genre_counts['Average rating'].astype(float)
    
    # Keeping only the genre that are represented more than 1000 times
    genre_counts = genre_counts[genre_counts.Count>=Min_movie]

    return genre_counts


def Mean_actor_per_movie(Actor):
    """
    Compute the mean number of actors per movie
    """
    Actor_bis = Actor[['Freebase_movie_ID', 'Freebase_actor_ID']]
    exploded_movies_ID = Actor_bis.explode('Freebase_movie_ID')
    mean_actor_movie = exploded_movies_ID.groupby('Freebase_movie_ID').size().mean()
    
    return mean_actor_movie


def Mean_movie_per_actor(Actor):
    """
    Compute the mean number of actors per movie
    """
    Actor_bis = Actor.copy()
    Actor_bis["Number_of_films"] = Actor_bis["actor_age_atmovierelease"].apply(len)
    mean_movie_actor = Actor_bis["Number_of_films"].mean()
    
    return mean_movie_actor


def Splitting_Louvain(Graph,Actor):
    Graph_bis = Graph.copy()
    partition = community.best_partition(Graph_bis)
    values = [partition.get(node) for node in Graph_bis.nodes()]
    counter=collections.Counter(values)
        
    partition = pd.Series(partition)
    partition.name = "cluster_id"

    # Merge the partition with the Actor dataset
    actors_clustered = Actor.copy().reset_index(drop=True)
    actors_clustered = pd.merge(actors_clustered, partition, right_index=True, left_on="Freebase_actor_ID")
    
    # Find the three largest clusters
    largest_clusters = pd.DataFrame(partition.value_counts()).reset_index()

    return actors_clustered, largest_clusters


def Centrality_Louvain(Actor_cluster,Cluster,Movie):
    
    Max_Cluster = Cluster.shape[0]
    Actors_Groups = {}
    Networks = {}

    for cluster in Cluster['cluster_id']:
        Actors_Groups[f'cluster_{cluster}'] = Actor_cluster[Actor_cluster['cluster_id'] == cluster].reset_index(drop=True)

    for cluster, actor in Actors_Groups.items():
        played_together = create_actor_network_2(actor, Movie, 5)
        nan_nodes = [node for node in played_together.nodes if pd.isna(node)]
        played_together.remove_nodes_from(nan_nodes)
        Networks[cluster] = played_together

    # Compute eigenvector centrality and update actor DataFrames
    for cluster, played_together in Networks.items():
        eigenvector_centrality = nx.eigenvector_centrality(played_together)
        Actors_Groups[cluster]["eigenvector_centrality_final"] = (Actors_Groups[cluster]["Freebase_actor_ID"].map(eigenvector_centrality))

    return Actors_Groups, Networks


def Top_centrality_Louvain(Actor_groups):
    Actor_groups_top = {}

    for cluster, actor in Actor_groups.items():
        mean_centrality = actor['eigenvector_centrality_final'].mean()
        top_actor = actor[actor['eigenvector_centrality_final'] >= mean_centrality].reset_index(drop=True)
        Actor_groups_top[cluster] = top_actor

    return Actor_groups_top


def keep_above_mean_rating(Actor,Actor_genre,Genre_counts):
    # Defining empty list for the results 
    keep = []
    keep_not = []
    Actor_genre_bis = Actor_genre.copy().reset_index()
    Actor_genre_bis = Actor_genre_bis.rename(columns = {'index':'Freebase_actor_ID'})

    # Checking for each actor if they are better than the average rating in their genre
    for _,Act in tqdm(Actor_genre_bis.iterrows(), desc="Checking their relative ranking"):
        Actor_id = Act['Freebase_actor_ID']
        Top_1 = Act['Top_1_Genre']
        Rating = Act['Top_1_Rating']
        for _,Genre in Genre_counts.iterrows():
            if Top_1 == Genre['Movie_genres'] and Rating >= Genre['Average rating']:
                keep.append(Actor_id)
                continue
            if Top_1 == Genre['Movie_genres'] and Rating < Genre['Average rating']:    
                keep_not.append(Actor_id)
                continue

    # Keeping only the ones above the mean
    Actor_new = Actor[Actor.index.isin(keep)]
    Actor_genre_new = Actor_genre[Actor_genre.index.isin(keep)]

    return Actor_new, Actor_genre_new, keep


def Actor_genre_count_ratings(Actor, Movie, min_movies=5):
    """
    Compute the number of movies per genre and the average ratings of actors per genre.
    Actors are considered only if they have played in at least `min_movies` movies.
    """
    # Making a copy of the dataframes
    Actor_bis = Actor.copy()
    Movie_bis = Movie.copy()

    # Splitting the genre column into lists
    Movie_bis['Movie_genres'] = Movie_bis['Movie_genres'].str.split(', ')
    
    # Filter actors who have played in at least `min_movies` films
    actors_with_min_x_movies = Actor_bis[Actor_bis['actor_age_atmovierelease'].apply(len) >= min_movies]
    
    # Explode the actor's movie list into separate rows and filter by movie genre
    df_explode = actors_with_min_x_movies.explode('Freebase_movie_ID')
    df_explode = df_explode.merge(Movie_bis[['Freebase_movie_ID', 'Movie_genres', 'Average rating']], on='Freebase_movie_ID', how='left')
    df_explode = df_explode.explode('Movie_genres')

    # Count the number of movie per genre
    genre_counts = df_explode.groupby('Freebase_actor_ID')['Movie_genres'].value_counts().unstack(fill_value=0)
    genre_counts = genre_counts.apply(pd.to_numeric)

    # Calculate the average rating per genre
    genre_ratings = df_explode.groupby(['Freebase_actor_ID', 'Movie_genres'])['Average rating'].mean().unstack(fill_value=0)
    genre_ratings = genre_ratings.apply(pd.to_numeric)
    
    # Computing the top genres per actor
    Top_X_genre = genre_counts.apply(lambda row: list(row.sort_values(ascending=False).items()), axis=1)
    
    # Merging the results
    genre_ratings_counts = pd.DataFrame(index=genre_counts.index)
    genre_ratings_counts[f'Top_1_Genre'] = Top_X_genre.apply(lambda x: f"{list(x)[1][0]}" if x != np.nan else None)
    genre_ratings_counts[f'Top_1_Number'] = Top_X_genre.apply(lambda x: f"{list(x)[0][1]}" if x != np.nan else None)
    genre_ratings_counts[f'Top_1_Rating'] = genre_ratings_counts.apply(
        lambda row: genre_ratings.loc[row.name, row[f'Top_1_Genre']]
        if pd.notna(row[f'Top_1_Genre']) and row[f'Top_1_Genre'] in genre_ratings.columns
        else None,
        axis=1
    )
    return genre_ratings_counts


def Actor_genre_count_ratings_2(Actor, Movie, min_movies=5):
    """
    Compute the number of movies per genre and the average ratings of actors per genre.
    Actors are considered only if they have played in at least `min_movies` movies.
    """
    # Making a copy of the dataframes
    Actor_bis = Actor.copy()
    Movie_bis = Movie.copy()

    # Splitting the genre column into lists
    Movie_bis['Movie_genres'] = Movie_bis['Movie_genres'].str.split(', ')
    
    # Filter actors who have played in at least `min_movies` films
    actors_with_min_x_movies = Actor_bis[Actor_bis['actor_age_atmovierelease'].apply(len) >= min_movies]
    
    # Explode the actor's movie list into separate rows and filter by movie genre
    df_explode = actors_with_min_x_movies.explode('Freebase_movie_ID')
    df_explode = df_explode.merge(Movie_bis[['Freebase_movie_ID', 'Movie_genres', 'Average rating']], on='Freebase_movie_ID', how='left')
    df_explode = df_explode.explode('Movie_genres')

    # Count the number of movie per genre
    genre_counts = df_explode.groupby('Freebase_actor_ID')['Movie_genres'].value_counts().unstack(fill_value=0)
    genre_counts = genre_counts.apply(pd.to_numeric)

    # Calculate the average rating per genre
    genre_ratings = df_explode.groupby(['Freebase_actor_ID', 'Movie_genres'])['Average rating'].mean().unstack(fill_value=0)
    genre_ratings = genre_ratings.apply(pd.to_numeric)
    
    # Computing the top genres per actor
    Top_X_genre = genre_counts.apply(lambda row: list(row.sort_values(ascending=False).items()), axis=1)
    
    # Merging the results
    genre_ratings_counts = pd.DataFrame(index=genre_counts.index)
    for i in range(5):
        genre_ratings_counts[f'Top_{i+1}_Genre'] = Top_X_genre.apply(lambda x: f"{list(x)[i][0]}" if i < len(x) else None)
        genre_ratings_counts[f'Top_{i+1}_Number'] = Top_X_genre.apply(lambda x: f"{list(x)[i][1]}" if i < len(x) else None)
        genre_ratings_counts[f'Top_{i+1}_Rating'] = genre_ratings_counts.apply(
        lambda row: genre_ratings.loc[row.name, row[f'Top_{i+1}_Genre']]
        if pd.notna(row[f'Top_{i+1}_Genre']) and row[f'Top_{i+1}_Genre'] in genre_ratings.columns
        else None,
        axis=1
    )
    return genre_ratings_counts