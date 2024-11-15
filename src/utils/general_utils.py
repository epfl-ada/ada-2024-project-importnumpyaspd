import networkx as nx
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import numpy as np

def create_actor_network(Actors, Movie, min_movies=50, min_releasedate=0):
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
    actors_with_min_x_movies = Actors[Actors['actor_age_atmovierelease'].apply(len) > min_movies]
    
    # Explode the actor's movie list into separate rows and filter by movie release date
    actors_df = actors_with_min_x_movies.explode('Freebase_movie_ID')

    # Precompute movie release dates for faster lookup
    movie_releasedates = Movie.set_index('Freebase_movie_ID')['release_date'].to_dict()

    G = nx.Graph()

    for movie_id, group in tqdm(actors_df.groupby('Freebase_movie_ID'), desc="Creating network"):
        if movie_releasedates.get(movie_id, 0) >= min_releasedate:  # Filter by movie release date
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


def create_director_network(Directors, min_movies=50):
    # Calculate the number of films for each director
    Directors["Number_of_films"] = Directors["age_at_movie_release"].apply(len)
    
    # Filter directors with at least `min_movies` films
    directors_with_min_x_movies = Directors[Directors["Number_of_films"] > min_movies]
    
    # Explode the DataFrame to have one row per film per director
    directors_df = directors_with_min_x_movies.explode('Freebase_movie_ID')
    G = nx.Graph()

    # Group by movie ID and create edges for directors in the same movie
    for movie_id, group in directors_df.groupby('Freebase_movie_ID'):
        director_ids = group['IMDb_director_ID'].tolist()
        for director1, director2 in combinations(director_ids, 2):
            if director1 != director2:
                if G.has_edge(director1, director2):
                    G[director1][director2]['weight'] += 1
                else:
                    G.add_edge(director1, director2, weight=1)

    # Add attributes to each director node
    for _, row in directors_df.iterrows():
        director_id = row['IMDb_director_ID']
        if director_id in G:
            G.nodes[director_id].update({
                'name': row['director_name']
            })
    return G


def create_actor_director_network(Actors, Movie, min_movies=50):
    # Calculate the number of films for each actor
    Actors["Number_of_films"] = Actors["actor_age_atmovierelease"].apply(len)
    
    # Filter actors with at least `min_movies` films
    actors_with_min_x_movies = Actors[Actors["Number_of_films"] > min_movies]
    
    # Explode the DataFrame to have one row per film per actor
    actors_df = actors_with_min_x_movies.explode('Freebase_movie_ID')

    # Merge actors with movies to include directors
    actors_with_directors = actors_df.merge(Movie[['Freebase_movie_ID', 'IMDb_director_ID']], on='Freebase_movie_ID', how='left')
    
    # Explode the directors list to have one row per director per actor
    actors_with_directors = actors_with_directors.explode('IMDb_director_ID')
    
    G = nx.Graph()

    # Group by director and create edges for actors working under the same director
    for director, group in actors_with_directors.groupby('IMDb_director_ID'):
        if pd.isna(director):  # Skip rows without director information
            continue
        actor_ids = group['Freebase_actor_ID'].tolist()
        for actor1, actor2 in combinations(actor_ids, 2):
            if actor1 != actor2:
                if G.has_edge(actor1, actor2):
                    G[actor1][actor2]['weight'] += 1
                else:
                    G.add_edge(actor1, actor2, weight=1)

    # Add attributes to each actor node
    for _, row in actors_with_directors.iterrows():
        actor_id = row['Freebase_actor_ID']
        if actor_id in G:
            G.nodes[actor_id].update({
                'name': row['actor_name'],
                'gender': row.get('actor_gender', None),
                'ethnicity': row.get('ethnicity', None),
                'height': row.get('actor_height', None)
            })
    
    return G