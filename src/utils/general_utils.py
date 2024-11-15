import networkx as nx
import pandas as pd
from itertools import combinations

def create_actor_network(Actors, min_movies=50):
    """
    Create the actor network for actors that played in at least min_movies movies. 
    Each node is an actor with name, gender, ethnicity and height as attributs.
    The weight of the edges correspond to the number of times they played together.

    Parameters:
    Actors (pd.DataFrame): A dataframe containing all the informraiton of the actors.

    Returns:
    Network object: A network (NetworkX object) reprenting the connexions between actors.
    """
    
    # Calculate the number of films for each actor
    Actors["Number_of_films"] = Actors["actor_age_atmovierelease"].apply(len)
    
    # Filter actors with at least `min_movies` films
    actors_with_min_x_movies = Actors[Actors["Number_of_films"] > min_movies]
    
    # Explode the DataFrame to have one row per film per actor
    actors_df = actors_with_min_x_movies.explode('Freebase_movie_ID')
    G = nx.Graph()

    # Group by movie ID and create edges for actors in the same movie
    # Each edges as a wiegth equal to the number of times they played together
    for movie_id, group in actors_df.groupby('Freebase_movie_ID'):
        actor_ids = group['Freebase_actor_ID'].tolist()
        for actor1, actor2 in combinations(actor_ids, 2):
            if actor1 != actor2:
                if G.has_edge(actor1, actor2):
                    G[actor1][actor2]['weight'] += 1
                else:
                    G.add_edge(actor1, actor2, weight=1)

    # Add attributes to each actor node
    for _, row in actors_df.iterrows():
        actor_id = row['Freebase_actor_ID']
        if actor_id in G:
            G.nodes[actor_id].update({
                'name': row['actor_name'],
                'gender': row.get('actor_gender', None),
                'ethnicity': row.get('ethnicity', None),
                'height': row.get('actor_height', None)
            })
    return G