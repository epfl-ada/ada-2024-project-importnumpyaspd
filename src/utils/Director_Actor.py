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
import plotly.graph_objects as go
import plotly.subplots as sp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter 

# j'ai enlevé les bails à propos des realease date car je n'en avais pas besoin
def create_actor_network_V2(Actors, Movie, min_movies=50):
    """
    Create the actor network for actors that played in at least `min_movies` movies. 
    Each node is an actor with name, gender, ethnicity, and height as attributes.
    The weight of the edges corresponds to the number of times they played together.

    Parameters:
    Actors (pd.DataFrame): A dataframe containing all the information of the actors.
    Movie (pd.DataFrame): A dataframe containing all the information of the movies.
    min_movies (int): A threshold to use the actors that played in at least `min_movies` movies.
    
    Returns:
    NetworkX Graph: A graph representing the connections between actors.
    """
    
    # Filter actors who have played in at least `min_movies` films
    actors_with_min_x_movies = Actors[Actors['actor_age_atmovierelease'].apply(len) >= min_movies]
    
    # Explode the actor's movie list into separate rows
    actors_df = actors_with_min_x_movies.explode('Freebase_movie_ID')

    G = nx.Graph()

    # Create edges between actors who acted in the same movie
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


def Create_Network_Actor_and_Attribute(Actor,Movie,min_movie=5):
    """
    Create the network of the actors if they played together and compute atrtibutes from the network
    Add these attrbiutes (degree of the node, eigenvector centrality) to the Actor
    """
    start_time = time.time()
    played_together = create_actor_network_V2(Actor,Movie,min_movie) 
    
    #remove the nodes that are nans                
    nan_nodes = [node for node in played_together.nodes if pd.isna(node)] 
    played_together.remove_nodes_from(nan_nodes)

    # Adding attribute from the network to the database of actors 
    degrees_actors = dict(played_together.degree())
    Actor["Degree_final"] = Actor["Freebase_actor_ID"].map(degrees_actors)
    eigenvector_centrality_actors = nx.eigenvector_centrality(played_together) 
    Actor["eigenvector_centrality_final"] = Actor["Freebase_actor_ID"].map(eigenvector_centrality_actors)
    
    print(f"Computation time:{time.time()-start_time:.2f}")
    
    return played_together, Actor


def create_directed_bipartite_network_actor_director(Actors, Directors, min_movies=50):
    """
    Create a directed bipartite network between actors and directors.
    """
    
    # Calculate the number of films for each actor
    Actors["Number_of_films"] = Actors["actor_age_atmovierelease"].apply(len)
    
    # Filter actors with at least `min_movies` films
    actors_with_min_x_movies = Actors[Actors["Number_of_films"] >= min_movies]

    # Explode the DataFrame to have one row per film per director
    director_df = Directors.explode('Freebase_movie_ID')
    
    # Explode the DataFrame to have one row per film per actor
    actors_df = actors_with_min_x_movies.explode('Freebase_movie_ID')

    # Merge actors with movies to include directors
    actors_with_directors = actors_df.merge(director_df[['Freebase_movie_ID', 'IMDb_director_ID']], on='Freebase_movie_ID', how='left')
    
    # Explode the directors list to have one row per director per actor
    actors_with_directors = actors_with_directors.explode('IMDb_director_ID')

    B = nx.DiGraph()

    # Create nodes for actors
    for _, actor in actors_with_min_x_movies.iterrows():
        B.add_node(actor['Freebase_actor_ID'], bipartite=0)

    # Create nodes for directors
    for _, director in Directors.iterrows():
        B.add_node(director['IMDb_director_ID'], bipartite=1)
    
    # Create edges
    for _, actor in actors_with_directors.iterrows():
        actor_id = actor['Freebase_actor_ID']
        director_id = actor['IMDb_director_ID']
        # Skip rows without director information
        if pd.isna(director_id): 
            continue
        # Checking if edge already exists
        if B.has_edge(actor_id, director_id):
            B[actor_id][director_id]['weight'] += 1
        else:
            B.add_edge(actor_id, director_id, weight=1)

    # Remove nodes with no edges
    B.remove_nodes_from(list(nx.isolates(B)))
    
    # Add attributes to each actor node
    for _, actor in actors_with_min_x_movies.iterrows():
        actor_id = actor['Freebase_actor_ID']
        if actor_id in B:
            B.nodes[actor_id].update({
                'name': actor['actor_name'],
                'gender': actor.get('actor_gender', None),
                'ethnicity': actor.get('ethnicity', None),
                'height': actor.get('actor_height', None),
            })
        
    # Add attributes to each director node
    for _, director in Directors.iterrows():
        director_id = director['IMDb_director_ID']
        if director_id in B:
            B.nodes[director_id].update({
               'name': director['director_name']
            })

    nodes_actor, nodes_director = nx.bipartite.sets(B)
    for director in nodes_director:
        sum_weight = sum(B[actor][director]['weight'] for actor in nodes_actor if B.has_edge(actor, director))
        if director in B:
            B.nodes[director].update({
               'weight': sum_weight
            })
    
    return B


def Create_Bipartite_Director(Actor_Success, Director, mon_movies=10):
    """
    Create the directed bipartite network between successful actors and directors.
    """
    # Create the bipartite directed network
    Di_B_network = create_directed_bipartite_network_actor_director(Actor_Success, Director, min_movies=mon_movies)
    
    # Remove nodes with missing values (NaN)
    nan_nodes = [node for node in Di_B_network.nodes if pd.isna(node)] 
    Di_B_network.remove_nodes_from(nan_nodes)
    
    # Verify the form of the graph
    is_bipartite = nx.is_bipartite(Di_B_network)
    is_directed = nx.is_directed(Di_B_network)

    if not is_bipartite or not is_directed :
        print("Is the graph bipartite?", is_bipartite)
        print("Is the graph directed?", is_directed)
    else :
        print("Bipartite graph generated successfully!")
    
    return Di_B_network
    

def Create_Director_Cluster_Highlight(Director_Success_KNN):
    Director_list = ['Quentin Tarantino', 'Woody Allen', 'Steven Spielberg',
                     'Lesli Linka Glatte', 'Thomas Schlamme', 'James Redford', 'Jed Johnson',
                     'David Miller', 'Mick Jackson', 'Paul Mazursky',
                     'Jack Sholder', 'Ernest Pintoff', 'Preston A. Whitmore II']
    Director_highlight = Director_Success_KNN.copy()
    Director_highlight = Director_highlight[Director_highlight['director_name'].isin(Director_list)]
    Director_highlight.sort_values(by=['Cluster_Label'],inplace=True)

    return Director_highlight
    

def add_start_age(person_df, person_type='actor'):
    """
    Adds a column to the person profile containing the starting age of their career.
    Works for both 'actor' and 'director'.

    person_type: 'actor' or 'director'
    """
    if person_type == 'actor':
        age_column = 'actor_age_atmovierelease'
    elif person_type == 'director':
        age_column = 'age_at_movie_release'

    # Handle ages
    person_df[age_column] = person_df[age_column].apply(
        lambda x: [-1 if pd.isna(age) or age < 0 else int(age) for age in x]
    )
    person_df["Career_Start_age"] = person_df.apply(
        lambda row: min((val for val in row[age_column] if val > 0), default=np.nan), axis=1
    )
    
    return person_df


def add_film_count(person_df, person_type='actor'):
    """
    Calculates the number of films the actor or director has been involved in.
    Works for both 'actor' and 'director'.

    person_type: 'actor' or 'director'
    """
    if person_type == 'actor':
        age_column = 'actor_age_atmovierelease'
    elif person_type == 'director':
        age_column = 'age_at_movie_release'

    # Number of films
    person_df['Number_of_films'] = person_df[age_column].apply(len)
    
    return person_df

def add_ratings(person_df, movie_df, person_type='actor'):
    """
    Creates a list of ratings for each movie that an actor or director has worked on.

    person_type: 'actor' or 'director'
    """
    if person_type == 'actor':
        id_column = 'Freebase_actor_ID'
        movie_column = 'Freebase_movie_ID'
    elif person_type == 'director':
        id_column = 'IMDb_director_ID'
        movie_column = 'Freebase_movie_ID'
    
    # Ratings per person
    df_explode = person_df.explode(movie_column)
    df_explode = df_explode.merge(movie_df[['Freebase_movie_ID','Average rating']], on='Freebase_movie_ID', how='left')
    ratings = df_explode.groupby(id_column)['Average rating'].apply(list)
    person_df = person_df.merge(ratings.rename('Ratings'), on=id_column, how='left')

    return person_df


def compute_mean_rating(person_df):
    """
    Compute the mean rating of the movies that the person worked on (actor/director).
    """
    person_df['Mean_Rating'] = person_df['Ratings'].apply(
        lambda ratings: sum(ratings) / len(ratings) if ratings and all(pd.notna(ratings)) else None
    )

    return person_df


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
    Mean_per_genre = Mean_per_genre.iloc[1:] # get rid of the nan

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


def Final_Pre_processing(Actor_original,Actor_5,Director,Genre_counts_200):
    # verify that each top genres is also in our genre with more than 200 movies
    df_actors = pd.DataFrame(np.where(Actor_5['Top_1_Genre'].isin(Genre_counts_200['Movie_genres']),
                                      Actor_5['Top_1_Genre'], 'NO_MATCH'), index=Actor_5.index)
    df_directors = pd.DataFrame(np.where(Director['Top_1_Genre'].isin(Genre_counts_200['Movie_genres']), 
                                         Director['Top_1_Genre'], 'NO_MATCH'), index=Director.index)
    
    #print(f"Number of actors which their primary genre is not present : {df_actors[df_actors[0]=='NO_MATCH'].shape[0]}")
    #print(f"Number of directors which their primary genre is not present : {df_directors[df_directors[0]=='NO_MATCH'].shape[0]}")

    # Dropping the actor from which their top genre is not in our list of top genres
    to_drop = df_actors[df_actors[0]=='NO_MATCH'].index
    Actor_final =  Actor_5.drop(to_drop)
    
    # Dropping the director from which their top genre is not in our list of top genres
    to_drop = df_directors[df_directors[0]=='NO_MATCH'].index
    Director_final =  Director.drop(to_drop)
    
    print(f"In the end, we have {Actor_final.shape[0]} actors and {Director_final.shape[0]} directors .")

    return Actor_final, Director_final


def Eigenvector_Mean_Median(Actor):
    """
    actor above average and median centrality
    """
    Mean = Actor['eigenvector_centrality_final'].mean()
    Median = Actor['eigenvector_centrality_final'].median()

    Actor_mean = Actor[Actor['eigenvector_centrality_final'] >= Mean]
    Actor_median = Actor[Actor['eigenvector_centrality_final'] >= Median]

    return Actor_mean, Actor_median


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
        played_together = create_actor_network_V2(actor, Movie, 5)
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


def Comparing_Mean_Median_Louvain(Keep_Mean,Keep_Median,Top_Louvain_Centrality):
    # Creating a list of all the index of actors from the Louvain method that are above the mean centrality of their group
    Louvain_method = 0
    keep_list_Louvain = []
    for cluster, actor in Top_Louvain_Centrality.items():
        Louvain_method = actor.shape[0] + Louvain_method
        keep_list_Louvain.extend(actor['Freebase_actor_ID'].tolist())
    
    # Comparing which actors have been taken for each method
    MeanVsMethod2 = Keep_Mean['Freebase_actor_ID'].isin(keep_list_Louvain)
    MedianVsMethod2 = Keep_Median['Freebase_actor_ID'].isin(keep_list_Louvain)
    keep_list_Louvain_df = pd.DataFrame(keep_list_Louvain,columns=['Freebase_actor_ID'])
    Method2VsMedian = keep_list_Louvain_df['Freebase_actor_ID'].isin(Keep_Median['Freebase_actor_ID'])

    # Printing the result
    print(f"Number of actors which are above the global mean of the eigenvector centrality : {Keep_Mean.shape[0]}")
    print(f"Number of actors which are above the global median of the eigenvector centrality : {Keep_Median.shape[0]}")
    print(f"Number of actors which are above the mean of their community eigenvector centrality : {Louvain_method}")
    
    # Printing the comparaison
    print('------------')
    print(f"Number of actors which are above the global mean but not above their community mean :")
    print(f"{MeanVsMethod2.value_counts()}")
    print('------------')
    print(f"Number of actors which are above the global median but not above their community mean :")
    print(f"{MedianVsMethod2.value_counts()}")
    print('------------')
    print(f"Number of actors which are above their community mean but not above the global median :")
    print(f"{Method2VsMedian.value_counts()}")

    return None


def keep_above_mean_rating(Actor,Genre_counts):
    """
    return actor above mean ratging for given genre movie
    """
    # Defining empty list for the results 
    keep = []
    Actor_bis = Actor.copy()
    Actor_bis = Actor_bis.set_index('Freebase_actor_ID',drop=False)

    # Checking for each actor if they are better than the average rating in their genre
    for _,Act in tqdm(Actor_bis.iterrows(), desc="Checking their relative ranking"):
        Actor_id = Act['Freebase_actor_ID']
        Top_1 = Act['Top_1_Genre']
        Rating = Act['Top_1_Rating']
        for _,Genre in Genre_counts.iterrows():
            if Top_1 == Genre['Movie_genres'] and Rating >= Genre['Average rating']:
                keep.append(Actor_id)
                continue

    # Keeping only the ones above the mean
    Actor_new = Actor_bis[Actor_bis.index.isin(keep)].reset_index(drop=True)

    return Actor_new, keep


def keep_successful_actor(Actor, Genre_counts):
    Actor_bis = Actor.set_index('Freebase_actor_ID',drop=False)
    # Computing the criteria
    Actor_12 = Actor_bis[Actor_bis['Number_of_films'] >= 12]
    Actor_Rating,_ = keep_above_mean_rating(Actor_bis,Genre_counts)
    Actor_Centrality,_ = Eigenvector_Mean_Median(Actor_bis)

    # Applying the criteria
    Actor_Success = Actor_12.copy()
    Actor_Success = Actor_Success[Actor_Success['Freebase_actor_ID'].isin(Actor_Rating['Freebase_actor_ID'])]
    Actor_Success = Actor_Success[Actor_Success['Freebase_actor_ID'].isin(Actor_Centrality['Freebase_actor_ID'])]

    Actor_Success = Actor_Success.reset_index(drop=True)

    return Actor_Success


def prepare_director_dataset_KNN(Director):
    """
    drop columns and scale in prevision of kmeans
    """
    
    Director_bis = Director.dropna(subset='Mean_Rating')
    columns_to_drop = ['IMDb_director_ID', 'director_name', 'birthYear_director','Career_Start_age','deathYear_director', 
                       'Freebase_movie_ID','age_at_movie_release', 'Ratings', 'Top_1_Genre', 'Top_1_Number', 
                       'Top_1_Rating','Freebase_actor_ID','Weight']
    
    Director_dataset = Director_bis.drop(columns=columns_to_drop, errors='ignore')
    scaler = StandardScaler()
    Director_dataset_std = scaler.fit_transform(Director_dataset)

    return Director_dataset, Director_dataset_std


def knn_clustering_director(Director, Director_dataset_std, n_clusters, random_state=0):
    """
    perform knn
    """
    # Filter out the rows where Mean_Rating is NaN
    Director_labels = Director.dropna(subset='Mean_Rating').copy()  # Use `.copy()` to avoid view modification issues
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(Director_dataset_std)
    
    # Assign cluster labels using `.loc` to avoid SettingWithCopyWarning
    Director_labels.loc[:, 'Cluster_Label'] = labels  # Assign labels to the 'Cluster_Label' column
    
    return Director_labels


def Count_Edge_Distribution(Director_Success):
    # Creating a list containing all the weight of the edges
    all_weight = [weight for weight_list in Director_Success['Weight'] for weight in weight_list]

    # Using built-in function to count the occurrences
    counts = Counter(all_weight)
    
    # Dataframe to store the results
    weights_edge = pd.DataFrame(counts.items(), columns=['Weight', 'Occurrences'])
    weights_edge.sort_values(by=['Weight'], inplace=True, ascending=True)
    
    return weights_edge


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
    Top_genre = genre_counts.apply(lambda row: list(row.sort_values(ascending=False).items()), axis=1)
    
    # Merging the results
    genre_ratings_counts = pd.DataFrame(index=genre_counts.index)
    genre_ratings_counts[f'Top_1_Genre'] = Top_genre.apply(lambda x: f"{list(x)[1][0]}" if x != np.nan else None)
    genre_ratings_counts[f'Top_1_Number'] = Top_genre.apply(lambda x: f"{list(x)[0][1]}" if x != np.nan else None)
    genre_ratings_counts[f'Top_1_Rating'] = genre_ratings_counts.apply(
        lambda row: genre_ratings.loc[row.name, row[f'Top_1_Genre']]
        if pd.notna(row[f'Top_1_Genre']) and row[f'Top_1_Genre'] in genre_ratings.columns
        else None,
        axis=1
    )
    return genre_ratings_counts


def Director_genre_count_ratings(Director, Movie, min_movies=1):
    """
    Compute the number of movies per genre and the average ratings of directors per genre.
    Directors are considered only if they have played in at least "min_movies" movies.
    """
    # Making a copy of the dataframes
    Director_bis = Director.copy()
    Movie_bis = Movie.copy()

    # Splitting the genre column into lists
    Movie_bis['Movie_genres'] = Movie_bis['Movie_genres'].str.split(', ')
    
    # Filter actors who have played in at least `min_movies` films
    directors_with_min_x_movies = Director_bis[Director_bis['age_at_movie_release'].apply(len) >= min_movies]
    
    # Explode the actor's movie list into separate rows and filter by movie genre
    df_explode = directors_with_min_x_movies.explode('Freebase_movie_ID')
    df_explode = df_explode.merge(Movie_bis[['Freebase_movie_ID', 'Movie_genres', 'Average rating']], on='Freebase_movie_ID', how='left')
    df_explode = df_explode.explode('Movie_genres')

    # Count the number of movie per genre
    genre_counts = df_explode.groupby('IMDb_director_ID')['Movie_genres'].value_counts().unstack(fill_value=0)
    genre_counts = genre_counts.apply(pd.to_numeric)

    # Calculate the average rating per genre
    genre_ratings = df_explode.groupby(['IMDb_director_ID', 'Movie_genres'])['Average rating'].mean().unstack(fill_value=0)
    genre_ratings = genre_ratings.apply(pd.to_numeric)
    
    # Computing the top genres per actor
    Top_genre = genre_counts.apply(lambda row: list(row.sort_values(ascending=False).items()), axis=1)
    
    # Merging the results
    genre_ratings_counts = pd.DataFrame(index=genre_counts.index)
    genre_ratings_counts[f'Top_1_Genre'] = Top_genre.apply(lambda x: f"{list(x)[1][0]}" if x != np.nan else None)
    genre_ratings_counts[f'Top_1_Number'] = Top_genre.apply(lambda x: f"{list(x)[0][1]}" if x != np.nan else None)
    genre_ratings_counts[f'Top_1_Rating'] = genre_ratings_counts.apply(
        lambda row: genre_ratings.loc[row.name, row[f'Top_1_Genre']]
        if pd.notna(row[f'Top_1_Genre']) and row[f'Top_1_Genre'] in genre_ratings.columns
        else None,
        axis=1
    )
    return genre_ratings_counts


def director_success1(Network, Director):
    """
    get directors that are link to successfull actor 
    """
    # Compute actor and director parts of the network
    nodes_actor, nodes_director = nx.bipartite.sets(Network)
    
    # Filter only the director part of the network
    Director_Success = Director[Director['IMDb_director_ID'].isin(nodes_director)].copy()
    Director_Success['Freebase_actor_ID'] = None  
    Director_Success['Weight'] = None  
    Director_Success['Sum_Weight'] = 0  
    Director_Success["Number_of_actors"] = None

    # Compute weights, total weights, and the actors for each director
    for director in nodes_director:
        Actor_ID = []
        Weight = []
        Sum_Weight = 0

        for actor in nodes_actor:
            edge_data = Network.get_edge_data(actor, director)
            if edge_data:
                Actor_ID.append(actor)
                Weight.append(edge_data["weight"])
                Sum_Weight += edge_data["weight"]
        
        # Update the director attributes
        idx = Director_Success[Director_Success['IMDb_director_ID'] == director].index[0]
        Director_Success.at[idx, 'Freebase_actor_ID'] = Actor_ID
        Director_Success.at[idx, 'Weight'] = Weight
        Director_Success.at[idx, 'Sum_Weight'] = Sum_Weight
    
    # Compute the number of actors for each director
    Director_Success["Number_of_actors"] = Director_Success["Freebase_actor_ID"].apply(len)

    return Director_Success

def merge_top_genre(df_people, df_top_genre, person_type='actor', printt = True):
    """
    merge top genre with a dataset of actor or director
    """
    if printt :
        print(f"There were {df_people.shape[0]} {person_type}s.")

    if person_type == 'actor':
        person_id_column = 'Freebase_actor_ID'
    elif person_type == 'director':
        person_id_column = 'IMDb_director_ID'
    
    # Perform the merge on the person_id_column and the index of top_genre_df
    df_people = df_people.merge(df_top_genre, how='inner', left_on=person_id_column, right_index=True)
    if printt:
        print(f"There are now {df_people.shape[0]} {person_type}s after merging and dropping rows where the top genre was not known.")
    
    return df_people