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
from plotly.subplots import make_subplots
import plotly.express as px

from src.utils.Director_Actor import *

def Elbow_method_genre(Movie,save=False):
    # Compute the number of genres of movies for a specific number and add the result to a list
    total_genres_per_number = []
    Number_of_movies = []
    
    for i in range(20):
        a = i * 50
        genre_counts = Number_per_genre(Movie, Min_movie=a)
        total_genres_per_number.append(len(genre_counts))
        Number_of_movies.append(a)
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the data trace for Total Genres
    fig.add_trace(go.Scatter(
        x=Number_of_movies,
        y=total_genres_per_number,
        mode='lines+markers',
        name='Total Genres',
        line=dict(color='blue'),
        marker=dict(symbol='circle', size=8)
    ))
    
    # Update the layout
    fig.update_layout(
        title='Elbow Method',
        xaxis_title='Minimum Movies per Genre',
        yaxis_title='Total Number of Genres',
        template='plotly',
        width=700,
        height=500
    )
    
    # Show and save the plot
    if save:
        fig.write_html('Elbow_method_genre.html')
    fig.show()

    return None


def Plot_Degree_Actor(Actor,save=False, describe = False):
    data = Actor['Degree_final'].dropna()

    # mean and median
    mean_degree = np.mean(data)
    median_degree = np.median(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Actor Degree (linear scale)
    sns.histplot(data, bins=150, color='skyblue', ax=axes[0])
    axes[0].axvline(mean_degree, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].axvline(median_degree, color='green', linestyle='-', linewidth=2, label='Median')
    axes[0].set_title('Degree Actor on Film', fontsize=14, fontweight="bold")
    axes[0].set_xlabel('Degree', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend(fontsize=10)
    
    # Actor Degree (log scale)
    sns.histplot(data, bins=150, color='skyblue', ax=axes[1])
    axes[1].axvline(mean_degree, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(median_degree, color='green', linestyle='-', linewidth=2, label='Median')
    axes[1].set_yscale('log')
    axes[1].set_title('Degree Actor on Film (Log Scale)', fontsize=14, fontweight="bold")
    axes[1].set_xlabel('Degree', fontsize=12)
    axes[1].set_ylabel('Frequency (Log Scale)', fontsize=12)
    axes[1].legend(fontsize=10)
    
    # Shwo the plot and save
    plt.tight_layout()
    if save:
        plt.savefig('Plot_Degree_Actor.png', transparent=True)
    plt.show()

    if describe :
        # Computing and plotting some statistical values
        print('Describe of the degree of actor on film')
        print(Actor['Degree_final'].describe())
    
    return None


def Plot_NOF_Actor(Actor,max_movie=200,save=False):
    # mean and median
    mean_actor = np.mean(Actor["Number_of_films"])
    median_actor = np.median(Actor["Number_of_films"])
    
    # Actor age at movie release
    sns.histplot(Actor[Actor['Number_of_films']<max_movie]['Number_of_films'], bins=200, color='skyblue')
    plt.yscale('log')
    plt.axvline(mean_actor, color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(median_actor, color='green', linestyle='-', linewidth=2, label='Median')
    plt.title('Distribution of number of films per actor',fontsize=14, fontweight="bold")
    plt.xlabel('Number of movie',fontsize=12)
    plt.ylabel('Frequency',fontsize=12)
    plt.legend(fontsize=10)
    
    
    # Show and save the plot
    plt.tight_layout()
    if save:
        plt.savefig('Plot_NOF_Actor.png', transparent=True)
    plt.show()

    if describe:
        # Computing and plotting some statistical values
        print('Describe of the degree of actor on film')
        print(Actor['Degree_final'].describe())
        print('--------------')
        print('Describe of the number of films per actor')
        print(Actor['Number_of_films'].describe())

    return None


def Print_result_degree(Actor_orignal,Actor_final):
    Mean_degree = Actor_final['Degree_final'].mean()

    # Computing different mean (per actor, per movie)
    print(f'Mean degree of actor : {Mean_degree:.2f}')
    print(f'Mean number of actor per movie for all movies : {Mean_actor_per_movie(Actor_orignal):.2f}')
    print(f'Mean number of movies per actor for all actors : {Mean_movie_per_actor(Actor_orignal):.2f}')
    print(f'Mean number of movies per actor for actors that have played at least 5 movies : {Mean_movie_per_actor(Actor_final):.2f}')
    print(f'Median number of movies per actor for actors that have played at least 5 movies : 9')
    
    print('------------')
    
    # Computing and printing statistical values from the describe above with the mean
    print(f'Mean degree of actor divided by mean number of actor per movie for all movies : {Mean_degree/Mean_actor_per_movie(Actor_orignal):.2f}')
    print(f'Mean number of actor per movie for all movies multiplied by the mean number of movies per actor (5 movies) :{Mean_actor_per_movie(Actor_orignal)*Mean_movie_per_actor(Actor_final):.2f}')
    print(f'Mean number of actor per movie for all movies multiplied by the median of the number of movies per actor (5 movies) : {Mean_actor_per_movie(Actor_orignal)*9:.2f}')

    return None


def Plot_Centrality_Actor(Actor,save=False, describe = False):
    # Set the figure and subplots layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    data = Actor['eigenvector_centrality_final'].dropna()

    mean_actor = np.mean(data)
    median_actor = np.median(data)
    
    # Eigenvector centrality 
    sns.histplot(data, bins=150, color='skyblue', ax=axes[0])
    axes[0].set_title('Eigenvector centrality Actor on film',fontsize=14, fontweight="bold")
    axes[0].axvline(mean_actor, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].axvline(median_actor, color='green', linestyle='-', linewidth=2, label='Median')
    axes[0].set_xlabel('Eigenvector centrality',fontsize=12)
    axes[0].set_ylabel('Frequency',fontsize=12)
    axes[0].legend(fontsize=10)
    
    # Eigenvector centrality log scale
    sns.histplot(data, bins=150, color='skyblue', ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title('Eigenvector centrality Actor on film',fontsize=14, fontweight="bold")
    axes[1].axvline(mean_actor, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(median_actor, color='green', linestyle='-', linewidth=2, label='Median')
    axes[1].set_xlabel('Eigenvector centrality',fontsize=12)
    axes[1].set_ylabel('Frequency',fontsize=12)
    axes[1].legend(fontsize=10)
    
    # Show the plot and save
    plt.tight_layout()
    if save:
        plt.savefig('Plot_Eigenvector_Actor.png', transparent=True)
    plt.show()
    if describe :
        print('Describe of the eigenvector centrality of actor on film')
        print(Actor['eigenvector_centrality_final'].describe())

    return None


def Plot_Centrality_Louvain_Actor(Actors_Groups,Cluster1='cluster_1',Cluster2='cluster_2',save=False):
    # Set the figure and subplots layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    data1 = Actors_Groups[Cluster1]
    data2 = Actors_Groups[Cluster2]
    
    # Eigenvector centrality 
    sns.histplot(data1['eigenvector_centrality_final'].dropna(), bins=50, color='skyblue', ax=axes[0])
    axes[0].set_title(f'Eigenvector centrality for Actor in {Cluster1}',fontsize=14, fontweight="bold")
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Eigenvector centrality',fontsize=12)
    axes[0].set_xlabel('Eigenvector centrality',fontsize=12)
    axes[0].set_ylabel('Frequency',fontsize=12)
    
    # Eigenvector centrality log scale
    sns.histplot(data2['eigenvector_centrality_final'].dropna(), bins=50, color='skyblue', ax=axes[1])
    axes[1].set_title(f'Eigenvector centrality for Actor in {Cluster2}',fontsize=14, fontweight="bold")
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Eigenvector centrality',fontsize=12)
    axes[1].set_ylabel('Frequency',fontsize=12)
    
    # Show the plot and save
    plt.tight_layout()
    if save:
        plt.savefig('Plot_Centrality_Louvain_Actor.png', transparent=True)
    plt.show()

    return None


def Plot_Mean_Rating_Actor(Actor_original,Actor,save=False):
    # mean and median
    mean_actor = np.mean(Actor_original['Mean_Rating'].dropna())
    median_actor = np.median(Actor_original['Mean_Rating'].dropna())
    
    mean_actor_5 = np.mean(Actor['Mean_Rating'].dropna())
    median_actor_5 = np.median(Actor['Mean_Rating'].dropna())
    
    # Set the figure and subplots layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Actor average rating per movie
    sns.histplot(Actor_original['Mean_Rating'].dropna(), bins=84, color='limegreen', ax=axes[0])
    axes[0].axvline(mean_actor, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].axvline(median_actor, color='green', linestyle='-', linewidth=2, label='Median')
    axes[0].set_title('All actors',fontsize=14, fontweight="bold")
    axes[0].set_xlabel('Average rating',fontsize=12)
    axes[0].set_ylabel('Frequency',fontsize=12)
    axes[0].legend(fontsize=10)
    
    # Actor with at least 5 films average rating per movie
    sns.histplot(Actor['Mean_Rating'].dropna(), bins=85, color='skyblue', ax=axes[1])
    axes[1].axvline(mean_actor_5, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(median_actor_5, color='green', linestyle='-', linewidth=2, label='Median')
    axes[1].set_title('Actor with at least 5 films',fontsize=14, fontweight="bold")
    axes[1].set_xlabel('Average rating',fontsize=12)
    axes[1].set_ylabel('Frequency',fontsize=12)
    axes[1].legend(fontsize=10)
    
    plt.suptitle('Distribution of the average rating of the movie', fontsize=16, fontweight="bold")
    
    # Plot
    plt.tight_layout()
    if save:
        plt.savefig('Plot_Mean_Rating_Actor.png')
    plt.show()
    
    print('Describe of the actor rating for all actors')
    print(Actor_original['Mean_Rating'].describe())
    print('------------')
    print('Describe of the actor rating for actors with more than 5 movies')
    print(Actor['Mean_Rating'].describe())

    return None


def Plot_Top_1_Rating_Vs_Mean_Rating(Actor,save=False):
    # mean and median
    mean_actor_genre = np.mean(Actor['Top_1_Rating'].dropna())
    median_actor_genre = np.median(Actor['Top_1_Rating'].dropna())
    
    mean_actor_5 = np.mean(Actor['Mean_Rating'].dropna())
    median_actor_5 = np.median(Actor['Mean_Rating'].dropna())
    
    # Set the figure and subplots layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Actor top 1 genre rating 
    sns.histplot(Actor['Top_1_Rating'].dropna(), bins=84, color='skyblue', ax=axes[0])
    axes[0].axvline(mean_actor_genre, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].axvline(median_actor_genre, color='green', linestyle='-', linewidth=2, label='Median')
    axes[0].set_title('Top 1 genre rating for actor with at least 5 film',fontsize=14, fontweight="bold")
    axes[0].set_xlabel('Average rating',fontsize=12)
    axes[0].set_ylabel('Frequency',fontsize=12)
    axes[0].legend(fontsize=10)
    
    # Actor with at lest 5 films average rating per movie
    sns.histplot(Actor['Mean_Rating'].dropna(), bins=85, color='skyblue', ax=axes[1])
    axes[1].axvline(mean_actor_5, color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1].axvline(median_actor_5, color='green', linestyle='-', linewidth=2, label='Median')
    axes[1].set_title('Mean rating of actor with at least 5 films',fontsize=14, fontweight="bold")
    axes[1].set_xlabel('Average rating',fontsize=12)
    axes[1].set_ylabel('Frequency',fontsize=12)
    axes[1].legend(fontsize=10)
    
    plt.suptitle('Distribution of the average rating of the movie', fontsize=16, fontweight="bold")
    
    # Plot
    plt.tight_layout()
    if save==True:
        plt.savefig('Plot_Top_1_Rating_Vs_Mean_Rating.png')
    plt.show()
    
    print(f"Mean of ratings of actor per genre : {mean_actor_genre:.3f}")
    print(f"Mean of ratings of actor : {mean_actor_5:.3f}")
    print(f"Median of ratings of actor per genre : {median_actor_genre:.3f}")
    print(f"Median of ratings of actor : {median_actor_5:.3f}")

    return None 


def Print_Comparison_Success(Actor_12_MOVIES, Actor_TOP_Mean_centraty, Actor_TOP_RATING):
    # Comparing which actors have been droppped for each criteria
    NumberFilmsVsRating = Actor_12_MOVIES['Freebase_actor_ID'].isin(Actor_TOP_RATING['Freebase_actor_ID'])
    CentralityVsRating = Actor_TOP_Mean_centraty['Freebase_actor_ID'].isin(Actor_TOP_RATING['Freebase_actor_ID'])
    CentralityVsNumberFilms = Actor_TOP_Mean_centraty['Freebase_actor_ID'].isin(Actor_12_MOVIES['Freebase_actor_ID'])
    
    
    # Printing the comparaison
    print(f"Actors which have played 12 films and are (or not) above their genre average rating:")
    print(f"{NumberFilmsVsRating.value_counts()}")
    print('------------')
    print(f"Actors above the global mean average and are (or not) above their genre average rating:")
    print(f"{CentralityVsRating.value_counts()}")
    print('------------')
    print(f"Actors above the global mean average and have (or not) played in at least 12 films:")
    print(f"{CentralityVsNumberFilms.value_counts()}")

    return None


def plot_elbow_method_Director(Director_dataset_std,cluster_range,random_state = 0,save=False):
    sse = []

    for n_clusters in tqdm(cluster_range):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(Director_dataset_std)
        sse.append(kmeans.inertia_)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(cluster_range),
        y=sse,
        mode='lines+markers',
        marker=dict(color='blue', size=10),
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        autosize = True,
        width=None,
        height = None,
        title="Elbow Method for Optimal Number of Clusters",
        xaxis_title="Number of Clusters",
        yaxis_title="SSE (Sum of Squared Errors)",
    )

    if save:
        fig.write_html('plot_elbow_method_Director.html')
    fig.show()

    return None


def Plot_Weight_Director(Director_Success_KNN, save=False):
    # Calculate the overall mean and median
    mean_Sum_Weight = np.mean(Director_Success_KNN['Sum_Weight'].dropna())
    median_Sum_Weight = np.median(Director_Success_KNN['Sum_Weight'].dropna())
    
    # Define the colors for each cluster 
    cluster_labels = Director_Success_KNN['Cluster_Label'].unique()
    colors = px.colors.qualitative.Plotly[:len(cluster_labels)]
    
    # Create the figure
    fig = go.Figure()

    # Add histograms and cluster mean lines
    for cluster, color in zip(cluster_labels, colors):
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]['Sum_Weight'].dropna()
        mean_cluster_weight = np.mean(cluster_data)
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=cluster_data,
            name=f'Cluster {cluster}',
            marker_color=color,
            opacity=0.5
        ))
        
        # Add mean line for the cluster
        fig.add_trace(go.Scatter(
            x=[mean_cluster_weight, mean_cluster_weight],
            y=[0, max(np.histogram(cluster_data, bins=50)[0])],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean'
        ))

    # Add overall mean and median lines
    fig.add_trace(go.Scatter(
        x=[mean_Sum_Weight, mean_Sum_Weight],
        y=[0, 1],
        mode='lines',
        line=dict(color='black', width=2),
        name='Overall Mean'
    ))
    fig.add_trace(go.Scatter(
        x=[median_Sum_Weight, median_Sum_Weight],
        y=[0, 1],
        mode='lines',
        line=dict(color='purple', width=2),
        name='Overall Median'
    ))

    # Update layout
    fig.update_layout(
        title='Total Weight of Each Director by Cluster',
        xaxis_title='Total Weight',
        yaxis_title='Frequency (Log Scale)',
        yaxis_type='log',
        barmode='overlay',
        legend=dict(
            title='Legend',
            font=dict(size=10),
            x=1.05,
            y=1
        )
    )
    
    # Save the figure if needed
    if save:
        fig.write_html('Plot_Weight_Director.html')
    
    # Show the figure
    fig.show()
    
    return None


def Plot_Edge_Weight_Distribution(Director_Success_KNN, save=False):
    # Creating a list containing all the weights of the edges
    all_weight = [weight for weight_list in Director_Success_KNN['Weight'] for weight in weight_list]
    
    # Calculate the mean and median of the edge's weight
    mean_Sum_Weight = np.mean(all_weight)
    median_Sum_Weight = np.median(all_weight)

    # Define cluster colors
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]
    
    # Create plot
    fig = go.Figure()
    
    for cluster, color in zip(clusters, colors):
        cluster_weights = [weight for weight_list in Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]['Weight']
                           for weight in weight_list]
        mean_cluster_weight = np.mean(cluster_weights)
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=cluster_weights,
            name=f'Cluster {cluster} (Weight)',
            marker_color=color,
            opacity=0.5
        ))
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=[mean_cluster_weight, mean_cluster_weight],
            y=[0, 1],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean (Weight)'
        ))
    
    # Add overall mean and median lines
    fig.add_trace(go.Scatter(
        x=[mean_Sum_Weight, mean_Sum_Weight],
        y=[0, 1],
        mode='lines',
        line=dict(color='black', width=2),
        name='Overall Mean (Weight)'
    ))
    fig.add_trace(go.Scatter(
        x=[median_Sum_Weight, median_Sum_Weight],
        y=[0, 1],
        mode='lines',
        line=dict(color='purple', width=2),
        name='Overall Median (Weight)'
    ))

    # Update layout
    fig.update_layout(
        title='Weight of Edge Distribution for All Directors',
        xaxis_title="Edge Weight Value",
        yaxis_title="Frequency (Log Scale)",
        yaxis_type="log",
        showlegend=True,
        height=600,
        width=800
    )
    
    if save:
        fig.write_html('Plot_Edge_Weight_Distribution.html')
    
    fig.show()


def Plot_Number_of_Edges(Director_Success_KNN, save=False):
    # Calculate the mean and median of the number of actors
    mean_Number_of_Actor = np.mean(Director_Success_KNN['Number_of_actors'])
    median_Number_of_Actor = np.median(Director_Success_KNN['Number_of_actors'])

    # Define cluster colors
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]
    
    # Create plot
    fig = go.Figure()
    
    for cluster, color in zip(clusters, colors):
        cluster_actors = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]['Number_of_actors']
        mean_cluster_actors = np.mean(cluster_actors)
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=cluster_actors,
            name=f'Cluster {cluster} (Actors)',
            marker_color=color,
            opacity=0.5
        ))
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=[mean_cluster_actors, mean_cluster_actors],
            y=[0, 1],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean (Actors)'
        ))
    
    # Add overall mean and median lines
    fig.add_trace(go.Scatter(
        x=[mean_Number_of_Actor, mean_Number_of_Actor],
        y=[0, 1],
        mode='lines',
        line=dict(color='black', width=2),
        name='Overall Mean (Actors)'
    ))
    fig.add_trace(go.Scatter(
        x=[median_Number_of_Actor, median_Number_of_Actor],
        y=[0, 1],
        mode='lines',
        line=dict(color='purple', width=2),
        name='Overall Median (Actors)'
    ))

    # Update layout
    fig.update_layout(
        title='Number of Edges (Actors) per Director',
        xaxis_title="Number of Edges",
        yaxis_title="Frequency (Log Scale)",
        yaxis_type="log",
        showlegend=True,
        height=600,
        width=800
    )
    
    if save:
        fig.write_html('Plot_Number_of_Edges.html')
    
    fig.show()


def Plot_Mean_Rating_Director_Cluster(Director_Success_KNN, save=False):
    # Define the colors for each cluster
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]
    
    # Create the figure
    fig = go.Figure()

    # Add histograms and mean lines for each cluster
    for cluster, color in zip(clusters, colors):
        # Extract data for the cluster
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]['Mean_Rating'].dropna()
        mean_cluster_rating = np.mean(cluster_data)
        
        # Add histogram for cluster data
        fig.add_trace(go.Histogram(
            x=cluster_data,
            name=f'Cluster {cluster}',
            marker_color=color,
            opacity=0.5
        ))
        
        # Add mean line for cluster data
        fig.add_trace(go.Scatter(
            x=[mean_cluster_rating, mean_cluster_rating],
            y=[0, 1],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean'
        ))

    # Update layout
    fig.update_layout(
        title='Mean Rating of Each Director by Cluster',
        xaxis_title='Mean Rating',
        yaxis_title='Frequency (Log Scale)',
        yaxis_type='log',
        barmode='overlay',
        legend=dict(
            title='Legend',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        height=600,
        width=900
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_Mean_Rating_Director_Cluster.html')
    
    # Show the figure
    fig.show()

    return None


def Plot_BOY_Director_Cluster(Director_Success_KNN, save=False):
    # Define the colors for each cluster
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]
    
    # Create the figure
    fig = go.Figure()

    # Add histograms and mean lines for each cluster
    for cluster, color in zip(clusters, colors):
        # Extract data for the cluster
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]['birthYear_director'].dropna()
        mean_cluster_year = np.mean(cluster_data)
        
        # Add histogram for cluster data
        fig.add_trace(go.Histogram(
            x=cluster_data,
            name=f'Cluster {cluster}',
            marker_color=color,
            opacity=0.5
        ))
        
        # Add mean line for cluster data
        fig.add_trace(go.Scatter(
            x=[mean_cluster_year, mean_cluster_year],
            y=[0, 1],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean'
        ))

    # Update layout
    fig.update_layout(
        title='Birth Year of Each Director by Cluster',
        xaxis_title='Birth Year of Director',
        yaxis_title='Frequency (Log Scale)',
        yaxis_type='log',
        barmode='overlay',
        legend=dict(
            title='Legend',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        height=600,
        width=900
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_BOY_Director_Cluster.html')
    
    # Show the figure
    fig.show()

    return None


def Plot_Career_Start_Director_Cluster(Director_Success_KNN, save=False):
    # Define the colors for each cluster
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]
    
    # Create the figure
    fig = go.Figure()

    # Add histograms and mean lines for each cluster
    for cluster, color in zip(clusters, colors):
        # Extract data for the cluster
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]['Career_Start_age'].dropna()
        mean_cluster_start_age = np.mean(cluster_data)
        
        # Add histogram for cluster data
        fig.add_trace(go.Histogram(
            x=cluster_data,
            name=f'Cluster {cluster}',
            marker_color=color,
            opacity=0.5
        ))
        
        # Add mean line for cluster data
        fig.add_trace(go.Scatter(
            x=[mean_cluster_start_age, mean_cluster_start_age],
            y=[0, 1],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean'
        ))

    # Update layout
    fig.update_layout(
        title='Career Start Age for Each Director by Cluster',
        xaxis_title='Career Start Age',
        yaxis_title='Frequency (Log Scale)',
        yaxis_type='log',
        barmode='overlay',
        legend=dict(
            title='Legend',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        height=600,
        width=900
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_Career_Start_Director_Cluster.html')
    
    # Show the figure
    fig.show()

    return None


def Plot_NOF_Director_Cluster(Director_Success_KNN, Max_movie=100, save=False):
    # Define the colors for each cluster
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]
    
    # Create the figure
    fig = go.Figure()

    # Add histograms and mean lines for each cluster
    for cluster, color in zip(clusters, colors):
        # Extract data for the cluster
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]['Number_of_films'].dropna()
        mean_cluster_films = np.mean(cluster_data)
        
        # Add histogram for cluster data
        fig.add_trace(go.Histogram(
            x=cluster_data,
            name=f'Cluster {cluster}',
            marker_color=color,
            opacity=0.5
        ))
        
        # Add mean line for cluster data
        fig.add_trace(go.Scatter(
            x=[mean_cluster_films, mean_cluster_films],
            y=[0, 1],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean'
        ))

    # Update layout
    fig.update_layout(
        title=f'Number of Films for Each Director by Cluster (Max Movies = {Max_movie})',
        xaxis_title='Number of Films',
        yaxis_title='Frequency (Log Scale)',
        yaxis_type='log',
        xaxis=dict(range=[0, Max_movie]),
        barmode='overlay',
        legend=dict(
            title='Legend',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        height=600,
        width=900
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_NOF_Director_Cluster.html')
    
    # Show the figure
    fig.show()

    return None


def Plot_NOF_Vs_Sum_Weight(Director_Success_KNN, save=False):
    # Define the colors for each cluster
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]
    
    # Create the figure
    fig = go.Figure()

    # Add scatter plots for each cluster
    for cluster, color in zip(clusters, colors):
        # Extract data for the cluster
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]
        number_of_films = cluster_data['Number_of_films']
        sum_weight = cluster_data['Sum_Weight']
        
        # Add cluster scatter points
        fig.add_trace(go.Scatter(
            x=number_of_films,
            y=sum_weight,
            mode='markers',
            marker=dict(color=color, size=8, opacity=0.6),
            name=f'Cluster {cluster}'
        ))
        
        # Calculate and plot the mean point for the cluster
        mean_x = np.mean(number_of_films)
        mean_y = np.mean(sum_weight)
        fig.add_trace(go.Scatter(
            x=[mean_x],
            y=[mean_y],
            mode='markers',
            marker=dict(color=color, size=12, symbol='circle', line=dict(color='black', width=2)),
            name=f'Cluster {cluster} Mean'
        ))

    # Update layout
    fig.update_layout(
        title='Number of Films vs. Sum Weight by Cluster',
        xaxis_title='Number of Films (Log Scale)',
        yaxis_title='Sum Weight (Log Scale)',
        xaxis=dict(type='log'),
        yaxis=dict(type='log'),
        legend=dict(
            title='Clusters',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        height=600,
        width=900
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_NOF_Vs_Sum_Weight.html')
    
    # Show the figure
    fig.show()

    return None


def Plot_NOA_Vs_Sum_Weight_Director_Cluster(Director_Success_KNN, save=False):
    # Define cluster colors
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]
    
    # Create the figure
    fig = go.Figure()

    # Add scatter plots for each cluster
    for cluster, color in zip(clusters, colors):
        # Extract data for the cluster
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]
        number_of_actors = cluster_data['Number_of_actors']
        sum_weight = cluster_data['Sum_Weight']
        
        # Add cluster scatter points
        fig.add_trace(go.Scatter(
            x=number_of_actors,
            y=sum_weight,
            mode='markers',
            marker=dict(color=color, size=8, opacity=0.6),
            name=f'Cluster {cluster}'
        ))
        
        # Calculate and plot the mean point for the cluster
        mean_x = np.mean(number_of_actors)
        mean_y = np.mean(sum_weight)
        fig.add_trace(go.Scatter(
            x=[mean_x],
            y=[mean_y],
            mode='markers',
            marker=dict(color=color, size=12, symbol='circle', line=dict(color='black', width=2)),
            name=f'Cluster {cluster} Mean'
        ))

    # Update layout
    fig.update_layout(
        title='Number of Actors vs. Sum Weight by Cluster',
        xaxis_title='Number of Actors (Log Scale)',
        yaxis_title='Sum Weight (Log Scale)',
        xaxis=dict(type='log'),
        yaxis=dict(type='log'),
        legend=dict(
            title='Clusters',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        height=600,
        width=900
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_NOA_Vs_Sum_Weight_Director_Cluster.html')
    
    # Show the figure
    fig.show()

    return None


def Plot_NOF_Vs_Career_Start(Director_Success_KNN, Max_movie=100, save=False):
    # Define cluster colors
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    colors = px.colors.qualitative.Plotly[:len(clusters)]
    
    # Create the figure
    fig = go.Figure()

    # Add scatter plots for each cluster
    for cluster, color in zip(clusters, colors):
        # Extract data for the cluster
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]
        career_start_age = cluster_data['Career_Start_age']
        number_of_films = cluster_data['Number_of_films']
        
        # Add scatter points for the cluster
        fig.add_trace(go.Scatter(
            x=career_start_age,
            y=number_of_films,
            mode='markers',
            marker=dict(color=color, size=8, opacity=0.6),
            name=f'Cluster {cluster}'
        ))
        
        # Calculate and plot the mean point for the cluster
        mean_x = np.mean(career_start_age)
        mean_y = np.mean(number_of_films)
        fig.add_trace(go.Scatter(
            x=[mean_x],
            y=[mean_y],
            mode='markers',
            marker=dict(color=color, size=12, symbol='circle', line=dict(color='black', width=2)),
            name=f'Cluster {cluster} Mean'
        ))

    # Update layout
    fig.update_layout(
        title='Number of Films vs. Career Start Age by Cluster',
        xaxis_title='Career Start Age (Log Scale)',
        yaxis_title='Number of Films (Log Scale)',
        xaxis=dict(type='linear'),
        yaxis=dict(range=[0, Max_movie]),
        legend=dict(
            title='Clusters',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        height=600,
        width=900
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_NOF_Vs_Career_Start.html')
    
    # Show the figure
    fig.show()

    return None