### This .py contains all functions used for plotting in all part. There are delimitations between the parts with full lines of  "#########".

# Import librairies
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import time
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from itertools import combinations
import community
import collections
import scipy.stats
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# import function from director-actor link part
from src.utils.Director_Actor import *


#############################################################################################################################################
# PART 1 : ACTOR CAREER PROFILE
#############################################################################################################################################
def plot_elbow_method(Career_dataset,cluster_range,random_state = 0):
    """
    plot elbow to choose suitable number of cluster for Kmean
    """
    sse = []

    for n_clusters in tqdm(cluster_range):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(Career_dataset)
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

    fig.show()

def plot_career_data(Career_dataset, labels, actor_name=None, n_clusters=5, alpha=0.6, save = False):
    """
    Plot the number of films through their career
    if a label is provided, plot a curves for each cluster.
    """
    # Columns to drop
    columns_to_drop = [
        'Freebase_actor_ID', 'actor_name', 'actor_DOB', 'actor_gender',
        'actor_height', 'ethnicity', 'Freebase_movie_ID', 'actor_age_atmovierelease',
        'Career_Start_age', 'Career_End_age', 'Career_length', 'Total_number_of_films', 'Labels'
    ]
        # Generate the viridis colors with transparency for clusters
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [mcolors.rgb2hex(c[:3]) + hex(int(alpha * 255))[2:].zfill(2) for c in viridis]

    # Convert hex colors to rgba format (e.g. #44015499 -> rgba(68, 1, 84, 0.6))
    rgba_colors = []
    for color in colors:
        r, g, b = mcolors.hex2color(color)
        rgba_color = f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {alpha})'
        rgba_colors.append(rgba_color)
    Career_dataset_cleaned = Career_dataset.drop(columns=columns_to_drop, errors='ignore')
    Career_dataset_cleaned.columns = range(len(Career_dataset_cleaned.columns))


    fig = go.Figure()

    if actor_name:
        # Plot only for a specific actor (select the row)
        actor_data = Career_dataset[Career_dataset['actor_name'] == actor_name]

        if actor_data.empty:
            # If actor isn't found print an error
            print("This actor isn't in our database!")
            return

        # Drop columns and clean data
        actor_data = actor_data.drop(columns=columns_to_drop, errors='ignore').transpose()
        actor_data.reset_index(inplace=True)
        actor_data.columns = ['Career Year', 'Number of Films']
        actor_data['Career Year'] = range(len(actor_data))

        fig.add_trace(go.Scatter(
            x=actor_data['Career Year'],
            y=actor_data['Number of Films'],
            mode='lines+markers',
            name=actor_name,
            line=dict(color=rgba_colors[0])
        ))

        fig.update_layout(
            autosize=True,
            title=f"{actor_name}'s Career",
            xaxis_title="Career Year",
            yaxis_title="Number of Movies"
        )
    else:

        # Centroids computation and plotting
        for i in range(n_clusters):
            cluster_data = Career_dataset_cleaned.iloc[labels == i]
            centroid = cluster_data.mean(axis=0)

            fig.add_trace(go.Scatter(
                x=centroid.index,
                y=centroid.values,
                mode='markers+lines',
                name=f'Cluster {i}',
                line=dict(color=rgba_colors[i])  # Set the color for the cluster
            ))

        fig.update_layout(
            autosize=True,
            title="Centroids Actor Cluster",
            xaxis_title="Year of Career",
            yaxis_title="Number of Movies"
        )

    # Show the plot
    fig.show()
    
    if save and actor_name!=None:
        filename = f"career_{actor_name}.html"
        fig.write_html(filename)
    elif save and actor_name==None:
        filename = f"career_with_{n_clusters}_cluster.html"
        fig.write_html(filename)

def plot_gender_proportions_by_cluster(dataset, n_clusters=2, alpha=0.55, save=False):
    """
    Plot proportion of gender in clusters. The plotly plot shows % and number of actor
    """
    label_column = 'Labels'
    gender_column = 'actor_gender'
    
    gender_counts = dataset.groupby([label_column, gender_column]).size().reset_index(name='count')
    cluster_totals = dataset.groupby(label_column).size().reset_index(name='total_count')
    gender_proportions = gender_counts.merge(cluster_totals, on=label_column)
    gender_proportions['proportion'] = gender_proportions['count'] / gender_proportions['total_count']
    
    clusters = gender_proportions[label_column].unique()

    # Generate custom colors using the viridis colormap
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [mcolors.rgb2hex(c[:3]) + hex(int(alpha * 255))[2:].zfill(2) for c in viridis]  # Add alpha

    # Create a subplot with one row and as many columns as clusters
    fig = sp.make_subplots(
        rows=1,
        cols=len(clusters),
        specs=[[{"type": "domain"} for _ in clusters]],
        subplot_titles=[f"Cluster {cluster}" for cluster in clusters]
    )
    
    # Add pie charts for each cluster
    for idx, cluster in enumerate(clusters):
        cluster_data = gender_proportions[gender_proportions[label_column] == cluster]
        labels = cluster_data[gender_column].values
        proportions = cluster_data['proportion'].values
        counts = cluster_data['count'].values

        # Map gender labels to colors from viridis with transparency
        custom_colors = [colors[i % n_clusters] for i in range(len(labels))]  # Cycle through available colors

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=proportions,
                hole=0.3,
                marker=dict(colors=custom_colors),
                hovertemplate=(
                    "<b>%{label}</b><br>"  # Gender label
                    "Count: %{customdata[0]}<br>"  # Raw count
                    "Percentage: %{percent:.2%}<extra></extra>"  # Percentage
                ),
                customdata=[(count,) for count in counts],  # Provide the count as custom data
            ),
            row=1,
            col=idx + 1
        )
    
    # Update layout
    fig.update_layout(
        autosize=True,
        title_text="Gender Proportions Across Clusters",
        title_x=0.5,
        showlegend=False,
        template="plotly_white",
    )
    
    # Show the combined plot
    fig.show()
    
    if save:
        filename = f"gender_camembert_cluster.html"
        fig.write_html(filename)

def plot_cluster_histogram(Actor_career, column_name, n_clusters=3, bin_width=1, 
                                  max_value=None, min_value=None, kde_option=False, 
                                  logscale=False, Pourcentage=False, alpha=0.3, save=False):
    """
    Plot a histogram for a given column, one histogram per cluster
    """
    labels_column = 'Labels'
    
    # Compute range for the axis
    if min_value is None:
        min_value = Actor_career[column_name].min()
    if max_value is None:
        max_value = Actor_career[column_name].max()
    
    # Generate Viridis colors with transparency
    alpha = 0.6
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Convert colors to rgba format (add alpha channel for transparency)
    colors = [
        f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, {alpha})'
        for c in viridis
    ]
    
    # Create histogram data
    fig = go.Figure()
    bins = np.arange(min_value, max_value + bin_width, bin_width)
    
    for i in range(n_clusters):
        cluster_data = Actor_career[Actor_career[labels_column] == i]
        hist_data = cluster_data[column_name]
        
        if Pourcentage:
            hist_stat = 'percent'
        else:
            hist_stat = 'count'
        
        fig.add_trace(
            go.Histogram(
                x=hist_data,
                xbins=dict(start=min_value, end=max_value, size=bin_width),
                name=f"Cluster {i}",
                marker_color=colors[i],
                opacity=alpha,
                histnorm=hist_stat if Pourcentage else None
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Distribution of {column_name.replace('_', ' ').title()}",
        xaxis_title=column_name.replace('_', ' ').title(),
        yaxis_title="Percentage" if Pourcentage else "Count",
        barmode='overlay',
        legend=dict(title="Clusters"),
        xaxis=dict(range=[min_value, max_value]),
        yaxis=dict(type='log' if logscale else 'linear')
    )

    # Save the plot
    if save:
        fig.write_html(f"{column_name}histo.html")

    fig.show()

#############################################################################################################################################
# PART 2 : LINK SUCCESSFUL ACTORS AND PRODUCTER
#############################################################################################################################################

def Elbow_method_genre(Movie,save=False):
    """
    elbow method for determining number of cluster
    """
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

def Plot_Camembert_Kmeans(Director_Success_Kmeans, save=False):
    """
    plot a camembert that display the distribution of clusters
    """
    # Computing data
    df = Director_Success_Kmeans['Cluster_Label'].value_counts().reset_index()
    df.columns = ['Label', 'Count']

    # Number of clusters
    n_clusters = len(df)

    # Get Viridis colors for each cluster
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)' for c in viridis]

    # Creation of the plot
    fig = go.Figure(
        data=[go.Pie(
            labels=df['Label'],
            values=df['Count'],
            textinfo="percent",
            hoverinfo="label+value",
            insidetextorientation="auto",
            marker=dict(colors=colors)  # Apply colors from Viridis
        )]
    )

    fig.update_layout(
        title=dict(
            text="Distribution of Clusters",
            x=0.5,  # Center the title horizontally
            y=0.95,  # Slightly lower than the default top position
            xanchor="center",
            yanchor="top"
        ),
        autosize=True,
        width=None,
        height=None,
        plot_bgcolor='white'  # White background for better clarity
    )
    
    # Show and save the plot
    if save:
        fig.write_html('Plot_Camembert_Kmeans.html')    
    fig.show()

    return None

def Plot_Subgraph_Directed_Graph(Di_Graph,Director,director_name='Ingmar Bergman',save=False):
    """
    plot a directed graph for a given graph
    """
    # determine the ID and the node based on the director name
    for node, attribute in Di_Graph.nodes(data=True):
        if attribute.get('name') == director_name:
            director_node = node
            break
    _, nodes_director = nx.bipartite.sets(Di_Graph) 
    
    # Compute the actor connected to the director 
    Actors_nodes = list(Di_Graph.predecessors(director_node))  
    nodes = [director_node] + Actors_nodes
    subgraph = Di_Graph.subgraph(nodes)
    
    pos_sub = nx.bipartite_layout(subgraph, Actors_nodes)
    
    # Draw the graph
    plt.figure(figsize=(8, 6))
    labels_node = {n: subgraph.nodes[n]['name'] for n in subgraph.nodes}
    #labels_node = {n: subgraph.nodes[n]['weight'] if n == director_node else '' for n in subgraph.nodes}
    labels_edge = {e: subgraph.edges[e]['weight'] for e in subgraph.edges}
    nx.draw(subgraph, pos_sub,with_labels=True, node_color=['skyblue' if node in Actors_nodes else 'lightgreen' for node in subgraph.nodes()],  
            arrowsize=20, edge_color='gray', node_size=800, font_size=10, 
            labels=labels_node
           )
    plt.title("Example of a node from the Bipartite Directed Network", fontsize=16)
    nx.draw_networkx_edge_labels(subgraph, pos_sub, edge_labels=labels_edge)
    if save:
        plt.savefig(f'Plot_Subgraph_Directed_Graph{director_name}.png', transparent=True)
    plt.show()

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

def Plot_NOF_Actor(Actor,max_movie=200,describe=False,save=False):
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

def plot_Success_actor_network(G, k=0.2, seed=42, enable_partition=False):
    """
    Plot the actor network using spring layout and optionally cluster with the Louvain method.
    
    """
    start_time = time.time()

    values = None
    if enable_partition:
        partition = community.best_partition(G, random_state=42)
        values = [partition.get(node) for node in G.nodes()]
        counter = collections.Counter(values)
        print(f"Partition summary: {counter}")

    sp = nx.spring_layout(G, k=k, seed=seed)
    plt.figure(figsize=(15, 15))
    nx.draw_networkx(
        G,
        pos=sp,
        with_labels=False,
        node_size=5,
        node_color=values if values is not None else '#1f78b4',
        width=0.05,
    )
    plt.title("Actor network clustered with Louvain method" if enable_partition else "Successfull Actor network", fontsize=15)
    plt.savefig('Alex_Plot_Graph_Random.png')
    plt.show()

    end_time = time.time()
    print(f"Time to compute: {end_time - start_time:.1f} seconds")


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
    """
    Plot the total weight of each director grouped by clusters using the Viridis colormap.
    """

    # Calculate the overall mean and median
    mean_Sum_Weight = np.mean(Director_Success_KNN['Sum_Weight'].dropna())
    median_Sum_Weight = np.median(Director_Success_KNN['Sum_Weight'].dropna())

    # Define the unique cluster labels and colors using Viridis
    cluster_labels = sorted(Director_Success_KNN['Cluster_Label'].unique())
    n_clusters = len(cluster_labels)
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [
        f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)'
        for c in viridis
    ]

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
            y=[0, 1000],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean'
        ))

    # Add overall mean and median lines
    fig.add_trace(go.Scatter(
        x=[mean_Sum_Weight, mean_Sum_Weight],
        y=[0, 1000],
        mode='lines',
        line=dict(color='black', width=2),
        name='Overall Mean'
    ))
    fig.add_trace(go.Scatter(
        x=[median_Sum_Weight, median_Sum_Weight],
        y=[0, 1000],
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
        ),
        autosize=True,
        width=None,
        height=None
    )
    
    # Save the figure if needed
    if save:
        fig.write_html('Plot_Weight_Director.html')
    
    # Show the figure
    fig.show()


def Plot_Edge_Weight_Distribution(Director_Success_KNN, save=False):
    """
    Plot the weight distribution of edges for directors, using Viridis colormap.
    """

    # Creating a list containing all the weights of the edges
    all_weight = [weight for weight_list in Director_Success_KNN['Weight'] for weight in weight_list]

    # Calculate the mean and median of the edge weights
    mean_Sum_Weight = np.mean(all_weight)
    median_Sum_Weight = np.median(all_weight)

    # Define cluster colors using Viridis colormap
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    n_clusters = len(clusters)
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [
        f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)'
        for c in viridis
    ]

    # Create the figure
    fig = go.Figure()

    # Add histograms and mean lines for each cluster
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
            y=[0, 20000],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean (Weight)'
        ))

    # Add overall mean and median lines
    fig.add_trace(go.Scatter(
        x=[mean_Sum_Weight, mean_Sum_Weight],
        y=[0, 20000],
        mode='lines',
        line=dict(color='black', width=2),
        name='Overall Mean (Weight)'
    ))
    fig.add_trace(go.Scatter(
        x=[median_Sum_Weight, median_Sum_Weight],
        y=[0, 20000],
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
        barmode='overlay',
        showlegend=True,
        autosize=True,
        width=None,
        height=None,
        legend=dict(
            title="Clusters",
            x=1.05,
            y=1,
            font=dict(size=10)
        )
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_Edge_Weight_Distribution.html')

    # Show the figure
    fig.show()


def Plot_Number_of_Edges(Director_Success_KNN, save=False):
    """
    Plot the distribution of the number of edges (actors) for directors, using Viridis colormap.
    """
    # Calculate the overall mean and median of the number of actors
    mean_Number_of_Actor = np.mean(Director_Success_KNN['Number_of_actors'])
    median_Number_of_Actor = np.median(Director_Success_KNN['Number_of_actors'])

    # Define cluster colors using Viridis colormap
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    n_clusters = len(clusters)
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [
        f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)'
        for c in viridis
    ]

    # Create the figure
    fig = go.Figure()

    # Add histograms and mean lines for each cluster
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
            y=[0, 1000],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean (Actors)'
        ))

    # Add overall mean and median lines
    fig.add_trace(go.Scatter(
        x=[mean_Number_of_Actor, mean_Number_of_Actor],
        y=[0, 1000],
        mode='lines',
        line=dict(color='black', width=2),
        name='Overall Mean (Actors)'
    ))
    fig.add_trace(go.Scatter(
        x=[median_Number_of_Actor, median_Number_of_Actor],
        y=[0, 1000],
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
        barmode='overlay',
        legend=dict(
            title="Clusters",
            x=1.05,
            y=1,
            font=dict(size=10)
        ),
                autosize=True,
        width=None,
        height=None
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_Number_of_Edges.html')

    # Show the figure
    fig.show()

def Plot_Mean_Rating_Director_Cluster(Director_Success_KNN, save=False):
    """
    Plot the distribution of mean ratings by director, grouped by cluster, using the Viridis colormap.
    """

    # Define cluster labels and colors using Viridis colormap
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    n_clusters = len(clusters)
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [
        f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)'
        for c in viridis
    ]

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
            y=[0, 200],
            mode='lines',
            line=dict(color=color, dash='dash'),
            name=f'Cluster {cluster} Mean'
        ))

    # Add overall mean and median lines
    overall_mean = np.mean(Director_Success_KNN['Mean_Rating'].dropna())
    overall_median = np.median(Director_Success_KNN['Mean_Rating'].dropna())

    fig.add_trace(go.Scatter(
        x=[overall_mean, overall_mean],
        y=[0, 200],
        mode='lines',
        line=dict(color='black', width=2),
        name='Overall Mean'
    ))
    fig.add_trace(go.Scatter(
        x=[overall_median, overall_median],
        y=[0, 200],
        mode='lines',
        line=dict(color='purple', width=2),
        name='Overall Median'
    ))

    # Update layout
    fig.update_layout(
        title='Mean Rating of Each Director by Cluster',
        xaxis_title='Mean Rating',
        yaxis_title='Frequency (Log Scale)',
        yaxis_type='log',
        barmode='overlay',
        legend=dict(
            title='Clusters',
            x=1.05,
            y=1,
            font=dict(size=10)
        ),
                autosize=True,
        width=None,
        height=None
    )

    # Save the figure if requested
    if save:
        fig.write_html('Plot_Mean_Rating_Director_Cluster.html')
    
    # Show the plot
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
                autosize=True,
        width=None,
        height=None
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
        autosize=True,
        width=None,
        height=None
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
            y=[0, 1200],
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
        autosize=True,
        width=None,
        height=None
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_NOF_Director_Cluster.html')
    
    # Show the figure
    fig.show()

    return None


def Plot_NOF_Vs_Sum_Weight(Director_Success_KNN, save=False):
    """
    Scatter plot for Number of Films vs. Sum Weight by Cluster using Viridis colormap.
    """

    # Define cluster labels and colors using Viridis colormap
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    n_clusters = len(clusters)
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [
        f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)'
        for c in viridis
    ]

    # Create the figure
    fig = go.Figure()

    # Add scatter plots for each cluster
    for cluster, color in zip(clusters, colors):
        # Extract data for the cluster
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]
        number_of_films = cluster_data['Number_of_films']
        sum_weight = cluster_data['Sum_Weight']

        # Add scatter points for the cluster
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

    # Add gridlines and logarithmic axes
    fig.update_layout(
        title='Number of Films vs. Sum Weight by Cluster',
        xaxis_title='Number of Films (Log Scale)',
        yaxis_title='Sum Weight (Log Scale)',
        xaxis=dict(type='log', gridcolor='lightgrey'),
        yaxis=dict(type='log', gridcolor='lightgrey'),
        legend=dict(
            title='Clusters',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        autosize=True,
        width=None,
        height=None,
        plot_bgcolor='white'
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_NOF_Vs_Sum_Weight.html')
    
    # Show the plot
    fig.show()

    return None


def Plot_NOA_Vs_Sum_Weight_Director_Cluster(Director_Success_KNN, save=False):
    """
    Scatter plot for Number of Actors vs. Sum Weight by Cluster using Viridis colormap.
    """

    # Define cluster labels and colors using Viridis colormap
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    n_clusters = len(clusters)
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [
        f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)'
        for c in viridis
    ]

    # Create the figure
    fig = go.Figure()

    # Add scatter plots for each cluster
    for cluster, color in zip(clusters, colors):
        # Extract data for the cluster
        cluster_data = Director_Success_KNN[Director_Success_KNN['Cluster_Label'] == cluster]
        number_of_actors = cluster_data['Number_of_actors']
        sum_weight = cluster_data['Sum_Weight']

        # Add scatter points for the cluster
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

    # Add gridlines and logarithmic axes
    fig.update_layout(
        title='Number of Actors vs. Sum Weight by Cluster',
        xaxis_title='Number of Actors (Log Scale)',
        yaxis_title='Sum Weight (Log Scale)',
        xaxis=dict(type='log', gridcolor='lightgrey'),
        yaxis=dict(type='log', gridcolor='lightgrey'),
        legend=dict(
            title='Clusters',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        autosize=True,
        width=None,
        height=None,
        plot_bgcolor='white'
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_NOA_Vs_Sum_Weight_Director_Cluster.html')
    
    # Show the plot
    fig.show()

    return None


def Plot_NOF_Vs_Career_Start(Director_Success_KNN, Max_movie=100, save=False):
    """
    Scatter plot for Number of Films vs. Career Start Age by Cluster using Viridis colormap.
    """

    # Define cluster labels and colors using Viridis colormap
    clusters = sorted(Director_Success_KNN['Cluster_Label'].unique())
    n_clusters = len(clusters)
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [
        f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, 0.6)'
        for c in viridis
    ]

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
        xaxis_title='Career Start Age',
        yaxis_title='Number of Films (Log Scale)',
        xaxis=dict(type='linear'),
        yaxis=dict(range=[0, Max_movie]),
        legend=dict(
            title='Clusters',
            font=dict(size=10),
            x=1.05,
            y=1
        ),
        autosize=True,
        width=None,
        height=None,
        plot_bgcolor='white'
    )

    # Save the figure if needed
    if save:
        fig.write_html('Plot_NOF_Vs_Career_Start.html')
    
    # Show the figure
    fig.show()

    return None

#############################################################################################################################################
# PART 3 : ACTOR NETWORKING THROUGH LANGUAGES
#############################################################################################################################################

def plot_language_histograms(result, save=False, alpha=0.6):
    """
    plot histograms for each entry in the result dictionary
    """
    for key, value in result.items():
        # Create a copy of the data to avoid modifying the original
        data = value.copy()

        # Get the total number of actors that have played in this language
        total = float(data.loc[key]["sum"])

        # Compute ratios for the plot for better visualization
        data["sum"] = data["sum"] / total

        # Extract the language name for the tick labels
        language_name = key.split(" ")[0]

        # Create a new figure
        fig = go.Figure()

        # Prepare language labels without "Language" suffix
        languages = [lang.split(" ")[0] for lang in data.index]

        # Define colors via the Viridis color scale, and add transparency
        viridis = plt.cm.viridis(np.linspace(0, 1, len(data)))

        # Generate a list of colors with the alpha applied for rgba format
        colors_with_alpha = [
            f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, {alpha})'
            for c in viridis
        ]

        # Add a bar trace with dynamic coloring based on proportions and transparent colors
        fig.add_trace(
            go.Bar(
                x=languages,
                y=data["sum"],
                marker=dict(
                    color=colors_with_alpha,  # Use the colors with transparency
                    showscale=False  # Disable the automatic colorbar
                )
            )
        )

        # Customize the layout of the plot
        fig.update_layout(
            title={'text': f"Other languages for actor in {language_name} films"},
            xaxis_title="Languages",
            yaxis_title="Actor [%]",
            xaxis=dict(
                tickangle=0,
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                tickfont=dict(size=12),
                tickformat=".0%"
            ),
            template="plotly_white",
            showlegend=False,  # Remove legend if you don't need it
            autosize=True,
            width=None,
            height=None,
        )

        # Show the figure
        fig.show()

        # Save the figure as an HTML file if 'save' is True
        if save:
            filename = f"histogram{language_name}.html"
            fig.write_html(filename)

def plot_group_distribution_language(df, nb_group, list_languages,  dict_actor_language,save=False):
    """
    Plot the distribtuion of languages movie / total person in its group for a given list of languages
    """
    # Create a copy of the data matrix and transpose it
    matrix = df.copy().T
    
    # Create a list to store the traces for each group
    traces = []
    
    # Generate Viridis colors with transparency
    alpha = 0.6
    viridis = plt.cm.viridis(np.linspace(0, 1, len(list_languages)))
    
    # Convert colors to rgba format (add alpha channel for transparency)
    viridis_colors = [
        f'rgba({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)}, {alpha})'
        for c in viridis
    ]
    
    # Loop over each group and create a bar trace for each
    for i in range(nb_group):
        group_values = matrix.iloc[i, :].values / dict_actor_language[i].shape[0]  # Compute proportions for each group
        languages = [lang.split(" ")[0] for lang in list_languages]
        
        # Create the trace for the bar chart
        trace = go.Bar(
            x=languages,
            y=group_values,
            name=f'Group {i}',  # Group name
            hoverinfo='x+y+name',  # Hover info to show during the mouse-over
            marker=dict(
                color=viridis_colors[i],  # Apply group color from the Viridis palette
                showscale=False            # Hide the scale in this case
            )
        )
        
        # Append the trace to the list of traces
        traces.append(trace)

    # Create the figure
    fig = go.Figure(data=traces)

    # Update the layout of the figure
    fig.update_layout(
        title='Distribution of language per group',
        xaxis_title='Language',
        yaxis_title='Proportion [%]',
        barmode='group',  # Group bars side by side
        template='plotly_white',
        autosize=True,
        width=None,
        height=None,
    )

    # Save the figure if requested
    if save:
        file_name="group_distribution_plot.html"
        fig.write_html(file_name)
        
    # Show the plot
    fig.show()

def plot_network_with_language(G, save=False):
    """
    plot the network and indentify the languages ( top 6 languages + others in one last gorup)
    """
    
    languages = ['English Language', 'French Language', 'Hindi Language', 'Spanish Language', 'Italian Language', 'German Language']
    alpha = 0.6
    viridis = plt.cm.viridis(np.linspace(0, 1, len(languages)))
    viridis_colors = [mcolors.rgb2hex(c[:3]) + hex(int(alpha * 255))[2:].zfill(2) for c in viridis]
    
    # map language to color
    langue_color_mapping = {languages[i]: viridis_colors[i] for i in range(len(languages))}

    # Default color for languages not in the mapping
    default_color = (0.1, 0.1, 0.1, 0.1)  # Semi-transparent black for "Other Languages"
    
    # Assign edge colors based on the languages from the edge attributes
    edge_colors = [langue_color_mapping.get(G[u][v]["langue"], default_color) for u, v in G.edges]

    # Create a legend with the specified colors
    legend_elements = [
        mlines.Line2D([], [], color=color, marker='o', markersize=10, linestyle='', label=langue) 
        for langue, color in langue_color_mapping.items()
    ]
    legend_elements.append(mlines.Line2D([], [], color=default_color,  markersize=10, linestyle='-', label='Other Languages'))

    start_time = time.time()
    
    # Compute layout for the network (spring layout for positioning nodes)
    sp = nx.spring_layout(G, k=0.2, seed=42)
    
    plt.figure(figsize=(15, 15))
    
    # Draw the network without labels, but including the edges colored based on language
    nx.draw_networkx(G, pos=sp, with_labels=False, node_size=0.5, node_color="k", width=0.025)
    nx.draw_networkx_edges(G, pos=sp, width=0.5, edge_color=edge_colors, style='solid')

    # Add the legend
    plt.legend(handles=legend_elements, loc="upper right", fontsize=15, title_fontsize=12)

    # Set plot title
    plt.title("Network between actors: languages identification", fontsize=20)

    # Save plot as an image if the save parameter is True
    if save:
        plt.savefig("network_with_languages.png", format="PNG", bbox_inches='tight', transparent=True)

    # Show the plot
    plt.show()

    end_time = time.time()
    print(f"time to compute: {end_time - start_time:.1f} seconds")