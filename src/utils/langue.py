import networkx as nx
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import plotly.graph_objects as go
import time
import matplotlib.lines as mlines
import matplotlib.colors as mcolors


def explode_movie_dataset(Movie):
    """
    Explodes the 'Movie_languages' column of the dataset into multiple rows 
    for each language, removing any empty or NaN values.
    """
    
    Movie_copy = Movie.copy()
    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.replace(r'\s*,\s*', ',', regex=True)
    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.split(',')# split the 'Movie_languages' into a list
    Movie_copy = Movie_copy.explode('Movie_languages') #explode dataset

    Movie_wo_nan_language = Movie_copy[Movie_copy["Movie_languages"] != ""]  # Remove rows with empty language

    return Movie_wo_nan_language

def get_most_represented_language(Movie, number_language):
    """
    Gets the most represented languages in the dataset.
    """
    
    Movie_copy = Movie.copy()
    Movie_wo_nan_language = explode_movie_dataset(Movie_copy) #explode dataset

    #get most frequent languages
    language_to_keep = Movie_wo_nan_language["Movie_languages"].value_counts().head(number_language).index

    return language_to_keep

def create_actor_language_dataset(Movie, Actor, number_language):
    """
    Creates a dataset of actor with languages movie they played in.
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
     creat a dict per languages kept. each value are df only with actor that have played in this specific language (key)
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
    creat network with connection : play together between actor
    identify the language of moive in the edge of the network
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

    # Initialize an empty undirected graph using NetworkX
    G = nx.Graph()

    # Iterate over each movie and its actors
    for movie_id, group in tqdm(actors_df.groupby('Freebase_movie_ID'), desc="Creating network"):
        actor_ids = group['Freebase_actor_ID'].tolist()  # Get the list of actor IDs for this movie
        
        # Iterate over each pair of actors who acted in the same movie
        for actor1, actor2 in combinations(actor_ids, 2):
            if actor1 != actor2:
                if G.has_edge(actor1, actor2):
                    # If the edge exists, increment the weight by 1 (co-appearance)
                    G[actor1][actor2]['weight'] += 1
                else:
                    # Determine the language of the movie
                    movie_language = Movie_copy.loc[Movie_copy['Freebase_movie_ID'] == movie_id, 'Movie_languages']
                    if not movie_language.empty:
                        movie_language = movie_language.iloc[0]
                        if isinstance(movie_language, list):
                            movie_language = movie_language[0].strip()  # Take the first language if it's a list
                        else:
                            # If not a list, split by comma and take the first language (reason : can't represent both color, there is less than 10% woith many language, it won't really change the gloabl aspect pf the plot)
                            movie_language = movie_language.split(',')[0].strip()
                    else:
                        movie_language = "Unknown"
                    if movie_language == "":
                        movie_language = "Unknown"
                    
                    # Add a new edge between the two actors with weight 1 and the movie language
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

def create_group(actor_count_movie_language):
    """
    create specific group describewd in notebook.
    """
    dict_group = {}
    
    group1 = actor_count_movie_language[actor_count_movie_language["nb_movie"]<=5]
    dict_group[1] = group1
    group2 = actor_count_movie_language[(actor_count_movie_language["nb_movie"]>5) & (actor_count_movie_language["nb_movie"]<11)]
    dict_group[2] = group2
    group3 = actor_count_movie_language[(actor_count_movie_language["nb_movie"]<26) & (actor_count_movie_language["nb_movie"]>10)]
    dict_group[3] = group3
    group4 = actor_count_movie_language[actor_count_movie_language["nb_movie"]>25]
    dict_group[4] = group4

    #check that the nb of actor per group is not that small
    print(f"Size of groups : {group1.shape[0],group2.shape[0],group3.shape[0],group4.shape[0]}")

    return dict_group

def create_group_success(actor_count_movie_language):
    """
    create specific group describewd in notebook.
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

def plot_language_histograms(result, save=False, alpha=0.6):
    """
    Plots interactive histograms for each entry in the result dictionary and saves them.
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
    
    # Generate viridis color
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