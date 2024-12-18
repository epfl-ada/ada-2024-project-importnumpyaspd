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


def explode_movie_dataset(Movie):
    Movie_copy = Movie.copy()  # Créer une copie du dataset Movie pour éviter la modification du original
    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.replace(r'\s*,\s*', ',', regex=True)
    Movie_copy['Movie_languages'] = Movie_copy['Movie_languages'].str.split(',')
    Movie_copy = Movie_copy.explode('Movie_languages')

    Movie_wo_nan_language = Movie_copy[Movie_copy["Movie_languages"] != ""]

    return Movie_wo_nan_language

def get_most_represented_language(Movie, number_language):
    
    Movie_copy = Movie.copy()  # Créer une copie du dataset Movie pour éviter la modification du original
    Movie_wo_nan_language = explode_movie_dataset(Movie_copy)

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

def custom_create_actor_network(Actors, Movie, min_movies=50, add_attributes=False):
    start_time = time.time()
    Actors_copy = Actors.copy()  # Créer une copie du dataset Actors
    Movie_copy = Movie.copy()  # Créer une copie du dataset Movie

    actors_with_min_x_movies = Actors_copy[Actors_copy['actor_age_atmovierelease'].apply(len) >= min_movies]
    actors_df = actors_with_min_x_movies.explode('Freebase_movie_ID')
    movie_releasedates = Movie_copy.set_index('Freebase_movie_ID')['release_date'].to_dict()

    G = nx.Graph()

    for movie_id, group in tqdm(actors_df.groupby('Freebase_movie_ID'), desc="Creating network"):
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
    end_time = time.time()
    print(f"time to compute: {end_time - start_time:.1f} seconds")
    return G

def create_distribution_for_each_group(dict_group, number_group):
    nb_group = 4
    dict_total = {}

    for g in range(1, nb_group + 1):
        group = dict_group[g]
        total_movie = group.nb_movie.sum()
    
        for i in group.columns:
            if i != "nb_movie":
                ratio = float(group[i].sum())
    
                if i in dict_total:
                    dict_total[i].append(ratio)
                else:
                    dict_total[i] = [ratio]
                    
    return dict_total

def create_group(actor_count_movie_language):
    dict_group = {}
    
    group1 = actor_count_movie_language[actor_count_movie_language["nb_movie"]<=5]
    dict_group[1] = group1
    group2 = actor_count_movie_language[(actor_count_movie_language["nb_movie"]>5) & (actor_count_movie_language["nb_movie"]<11)]
    dict_group[2] = group2
    group3 = actor_count_movie_language[(actor_count_movie_language["nb_movie"]<26) & (actor_count_movie_language["nb_movie"]>10)]
    dict_group[3] = group3
    group4 = actor_count_movie_language[actor_count_movie_language["nb_movie"]>25]
    dict_group[4] = group4
    
    print(f"Size of groups : {group1.shape[0],group2.shape[0],group3.shape[0],group4.shape[0]}")

    return dict_group

def print_result_chi2(chi2, p_value, dof, expected):
    # Résultats
    print("Chi-2 :", chi2)
    print("P-value :", p_value)
    print("degree of freedom :", dof)

    if p_value < 0.05:
        print("Distribution are significantly statistically different (p < 0.05).")
    else:
        print("No significance statistical differences (p >= 0.05).")

def plot_language_histograms(result, save=False):
    """
    Affiche des histogrammes interactifs pour chaque entrée de la variable `result` avec Plotly
    et enregistre chaque graphique sous forme de fichier HTML.
    
    Paramètres :
        result : dict
            Dictionnaire contenant comme clés les catégories (ex: les films par langues),
            et comme valeurs des DataFrames avec une colonne 'sum' représentant la quantité.
    """
    for key, value in result.items():
        data = value.copy()  # Créer une copie pour éviter les modifications
        total = float(data.loc[key]["sum"])
        data["sum"] = data["sum"] / total  # Calcul des proportions

        # Extraire la langue sans "Language" (ex: "English Language" -> "English")
        language_name = key.split(" ")[0]  # Diviser la clé et prendre la première partie

        # Création de l'histogramme interactif
        fig = go.Figure()

        languages = [lang.split(" ")[0] for lang in data.index]

        fig.add_trace(
            go.Bar(
                x=languages,   # Noms des langues
                y=data["sum"],  # Proportions
                marker=dict(
                    color=data["sum"], colorscale="Viridis", showscale=True  # Couleurs dynamiques
                ),
                name=language_name  # Utiliser seulement la langue
            )
        )

        # Personnalisation du layout
        fig.update_layout(
            title={
                'text': f"Other languages for actor in {language_name} films",
            },
            xaxis_title="Languages",
            yaxis_title="Actor [%]",
            xaxis=dict(
                tickangle=0,  # Tick horizontal
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                tickfont=dict(size=12),
                tickformat=".0%"  # Format des pourcentages
            ),
            template="plotly_white",
            showlegend=False,
            height=700,  # Ajuster la hauteur pour éviter l'effet écrasé
            width=950   # Ajuster la largeur pour un ratio agréable
        )

        # Afficher la figure
        fig.show()

        # Sauvegarder la figure si demandé
        if save:
            filename = f"histogram_{language_name}.html"
            fig.write_html(filename)
            print(f"Figure saved as {filename}") 

def plot_group_distribution_language(df, nb_group, list_languages, save=False, file_name="group_distribution_plot.html"):
    """
    Crée un graphique de comparaison des distributions de valeurs par groupes et langues.
    
    Paramètres :
    - df : DataFrame (matrice contenant les données des groupes)
    - nb_group : int (nombre de groupes à comparer)
    - list_languages : list (liste des langues)
    - save : bool (détermine si le graphique doit être sauvegardé au format HTML)
    - file_name : str (nom du fichier pour l'enregistrement, par défaut "group_distribution_plot.html")
    
    Retour :
    - Affiche un graphique interactif Plotly et optionnellement enregistre le fichier en HTML
    """
    
    # Matrice des données
    matrix = df.copy().T
    
    # Créer une liste pour stocker les traces de chaque groupe
    traces = []
    
    # Remplir dynamiquement les traces pour chaque groupe
    for i in range(nb_group):
        # Calcul des proportions pour chaque groupe
        group_values = matrix.iloc[i, :].values / np.sum(matrix.iloc[i, :].values)
        
        # Ajouter une trace (un groupe) au graphique
        trace = go.Bar(
            x=list_languages,
            y=group_values,
            name=f'Group {i + 1}',  # Nom du groupe
            hoverinfo='x+y+name',  # Information à afficher lors du survol de la souris
            marker=dict(colorscale="Viridis")  # Ajout du coloriage par couleurscale
        )
        traces.append(trace)
    
    # Création de la figure avec Plotly
    fig = go.Figure(data=traces)
    
    # Mettre à jour les titres et les axes
    fig.update_layout(
        title='Distribution des langues pour chaque groupe',  # Reformulation du titre
        xaxis_title='Langue',
        yaxis_title='Proportion [%]',
        barmode='group',  # Décalage des barres pour comparer les groupes côte à côte
        height=700,
        width=950,
        template='plotly_white'  # Utilisation d'un thème léger
    )
    
    # Si le paramètre 'save' est True, enregistrer le graphique au format HTML
    if save:
        fig.write_html(file_name)
        print(f"Le graphique a été enregistré sous le nom : {file_name}")
    
    # Afficher le graphique
    fig.show()

def plot_network_with_language(G):
    langue_color_mapping = {
    'English Language': '#440154',  # Viridis - Dark purple
    'French Language': '#46327e',   # Viridis - Purple
    'Hindi Language': '#365c8d',    # Viridis - Blue-purple
    'Spanish Language': '#277f8e',  # Viridis - Teal
    'Italian Language': '#1fa187',  # Viridis - Green-teal
    'German Language': '#4ac16d',   # Viridis - Green
    'Silent film': '#a0da39',       # Viridis - Yellow-green
    'Japanese Language': '#fde725', # Viridis - Yellow
    }
    default_color = (0.5, 0.5, 0.5, 0.5)
    edge_colors = [langue_color_mapping.get(G[u][v]["langue"], default_color) for u, v in G.edges]
    
    legend_elements = [mlines.Line2D([], [], color=color, marker='o', markersize=10, linestyle='', label=langue) for langue, color in langue_color_mapping.items()]
    legend_elements.append(mlines.Line2D([], [], color=default_color, marker='o', markersize=10, linestyle='', label='Unknown Language'))
    start_time = time.time()
    sp = nx.spring_layout(G, k=0.2, seed=42)
    plt.figure(figsize=(15, 15))
    nx.draw_networkx(G, pos=sp, with_labels=False, node_size=0.5, node_color="k", width=0.05)
    nx.draw_networkx_edges(G, pos=sp, width=0.5, edge_color=edge_colors, style='solid')
    plt.legend(handles=legend_elements,loc="upper right", fontsize=10, title_fontsize=12) 
    # plt.axes('off')
    plt.title("Actor network clustered with louvain method", fontsize=15)
    plt.show()
    end_time = time.time()
    print(f"time to compute: {end_time - start_time:.1f} seconds")