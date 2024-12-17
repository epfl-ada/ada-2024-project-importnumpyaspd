import networkx as nx
import pandas as pd
from itertools import combinations
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so

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
    """
    Affiche des histogrammes pour chaque entrée de la variable `result`.
    Paramètres :
        result : dict
            Dictionnaire contenant comme clés les catégories (ex: les films par langues),
            et comme valeurs des DataFrames avec une colonne 'sum' représentant la quantité.
    """
    # Configuration globale de Seaborn pour des plots jolis
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    for key, value in result.items():
        data = value.copy()  # Créer une copie pour éviter les modifications
        total = float(data.loc[key]["sum"])
        data["sum"] = data["sum"]/total
        # Créer le plot
        plt.figure(figsize=(12, 7))  # Taille de la figure
        sns.barplot(x=data.index, y='sum', data=data, palette='viridis', log=False)
        
        # Ajouter un titre et des étiquettes
        plt.title(f"Other movies languages for actor that have been played in {key}", fontsize=16, fontweight='bold')
        plt.xlabel("Languages", fontsize=14)
        plt.ylabel("Actor [%]", fontsize=14)
        
        # Rotation des étiquettes de l'axe X
        plt.xticks(rotation=45, ha='right') 
        
        # Afficher le plot proprement
        plt.tight_layout()
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
    print("Statistique du Chi-2 :", chi2)
    print("P-value :", p_value)
    print("Degrés de liberté :", dof)
    print("\n")
    print("Tableau attendu théorique :\n", expected)
    print("\n")

    # Conclusion
    if p_value < 0.05:
        print("Les distributions sont statistiquement différentes (p < 0.05).")
    else:
        print("Aucune différence significative entre les distributions (p >= 0.05).")

def plot_group_distribution_language(df, nb_group, list_languages):
    """
    Crée un graphique de comparaison des distributions de valeurs par groupes et langues.

    Paramètres :
    - matrix : DataFrame ou array (matrice contenant les données des groupes)
    - nb_group : int (nombre de groupes à comparer)
    - list_languages : list (liste des langues)

    Retour :
    - Affiche un graphique Seaborn.
    """

    matrix = df.copy()
    matrix = matrix.T
    # Créer une liste pour stocker tous les DataFrames individuels
    group_dfs = []

    # Remplir dynamiquement les DataFrames pour chaque groupe
    for i in range(nb_group):
        group_df = pd.DataFrame({
            'Language': list_languages,
            'Value': matrix.iloc[i, :].values / np.sum(matrix.iloc[i, :].values),
            'Type': f'group{i + 1}'
        })
        group_dfs.append(group_df)

    # Combiner tous les groupes dans un seul DataFrame
    combined_df = pd.concat(group_dfs, axis=0)

    # Créer le plot avec seaborn.objects
    p = (
        so.Plot(combined_df, x="Language", y="Value", color="Type")  # Comparaison des groupes
        .add(so.Bar(), so.Dodge())  # Décalage pour distinguer les groupes
        .layout(size=(10, 6))  # Taille du graphique
        .label(title="Comparaison des valeurs Observées et Attendues", 
               x="Langue", 
               y="Proportion")
    )

    # Afficher le graphique
    p.show()
    
    