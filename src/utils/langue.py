################################################################################################################################
def create_actor_language_dataset(Movie, Actor, number_language):

    Movie['Movie_languages'] = Movie['Movie_languages'].str.replace(r'\s*,\s*', ',', regex=True)

    # Convertir la colonne en liste si elle contient des langues séparées par des virgules
    Movie['Movie_languages'] = Movie['Movie_languages'].str.split(',')
    Movie = Movie.explode('Movie_languages')
    
    # Supprimer les films sans langue
    Movie_wo_nan_language = Movie[Movie["Movie_languages"] != ""]
    print(f"We lose {(Movie.shape[0] - Movie_wo_nan_language.shape[0]) / Movie.shape[0] * 100:.2f}% of the dataset of movies with this operation.")

    # Sélectionner les principales langues
    language_to_keep = Movie_wo_nan_language["Movie_languages"].value_counts().head(number_language).index
    Movie_selected_languages = Movie[Movie.Movie_languages.isin(language_to_keep)]

    # Créer un dictionnaire Actor -> Movies
    Actor_movie = Actor["Freebase_movie_ID"]
    Actor_movie.index = Actor.Freebase_actor_ID
    dict_Actor_movie = Actor_movie.to_dict()

    # Inverser ce dictionnaire: Movie -> Actor
    dict_movie_actor = {}
    for key, values in dict_Actor_movie.items():
        for value in values:
            if value not in dict_movie_actor:
                dict_movie_actor[value] = []
            dict_movie_actor[value].append(key)
    
    # Ajouter la colonne 'list_actor' dans le DataFrame Movie
    Movie = Movie.reset_index()
    Movie["list_actor"] = Movie["Freebase_movie_ID"].map(dict_movie_actor)

    # Supprimer les films sans acteurs
    Movie = Movie[~Movie.list_actor.isna()]

    # Créer un dictionnaire pour chaque langue contenant les films correspondants
    movie_dict = {}
    for language in language_to_keep:
        movie_dict[language] = Movie[Movie.Movie_languages == language]
    
    # Créer un dictionnaire pour stocker les acteurs par langue (1 si présent, 0 sinon)
    actor_per_language = {language: {} for language in language_to_keep}

    # Vérifier la présence des acteurs dans chaque langue
    for language in language_to_keep:
        df = movie_dict[language]
        
        # Pour chaque film, lister les acteurs et marquer leur présence dans cette langue
        for index, row in df.iterrows():
            actor_list = row["list_actor"]  # Liste des acteurs pour ce film
            for actor in actor_list:
                actor_per_language[language][actor] = 1  # Marquer 1 si l'acteur est présent

    # Fusionner tous les dictionnaires dans un seul DataFrame
    actor_lists = []
    for language in language_to_keep:
        # Transformer chaque dictionnaire en DataFrame (colonne d'acteurs par langue)
        actor_df = pd.DataFrame.from_dict(actor_per_language[language], orient='index', columns=[language])
        actor_lists.append(actor_df)

    # Concaténer tous les DataFrames (les acteurs qui n'apparaissent pas dans une langue auront NaN)
    total = pd.concat(actor_lists, axis=1, join='outer')
    total.columns = language_to_keep
    # Retourner le dictionnaire des acteurs par langue (présence = 1)

    total = total.applymap(lambda x: 1 if pd.notna(x) else 0)
    
    return total

def create_cross_language(df):
    result = {}
    n_language = df.shape[1]
    
    for index in range(n_language):
        # Créer une copie du DataFrame pour y ajouter les résultats
        newdf = df.copy()
        
        # Filtrer la colonne pour garder uniquement les éléments non-nuls (différents de 0)
        filtered_column = df.iloc[:, index][df.iloc[:, index] != 0]
        
        # Pour chaque langue, ajouter les données relatives à l'acteur et film
        for j in df.columns:
            if j != df.columns[index]:
                # Remplir newdf[j] en marquant '1' quand un acteur a joué dans un film de la langue courante
                newdf[j] = (df[df.columns[index]] + df[j] > 1).astype(int) # Si acteur dans cette langue et film, on met 1
         
        # Appliquer le filtre pour ne garder que les lignes où l'acteur a joué dans un film de la langue en question
        newdf = newdf.loc[filtered_column.index, :]

        # Transposer le DataFrame pour aligner les résultats avec chaque langue en colonnes
        newdf = newdf.T
        
        # Ajouter une colonne de somme, indiquant combien de films un acteur a dans chaque langue
        newdf["sum"] = newdf.sum(axis=1)

        # Ajouter le DataFrame traité dans le dictionnaire des résultats
        result[df.columns[index]] = newdf

    return result
    
def plot_language_histograms(result):
    # Parcourir chaque langue et ses résultats dans le dictionnaire
    for key, value in result.items():
        # Extraire les données que vous voulez afficher (les sommes pour chaque langue)
        data = value
        
        # Plot
        plt.figure(figsize=(10, 6))
        
        # Créer un histogramme des valeurs de sum
        plt.bar(data.index, data['sum'])
        
        # Ajouter un titre et des labels
        plt.title(f"Histogramme des films pour {key}")
        plt.xlabel("Langues")
        plt.ylabel("Nombre de films")
        
        # Appliquer une échelle logarithmique à l'axe des ordonnées
        plt.yscale('log')
        
        # Afficher le graphique
        plt.xticks(rotation=45)  # Faire pivoter les labels des axes x pour mieux les afficher
        plt.show()