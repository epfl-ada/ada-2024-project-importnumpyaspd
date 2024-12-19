import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import plotly.graph_objects as go
import plotly.subplots as sp

from tqdm import tqdm

# compute cumulative number of films realsed per year
def fill_career_years(row, max_career_years):
    age_at_release = np.array(row["actor_age_atmovierelease"])
    nbr_films = len(age_at_release)
    counts = np.zeros(max_career_years + 1)
    if nbr_films != 0:
        for year in age_at_release:
            if year >= 0 and row["Career_Start_age"] >= 0:
                counts[year - row["Career_Start_age"]] += 1
    return counts

def create_actor_career_dataset(Actor):
    """
    Add informations about actor career in dataset such as : career start age, career end age, career length, number of film release per career stages.
    """
    # create actor df copy
    Actor_career = Actor.copy()
    
    # clean trhe age at movie realease column
    Actor_career["actor_age_atmovierelease"] = Actor_career["actor_age_atmovierelease"].apply(
        lambda x: [pd.NA if pd.isna(age) else int(age) for age in x]
    )
    Actor_career["actor_age_atmovierelease"] = Actor_career["actor_age_atmovierelease"].apply(
        lambda x: [val for val in x if not (pd.isna(val) or val < 0)]
    )

    # add career start/end age, career length
    Actor_career["Career_Start_age"] = Actor_career.apply(
        lambda row: min([val for val in row["actor_age_atmovierelease"] if val > 0], default=-1), axis=1
    )
    Actor_career["Career_End_age"] = Actor_career.apply(
        lambda row: max([val for val in row["actor_age_atmovierelease"] if val > 0], default=-1), axis=1
    )
    Actor_career["Career_length"] = Actor_career["Career_End_age"] - Actor_career["Career_Start_age"]

    # keep only valid career start age + keep only if gender
    Actor_career = Actor_career[Actor_career["Career_Start_age"] >= 0].reset_index(drop=True)
    Actor_career = Actor_career[Actor_career["actor_gender"].notna()].reset_index(drop=True)

    # compute max nbr career year
    max_career_years = int(
        max(Actor_career.apply(lambda row: max(row["actor_age_atmovierelease"]), axis=1) - Actor_career["Career_Start_age"])
    )

    # compute cumulative number of films realsed per year
    career_counts = np.array(
        Actor_career.apply(fill_career_years, axis=1, max_career_years=max_career_years).tolist()
    )
    Nbr_films = [f"Nbr_films_{i+1}" for i in range(max_career_years)]
    Actor_career = pd.concat([Actor_career, pd.DataFrame(0.0, index=Actor_career.index, columns=Nbr_films)], axis=1)

    # add count per year
    for i in range(max_career_years):
        Actor_career[f"Nbr_films_{i+1}"] = career_counts[:, i]

    #total number of films for each actor
    Actor_career['Total_nbr_films'] = Actor_career.iloc[:, -max_career_years:].sum(axis=1)
    
    return Actor_career


def prepare_career_dataset_KNN(Actor_career):
    """
    Drop unnecessary columns and scale data
    """
    columns_to_drop = ['Freebase_actor_ID', 'actor_name', 'actor_DOB', 'actor_gender',
        'actor_height', 'ethnicity', 'Freebase_movie_ID', 'actor_age_atmovierelease',
        'Career_Start_age', 'Career_End_age', 'Career_length']
    
    Career_dataset = Actor_career.drop(columns=columns_to_drop, errors='ignore')
    scaler = StandardScaler()
    Career_dataset_std = scaler.fit_transform(Career_dataset)

    return Career_dataset_std


def plot_elbow_method(Career_dataset,cluster_range,random_state = 0):
    """
    plot elbow to choose nbr of necessary cluster
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



def knn_clustering(Career_dataset_std, n_clusters, random_state=0):
    """
    Do the clustering for n_clusters 
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(Career_dataset_std)
    return labels


def plot_career_data(Career_dataset, labels, actor_name=None, n_clusters=5, alpha=0.6, save = False):
    """
    Plot the number of films through the career.
    If labels are provided, plot distinct curves for each cluster.
    Otherwise, plot the same way.
    """
    # Columns to drop
    columns_to_drop = [
        'Freebase_actor_ID', 'actor_name', 'actor_DOB', 'actor_gender',
        'actor_height', 'ethnicity', 'Freebase_movie_ID', 'actor_age_atmovierelease',
        'Career_Start_age', 'Career_End_age', 'Career_length', 'Total_nbr_films', 'Labels'
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


def plot_gender_proportions_by_cluster(dataset, n_clusters=2, alpha=0.55, save = False):
    """
    Plot proportion of gender in clusters with custom colors from viridis palette and transparency
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

        # Map gender labels to colors from viridis with transparency
        custom_colors = [colors[i % n_clusters] for i in range(len(labels))]  # Cycle through available colors

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=proportions,
                hole=0.3,  
                marker=dict(colors=custom_colors)
            ),
            row=1,
            col=idx + 1
        )
    
    # Update layout
    fig.update_layout(
        autosize=True,
        width=None,
        height=None,
        title_text="Gender Proportions Across Clusters",
        title_x=0.5,
        showlegend=False,
    )
    
    # Show the combined plot
    fig.show()
    
    if save:
        filename = f"gender_camembert_cluster.html"
        fig.write_html(filename)
    
def plot_cluster_histogram(Actor_career, column_name, n_clusters=3, bin_width=1, max_value=None, min_value=None, kde_option = False, logscale = False,Pourcentage = False, alpha = 0.3, save = False):
    """
    plot histogramm relatively to column "column name" parameter. Identify the distribution for each cluster (define in "Labels" columns) 
    """
    # column containing label of cluster
    labels_column='Labels'
    
    # compute range (for axis)
    if min_value is None:
        min_value = Actor_career[column_name].min()
    if max_value is None:
        max_value = Actor_career[column_name].max()
    
    # generate color to be good wiht veridis
    viridis = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    colors = [mcolors.rgb2hex(c) for c in viridis]
    

    plt.figure(figsize=(10, 6))
    
    # one color per cluster
    for i in range(n_clusters):
        cluster_data = Actor_career[Actor_career[labels_column] == i]
        bins = np.arange(min_value, max_value + bin_width, bin_width)
        if Pourcentage:
            sns.histplot(
            data=cluster_data, x=column_name, bins=bins, kde=kde_option,stat='percent',
            alpha=alpha, color=colors[i], label=f'Cluster {i}')
        else :
            sns.histplot(
            data=cluster_data, x=column_name, bins=bins, kde=kde_option,
            alpha=alpha, color=colors[i], label=f'Cluster {i}')

    # x,y limit for axis
    plt.xlim(min_value, max_value)

    if logscale : #log 
        plt.yscale('log')
    plt.title(f"Distribution of {column_name.replace('_', ' ').title()}", fontsize=16, fontweight='bold')
    plt.xlabel(f"{column_name.replace('_', ' ').title()}", fontsize=14)
    if Pourcentage:
        plt.ylabel("Pourcentage", fontsize=14)
    else :
        plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [i for i in range(n_clusters)]  # order cluster
    plt.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        title="Clusters",
        fontsize=12,
        title_fontsize=14,
        loc='upper right',
    )
    
    plt.show()

    if save:
        plt.savefig(f"{column_name}histo.png", dpi=300, bbox_inches="tight", transparent=True)

def get_dict_cluster(nb_clusters, df, label_column='Labels'):
    """
    return a dict with as key the index of clusetr
    """
    cluster_data = {}

    for i in range(nb_clusters):
        cluster_data[i] = df[df[label_column] == i]

    return cluster_data