import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.cm as cm

import plotly.graph_objects as go
import plotly.subplots as sp

from tqdm import tqdm




def create_actor_career_dataset(Actor):

    Actor_career = Actor.copy()
    Actor_career["actor_age_atmovierelease"] = Actor_career["actor_age_atmovierelease"].apply(lambda x: [pd.NA if pd.isna(age) else int(age) for age in x])
    
    Actor_career["actor_age_atmovierelease"] = Actor_career["actor_age_atmovierelease"].apply(lambda x: [val for val in x if not (pd.isna(val) or val < 0)])

    # Compute career start, end, and length
    Actor_career["Career_Start_age"] = Actor_career.apply(lambda row: min([val for val in row["actor_age_atmovierelease"] if val > 0], default=-1),axis=1,)
    Actor_career["Career_End_age"] = Actor_career.apply(lambda row: max([val for val in row["actor_age_atmovierelease"] if val > 0], default=-1),axis=1,)
    Actor_career["Career_length"] = Actor_career["Career_End_age"] - Actor_career["Career_Start_age"]

    # Filter out invalid career start ages, actor heights and genders
    Actor_career = Actor_career[Actor_career["Career_Start_age"] >= 0].reset_index(drop=True)
    Actor_career = Actor_career[Actor_career["actor_gender"].notna()].reset_index(drop=True)


    # Helper function to fill career years
    def fill_career_years(row, max_career_years):
        age_at_release = np.array(row["actor_age_atmovierelease"])
        nbr_films = len(age_at_release)
        counts = np.zeros(max_career_years + 1)
        if nbr_films != 0:
            for year in age_at_release:
                if year >= 0 and row["Career_Start_age"] >= 0:
                    counts[year - row["Career_Start_age"]] += 1
        return counts


    max_career_years = int(max(Actor_career.apply(lambda row: max(row["actor_age_atmovierelease"]), axis=1) - Actor_career["Career_Start_age"]))

    # Compute career profils
    career_counts = np.array(Actor_career.apply(fill_career_years, axis=1, max_career_years=max_career_years).tolist())
    Nbr_films = [f"Nbr_films_{i+1}" for i in range(max_career_years)]
    Actor_career = pd.concat([Actor_career, pd.DataFrame(0.0, index=Actor_career.index, columns=Nbr_films)], axis=1)
    for i in range(max_career_years):
        Actor_career[f"Nbr_films_{i+1}"] = career_counts[:, i]

    Actor_career['Total_nbr_films'] = Actor_career.iloc[:,-max_career_years:].sum(axis=1)
    
    return Actor_career


def prepare_career_dataset_KNN(Actor_career):

    columns_to_drop = ['Freebase_actor_ID', 'actor_name', 'actor_DOB', 'actor_gender',
        'actor_height', 'ethnicity', 'Freebase_movie_ID', 'actor_age_atmovierelease',
        'Career_Start_age', 'Career_End_age', 'Career_length']
    
    Career_dataset = Actor_career.drop(columns=columns_to_drop, errors='ignore')
    scaler = StandardScaler()
    Career_dataset_std = scaler.fit_transform(Career_dataset)

    return Career_dataset_std


def plot_elbow_method(Career_dataset,cluster_range,random_state = 0):
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



def knn_clustering_and_plot(Career_dataset_std, n_clusters, Career_dataset, random_state=0):

    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(Career_dataset_std)
    
    columns_to_drop = ['Freebase_actor_ID', 'actor_name', 'actor_DOB', 'actor_gender',
        'actor_height', 'ethnicity', 'Freebase_movie_ID', 'actor_age_atmovierelease',
        'Career_Start_age', 'Career_End_age', 'Career_length','Total_nbr_films','Labels']
    Career_dataset = Career_dataset.drop(columns=columns_to_drop, errors='ignore')
    Career_dataset.columns = range(0, len(Career_dataset.columns))

    fig = go.Figure()


    for i in range(n_clusters):
        cluster_data = Career_dataset[labels == i]
        centroid = cluster_data.mean(axis=0)
        
        fig.add_trace(go.Scatter(
            x=centroid.index,  
            y=centroid.values,  
            mode='markers+lines',  
            name=f'Cluster {i}'))


    fig.update_layout(
        autosize = True,
        width=None,
        height = None,
        title="Cluster Centroids",
        xaxis_title="Career year",
        yaxis_title="Number of films",
    )
    fig.show()

    return labels


def career_plot(Career_dataset, actor_name):
    actor_data = Career_dataset[Career_dataset['actor_name'] == actor_name]

    if actor_data.empty:
        print(f"No actor found for: {actor_name}")
        return
    
    columns_to_drop = ['Freebase_actor_ID', 'actor_name', 'actor_DOB', 'actor_gender',
                       'actor_height', 'ethnicity', 'Freebase_movie_ID', 'actor_age_atmovierelease',
                       'Career_Start_age', 'Career_End_age', 'Career_length', 'Total_nbr_films', 'Labels']
    actor_data = actor_data.drop(columns=columns_to_drop, errors='ignore')

    actor_data = actor_data.transpose().reset_index()
    actor_data.columns = ['Career Year', 'Number of Films']
    actor_data['Career Year'] = range(0,len(actor_data))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actor_data['Career Year'],  
        y=actor_data['Number of Films'],  
        mode='lines+markers',
        name=actor_name
    ))

    fig.update_layout(
        autosize=True,
        width=None,
        height = None,
        title=f"{actor_name}'s Career",
        xaxis_title="Career Year",
        yaxis_title="Number of Films",
    )

    fig.show()





def plot_gender_proportions_by_cluster(dataset, label_column='Labels', gender_column='actor_gender'):

    gender_counts = dataset.groupby([label_column, gender_column]).size().reset_index(name='count')
    cluster_totals = dataset.groupby(label_column).size().reset_index(name='total_count')
    gender_proportions = gender_counts.merge(cluster_totals, on=label_column)
    gender_proportions['proportion'] = gender_proportions['count'] / gender_proportions['total_count']
    
    clusters = gender_proportions[label_column].unique()


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
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=proportions,
                hole=0.3,  
            ),
            row=1,
            col=idx + 1
        )
    
    # Update layout
    fig.update_layout(
        autosize = True,
        width=None,
        height = None,
        title_text="Gender Proportions Across Clusters",
        title_x=0.5,
        showlegend=False, 
    )
    
    # Show the combined plot
    fig.show()