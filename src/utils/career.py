import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats

def ttest(df1, df2, feature, alpha=0.05):
    """
    Perform an independent t-test on a specified feature between two dataframes, and say it's the test is fine !
    """
    df1_copy=df1.copy()
    df2_copy=df2.copy()
    
    df1_copy = pd.to_numeric(df1_copy[feature], errors='coerce').dropna()
    df2_copy = pd.to_numeric(df2_copy[feature], errors='coerce').dropna()
    # Drop rows with NaN values in the specified feature
    df1_copy = df1_copy.dropna()
    df2_copy = df2_copy.dropna()
    
    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(df1_copy, df2_copy)
    
    # Check if we reject the null hypothesis
    reject_null = p_value < alpha

    if reject_null :
        print(f"The null hypothesis can be rejected, there is a significance difference between the 2 distributioms. (p_value : {p_value})")
    else :
        print(f"The null hypothesis can't be rejected, there is no significance differences. (p_value : {p_value})")
    
    return t_stat, p_value

def fill_career_years(row, max_career_years):
    """
    compute cumulative number of films realsed per year
    """
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
    add informations about actor career in dataset such as : career start age, career end age, career length, number of film release per career stages.
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
    Actor_career['Total_number_of_films'] = Actor_career.iloc[:, -max_career_years:].sum(axis=1)
    
    return Actor_career

def prepare_career_dataset_KMeans(Actor_career):
    """
    drop unnecessary columns and scale data
    """
    columns_to_drop = [
        'Freebase_actor_ID', 'actor_name', 'actor_DOB', 'actor_gender',
        'actor_height', 'ethnicity', 'Freebase_movie_ID', 'actor_age_atmovierelease',
        'Career_Start_age', 'Career_End_age', 'Career_length', 'Total_number_of_films', 'Labels'
    ]
    Career_dataset = Actor_career.drop(columns=columns_to_drop, errors='ignore')
    scaler = StandardScaler()
    Career_dataset_std = scaler.fit_transform(Career_dataset)

    return Career_dataset_std

def kmeans_clustering(Career_dataset_std, n_clusters, random_state=0):
    """
    do the clustering for n_clusters 
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(Career_dataset_std)
    return labels

def get_dict_cluster(nb_clusters, df, label_column='Labels'):
    """
    return a dict with as key the index of clusetr
    """
    cluster_data = {}

    for i in range(nb_clusters):
        cluster_data[i] = df[df[label_column] == i]

    return cluster_data