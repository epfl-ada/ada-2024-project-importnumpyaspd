# "Charting Careers: The Data Map of Acting Success and Collaborations"

## 1. Abstract: 
The cinematic universe has always been a fascinating world. How many children had the dream to become the next star of Hollywood/Bollywood (or other)? Unfortunately, very few people have achieved their dream, and it is common sense to think that it needs more than passion and willingness; sometimes having a good network, especially with film directors, could help. The project aims to better understand what can make an actor successful, and the angle of research is based on the possible network and relations between actors and directors. This analysis could be done thanks to the [CMU Movie Summary corpus](http://www.cs.cmu.edu/~ark/personas/), a dataset that provides information about actors for over 80000 movies, and the [IMDB database](https://datasets.imdbws.com) for additional data on directors and film ratings. The analysis is done in three parts. First, an analysis of the career profiles of actors is performed to identify what makes you a prolific actor and what you would likely not be. Then, the connections between these successful actors and film directors are analysed mostly throughout directed graph analysis. Finally, since having relations with directors is not the unique reason for success, an analysis between actor-actor is also performed. More specifically, we analyze the network from the point of view of the languages used in the films to see whether global connections could help build a prolific career.

## 2. Research Questions: 
Principal axes:
* Career profiles:
    * What are the main factors characterizing the career profiles of the actors?
* Networking between actors and movie directors
    * What made an actor successful?
    * What are the types of directors that are linked to successful actors?
* Actor Network through languages
    * What is the link between languages and networking, and what is the relationship between career profiles and the degree of internationalization of actors?

## 3. Additional datasets : 
[IMDb database](https://datasets.imdbws.com) : Huge database of information about films (millions of movies, series,...), directors,... [(readme.md from IMDb)](https://developer.imdb.com/non-commercial-datasets/)
* As the datasets are quite big (500mo to 1Go), they were not pushed to the repo in github. Specific information about the ratings and the profiles of the directors was taken from the database. The preprocessing is explained in data_preprocessing.ipynb. The readme.md of this dataset can be found in the data folder inside the IMDb folder.
* In order to retrieve the ID of some actor, we use web scrapping method using the library `BeautifulSoup`

## 4. Methods :

### K-means Unsupervised
- **Usupervised K-means**: An unsupervised machine learning algorithm that partitions data into distinct groups (clusters) by iteratively assigning data points to the nearest cluster centroid and recalculating centroids. The algorithm minimizes the variance within each cluster without requiring labeled data, making it ideal for exploratory data analysis.
- **Elbow's Method**: A technique to determine the optimal number of clusters in K-means by plotting the sum of squared errors (SSE) against the number of clusters and identifying the "elbow point" where the SSE starts diminishing less rapidly.
- **Centroids**: The central points of clusters calculated as the mean position of all points in the cluster. These are updated iteratively during the K-means process to better represent the cluster centers.

### Network Graphs and Directed Graphs
- **Undirected Networks graph**: Undirected graphs where nodes (actors) represent entities and edges (movies) represent relationships or interactions between them.
- **Bi-partite Directed Graph**: A graph where nodes are divided into two distinct sets (successful actors to directors) with edges directed from one set to the other.
- **Eigen-vector Centrality**: A measure of the influence of a node (actor) in a network based on the idea that connections to high-scoring nodes contribute more to the score of the node in question.
- **Spring Layout Algorithm**: A visualization algorithm that positions nodes based on a force-directed layout, simulating physical forces such as attraction and repulsion to produce a visually appealing graph.
- **Louvain Algorithm**: A community detection algorithm that identifies clusters or communities in a graph by optimizing modularity, a measure of the density of edges within communities compared to edges between them.

### Statistical Tests
- **t-Test**: A parametric test used to compare the means of two groups (in our case, two normal distributed characteristics) and assess whether their differences are statistically significant.
- **Chi2 Test**: A statistical test used to determine whether there is a significant association between categorical variables by comparing observed and expected frequencies in a contingency table.


## 5. Organization of the github repo : 

```
├── data                        <- Project data files
│   ├── CMU                             <- Directory that contains all CMU "raw" datasets and a README that explains all features.
│   ├── IMDb                            <- Directory that contains only a README that explains all features of every (IMDb) dataset that we have added to CMU.
│   ├── Actor.pkl                       <- dataset that contains many features **per actor**
│   └── Movie.pkl                       <- dataset that contains many features **per movie**
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   │     └── data_preprocessing.ipynb  <- Notebook that reproduces all the preprocessing steps in order to get Actor.pkl and Movie.pkl (Merge CMU and IMDb)
│   └──utils                    <- Utility directory
│         ├── career.py                 <- helper functions for section 1 (Career profiles)
│         ├── Director_Actor.py         <- helper functions for section 1 (Networking between actors and movie directors)
│         ├── langue.py                 <- helper functions for section 1 (Actor Network through languages)
│         └── plot.py                   <- helper functions for plotting  (All sections)
├── results.ipynb               <- a well-structured notebook showing the results
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing Python dependencies
└── README.md
```


## 6. Organization within the team 

Thomas Walter: Data retrieval, data cleaning, web site development. 

Emile Favre: Network clustering, ethnic analysis, language analysis. 

Alexandre Mascaro: Actor/Director relations networks creation and analysis.

Thibault Schiesser: Data retrieval, data cleaning, network clustering, language analysis. 

Pierre Maillard: undirected graph creation, career profile creation and analysis.

