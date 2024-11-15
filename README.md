# Evolution of the network of actors in the film industry through time 

## 1. Abstract :
The dataset that will be analyse during this project is the [CMU Movie Summary corpus](http://www.cs.cmu.edu/~ark/personas/). This dataset provides lots of informations about +80000 movies. With this dataset as a base, this project aims to analyse and to map the evolution of the network of actors through time. This interconnected system between actors, directors and films is interesting as it is governed and influenced by a multitude of factors that we want to analyse, such as ethnicity, gender, movie genre, etc. To perform our analysis, we will first find and collect more [data](https://datasets.imdbws.com) regarding film directors. After that, the network of actors and directors will be mapped and linked. A multitude of clusters could arise from those networks and will be analysed. Moreover, different statistical analysis will be performed in order to see correlation between different factors. By doing so, it is possible to conclude patterns across the different evolutions of each actors’networks. The evolution of these networks will also be analyzed in conjunction with the evolution of the actors' careers, to see if any trends emerge.

## 2. Research Questions : 
Principal axes : 
* Network through time
  * What are the main factors characterizing the careers of the actors and the evolution of the networks?
* Factor determining clusters (ethnicity, genre, genre of movie)
  * What are the main characteristics of the actors behind the formation of clusters in film production collaboration networks?
* Network between actors and directors
  * What is the link between successful actors and directors?

## 3. Additional datasets : 
[IMDb database](https://datasets.imdbws.com) : Huge database of information about films (millions of movies, series,...), directors,... [(readme.md from IMDb)](https://developer.imdb.com/non-commercial-datasets/)
* As the datasets are quite big (500mo to 1Go), they were not pushed to the repo in github. Specific information about the ratings and the profiles of the directors were taken from the database. The pre-processing is explained in data_preprocessing.ipynb. The readme.md of this dataset can be found in the data folder inside the IMDb folder.

## 4. Methods : 
### Part 1 : Beginning
* **additional dataset** : After choosing and deciding on the project, it was decided that we should search for ways to enrich our networks. Our solution was to add more information about the movie as well as more people to the networks. Hence, we chose to use the IMDb dataset, which is presented in the above section.
* **Literature review** : In parallel to the research of new data, some group members were tasked with finding relevant information about networks theory and useful import in Python. In ADA’s course from last year, week 10 is dedicated to the study of networks, which was a lot of help. We also read the documentation from [NetworkX](https://networkx.org/). However, due to the large number of actors/directors and movies in our networks, it is possible that we will also use another Python library such as [igraph](https://igraph.org/) or [graph-tool](https://graph-tool.skewed.de/) which are more powerful.
* **Pre-processing data** : As the additional dataset was chosen, it was now time to merge the two datasets in order to enrich our data. The processus behind the merging is explained in the data_preprocessing.ipynb notebook. In addition to that, the ethnicity from each actor was also found and added and scrapped from the web. Finally, we were able to create two pickle files which contain all the necessaries information. 
* **Verify feasibility** : As the data was pre-processed and usable, we verified the feasibility of our research questions and adjusted them in accordance with our data and perspective of the projects. The results and the visualizations are available in the results.ipynb notebook.

### Part 2 : Networking
* **Mapping the network**: Using Python library such as NetworkX in order to make graph about the network of actors and directors. Potentially sparse matrix may also be used in order to better visualize the network due to its large size.
* **Network’s metrics and statistics** : Based on our research, multiple metrics related to networks will be computed such as: Degree distribution, community structure, average shortest path length, navigability, centrality measures (degree, closeness, betweenness, katz, PageRank). Based on those metrics, statistical computation may potentially be done in order to find factors that greatly influenced the evolution/form of the networks.

### Part 3 : Unsupervised learning and networks through time 
* **Unsupervised learning** : To answer our second research question, clustering method such as K-NN will be used to find cluster in our network and to define “type” of actors.
* **Networks through time** : Previous mapping will be redone for each year to have an actual evolution of the networks through. As the computation may be time-consuming, this may only be done for actors who have interesting career. 
* **Statistics** : The purpose will be to determine correlations between actors and different careers based on different statistical test (Chi2, t-test,…) on factors and then analyse them. 

### Part 4 : Analysing and reporting 
* **Observation and analysis** : After part 3 is done and results and information have been obtained, analyses and conclusions will be made. 
* **Plot and representation** : In order to showcase our results and to better analyse them, new graph and plot may be done based on our results and statistical findings. 
* **Redaction** : Redaction of the report containing the observations/conclusions on Jekyll

## 5. Organization of the github repo : 

```
├── data                        <- Project data files
│   ├── CMU                             <- Directory that contains all CMU "raw" datasets and a README that explains all features.
│   ├── IMDb                            <- Directory that contains only a README that explains all features of every (IMDb) dataset that we have added to CMU.
│   ├── Actor.pkl                       <- dataset that contains many features **per actor**
│   ├── Movie.pkl                       <- dataset that contains many features **per movie**
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   │     └── data_preprocessing.ipynb       <- Notebook that reproduce all the preprocessing steps in order to get Actor.pkl and Movie.pkl (Merge CMU and IMDb)
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

## 6. Proposed timeline : 
* Part 1 : week 8,9
*	Part 2 : week 10,11
*	Part 3 : week 12,13
* Part 4 : week 13,14

## 7. Organization within the team 
Each team member will be asigned with a specific research question and the others will have to check on the other and to give them neutral feedback. In addition to that, half of the team will directly begin working on the homework 2 so that is done by next week. Finally, one or two person will have the responsability to write the report with Jekyll while the others finish the analysis. 

