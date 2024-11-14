# Evolution of the network of actors in the film industry through time 

## 1. Abstract :
The dataset that will be analyse during this project is the [CMU Movie Summary corpus](http://www.cs.cmu.edu/~ark/personas/). This dataset provides lots of informations about +80000 movies. With this dataset as a base, this project aims to analyse and to map the evolution of the network of actors through time. This interconnected system between actors, directors and films is interesting as it is governed and influenced by a multitude of factors such as ethnicity, movie genre, etc. To perform our analysis, we will first find and collect more [data](https://datasets.imdbws.com) regarding film directors. After that, the network of actors and directors will be mapped and linked. A multitude of clusters could arise from those networks and will be analysed. Moreover, different statistical analysis will be performed in order to see correlation between different factors. By doing so, it is possible to conclude patterns across the different evolutions of each actors’networks. The evolution of these networks will also be analyzed in conjunction with the evolution of the actors' careers, to see if any trends emerge.

## 2. Research Questions : 
Principal axes : 
* Network through time
  * What are the main factors characterizing the careers of the players and the evolution of the network??
* Factor determining clusters (ethnicity, genre, genre of movie)
* Relation and network between actors and directors 

## 3. Proposed additional dataset : 
[IMDb database](https://datasets.imdbws.com) : Huge database of information about film (millions of movies, series,...), directors,... [(readme.md of the database)](https://developer.imdb.com/non-commercial-datasets/)
* Name.basics.tsv.gz => unique index of the name and the according name
* Title.basics.tsv.gz => unique index of film and other information about them
* Title.crew.tsv.gz => unique index of film and unique index of directors and writers
* Title.ratings.tsv.gz => unique index of film and ratings, number of votes
* As the datasets are quite big (500mo to 1Go), they were not push to the repo in github. Specific information about the ratings and the profile of the director were taken from the database. The pre-processing is explain in data_preprocessing.ipynb.

## 4. Methods : 
Part 1 : Beginning
*	Data-scraping and finding additional dataset:
 *	After choosing and deciding on the project, it was decided that we should search for ways to enrich our networks. The most direct solution was to add more information about the movie as well as more people to the networks. Hence, we chose to use the IMDb dataset, which is presented in the above section. 
*	Literature review:
 *	In parallel to the research of new data, some group members were tasked with finding relevant information about networks theory and useful import in Python. In ADA’s course from last year, week 10 is dedicated to the study of networks, which was a lot of help. We also read the documentation from [NetworkX](https://networkx.org/). However, due to the large number of actors/directors and movies in our networks, it is possible that we will also use another Python library such as [igraph](https://igraph.org/) or [graph-tool](https://graph-tool.skewed.de/) which are more powerful.
*	Pre-processing the data: 
 *	As the additional dataset was chosen, it was now time to merge the two datasets in order to enrich our data. The processus behind the merging is explained in the data_preprocessing.ipynb notebook. In addition to that, the ethnicity from each actor was also found and added and scrapped from the web. Finally, we were able to create two pickle files which contain all the necessaries information. 
*	Verify the feasibility:
 *	As the data was pre-processed and usable, we verified the feasibility of our research questions and adjusted them in accordance with our data and perspective of the projects. The results and the visualization are available in the results.ipynb notebook.

Part 2 : Networking
*	Mapping the network: 
 *	Using Python library such as NetworkX in order to make graph about the network of actors and directors. Potentially based them on different factors. Potentially sparse matrix may also be used in order to better visualize the network due to its large size. 
*	Computing network’s metrics and doing network’s statistics:
 *	Based on our research, multiple metrics related to networks will be computed such as: Degree distribution, community structure, average shortest path length, navigability, centrality measures (degree, closeness, betweenness, katz, PageRank),…
 *	Based on those metrics, statistical computation may potentially be done in order to find factors that greatly influenced the evolution/form of the networks.

Part 3 : Unsupervised learning and networks through time 
*	Unsupervised learning: 
 *	To answer our second research question, clustering method such as K-NN will be used to find cluster in our network and to define “type” of actors. 
*	Networks through time:
 *	 Previous mapping will be redone for each year to have an actual evolution of the networks through. As the computation may be time-consuming, this may only be done for actors who have interesting career. 
*	Statistics:
 *	The purpose will be to determine correlations between actors and different careers based on different statistical test (Chi2, t-test,…) on factors and then analyse them. 

Part 4 : Analysing and reporting 
*	Observation and analysis :
 *	After part 3 is done and results and information have been obtained, analyses and conclusions will be made. 
*	Plot and representation :
 *	In order to showcase our results and to better analyse them, new graph and plot may be done based on our results and statistical findings. 
*	Redaction :
 *	Redaction of the report containing the observations/conclusions on Jekyll

## 5. Organization of the github repo : 

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
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
Each of the team member will be asigned with a specific research question and the other will have for duty to to check on the other and to give them neutral feedback. In addition to that, half of the team will directly begin working on the homework 2 so that is done by next the middle of next week. Finally, one or two person will have the responsability to write the report with Jekyll while the other finished the analysis. 

