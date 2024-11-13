# Evolution of the network of actors in the film industry through time 

## 1. Abstract :

The dataset that will be analyse during this project is the [CMU Movie Summary corpus](http://www.cs.cmu.edu/~ark/personas/). This dataset provides lots of informations about +80000 movies. With this dataset as a base, this project aims to analyse and to map the evolution of the network of actors through time. This interconnected system between actors, directors and films is interesting as it is governed and influenced by a multitude of factors such as ethnicity, movie genre, etc. To perform our analysis, we will first find and collect more [data](https://datasets.imdbws.com) regarding film directors. After that, the network of actors and directors will be mapped and linked. A multitude of clusters could arise from those networks and will be analysed. Moreover, different statistical analysis will be performed in order to see correlation between different factors. By doing so, it is possible to conclude patterns across the different evolutions of each actors’networks. 

## 2. Research Questions : 
Principal axes : (to be refined into questions and add explanations/axes below)
* Network through time
* Factor determining clusters (ethnicity, genre, genre of movie)
* Relation and network between actors and directors 

Old questions : 
* How is an actor network’s impacted during his career ?
* What impacts this evolution ? ethnicity, gender, genre of movie
* Duo, trio or quartet of actors ? Trend to work together multiple times ? 
* Specific crew or chosen actors per director ? (multiple link between actor and director, working multiple times together)
* Cluster based on different factors (ethnicity, genre, number of films together) ?
* Does the network and its form evolved during the last century ? Different trend ?
* Different “type” of actor ? 
* What are the relation between successful actors and movie directors ?
* Impact of a specific movie on relation and network of actors ?

## 3. Proposed additional dataset : 
[IMDb database](https://datasets.imdbws.com) : Huge database of information about film (millions of movies, series,...), directors, crew member [(readme.md of the database)](https://developer.imdb.com/non-commercial-datasets/)
* Name.basics.tsv.gz => unique index of the name and the according name
* Title.basics.tsv.gz => unique index of  film and other information (we only use the index and name of the film)
* Title.crew.tsv.gz => unique index of film and unique index of directors and writers
* Title.ratings.tsv.gz => unique index of film and ratings, number of votes
* Most of those datasets contains irrelevant information for our analysis and hence rows/columns were discard. This was motivated by the fact that each dataset were larger than 500mo and some more than Go. As we did not want to fill all the available space on Github and to load each time Go of data when we will use the python file, we decided to drop entries in the different database.  

## 4. Methods : 
Different methods will be use in order to answer to our research questions :
1. Literature review :
   * Making some research on network mapping and statistics in order to determine relevant factors that can be used in order to analyse networks, edge and nodes in our project. This can be done by looking at the course from last year as it covered in the last week networks or by simply looking at some articles and project online. 
3. Mapping different network of actors and directors :
   * Linking different actors based on the same movie
   * Linking different actors on the same directors
   * Linking different actors based multiple factors (movie and ethnicity)
4. Unsupervised learning :
   * Using clustering method such as K-NN in order to determine cluster and potentially different type of relation between actors or event type of actors or directors
6. Statistics :
   * Using statistical approach in order to determine correlations between factors and to determine the most influential factors on different networks
   * T-test, Chi2, …
8. Networks statistics :
   * As we will analyse networks, it is necessary to compute and to define coefficient and value that can be used in order to make observations and conclusion about our data. As such, based on our literature review, we will use the following :
   * Diameters, largest nodes, hubs classification, shortest path, transitivity, clustering coefficient, Katz centrality, betweenness centrality, etc.


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
Week by week till the last Milestone : 
* Week 8 : Pre-processing data, verify feasibility of research questions, literature review on network
* Week 9 : merging dataset, verify feasibility of research questions, deliver milestone 2
* Week 10 : Half of the team on homework, other literature review, start mapping
* Week 11 : mapping the network, different versions, clustering
* Week 12 : Statistical hypothesis and computation, clustering
* Week 13 : Plot and redaction of Jekyll
* Week 14 : Redaction of jekyll and final touch

## 7. Organization within the team 
Faire un tableau ?
* Pierre : literature review of networks 
* Thibault : literature review of networks
* Thomas : Pre-processing of data
* Emile : 
* Alexandre :
* À faire : rédaction sur le site Jekill, homework 2 

## 8. Question TA (Optional) :
