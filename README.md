# Evolution of the network of actors in the film industry through time 

## 1. Abstract : => réparer le titre

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
=> rajouter les méthodes de network/mapping de Pierre (literature review aussi)
* Mapping the whole network (different map), clustering method (Unsupervised learning, K-NN,…), literature review, statistics (T-test, Chi2, …)

## 5. Organisation of the github repo : 

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
* Émile : 
* Alexandre :
* À faire : rédaction sur le site Jekill, homework 2 

## 8. Question TA (Optional) :
