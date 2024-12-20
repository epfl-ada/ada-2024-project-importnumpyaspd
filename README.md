# "Charting Careers: The Data Map of Acting Success and Collaborations"

## 1. Abstract : 
The cinema universe has always been a fascinating world. How many children had the dream to become the next star of Hollywood/Bollywood (or other)? Unfortunately, very few people has achieved their dream, and it is common sense to think that it need more than passion and willingness, sometimes having a good network, especially with films directors, could help. The project aims to better understand what can make an actor successful, and the angle of research is based on the possible network and relations between actors and directors. This analyse could be done thanks to the [CMU Movie Summary corpus](http://www.cs.cmu.edu/~ark/personas/), a dataset that provides information about actors for over 80000 movies, and the [IMDB database](https://datasets.imdbws.com) for additional data on directors and film ratings. The analysis is done in three parts. First, an analyse of the career profile of actors is performed to identify what makes you a prolific actor, and what would likely not to. Then, the connections between these successful actors and films directors is analysed mostly throughout directed graph analysis. Finally, since having relations with directors is not the unique reason to success, an analysis between actor-actor is also performed. More specifically, we analyze the network from the point of view of the languages used in the films, to see whether global connections could help build a prolific career.

## 2. Research Questions : 
Principal axes:
* Career profiles:
    * What are the main factors characterizing the careers profiles of the actors?
* Networking between actors and movie directors
    * What made an actor successful?
    * What are the type of directors that are linked to successful actors?
* Actor Network through languages
    * What is the link between languages and networking, and what is the relationship between career profiles and the degree of internationalization of actors?

## 3. Additional datasets : 
[IMDb database](https://datasets.imdbws.com) : Huge database of information about films (millions of movies, series,...), directors,... [(readme.md from IMDb)](https://developer.imdb.com/non-commercial-datasets/)
* As the datasets are quite big (500mo to 1Go), they were not pushed to the repo in github. Specific information about the ratings and the profiles of the directors were taken from the database. The pre-processing is explained in data_preprocessing.ipynb. The readme.md of this dataset can be found in the data folder inside the IMDb folder.

## 4. Methods : 
### K-means unsuperviesed : 
* **K-means** : 
* **Elbow's method** :
* **Centroïds** :
### Network graphs and directed graphs : 
* **Basic Networks**: 
* **By parted directed graph** :
* **Eigen-vector centrality** :
* **Spring layout algorithm** :
* **Louvain algorithm** :
### Statistical tests :
* **t-Test** :
* **Chi2 test** :


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
│   │     └── data_preprocessing.ipynb       <- Notebook that reproduces all the preprocessing steps in order to get Actor.pkl and Movie.pkl (Merge CMU and IMDb)
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing Python dependencies
└── README.md
```

## 6. Proposed timeline : 
* Part 1 : week 8,9
*	Part 2 : week 10,11
*	Part 3 : week 12,13
* Part 4 : week 13,14

## 7. Organization within the team 
Each team member will be assigned a specific research question and the others will have to check on the other and give them neutral feedback. In addition to that, half of the team will directly begin working on homework 2 so that is done by next week. Finally, one or two people will have the responsibility to write the report with Jekyll while the others finish the analysis. 

