# Title : Evolution of the network of actors in the film industry through time 

## 1. Abstract :
This project aims to analyse and to map the evolution of the network of actors through time and  especially to determine the influence of different factors on the network. This interconnected system between actors, directors, films is interesting as it is governed and influenced by a multitude of factors such as ethnicity, movie genre, success, etc. In order to perform our analysis, we will first find and collect more data regarding film directors. After that, the network of actors and the network of directors will both be mapped and linked. A multitude of clusters could arise from those network and will be analysed. Moreover, different statistical analysis will be perform in order to see correlation between factors and to define the most influenced factors in the evolution of the network. By doing so, it is possible to conclude patterns across the different evolutions of each actors networks. 

## 2. Research Questions : 
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
IMDb database : Huge database of information about film (10’000’000 movies), directors, crew member
* Name.basics.tsv.gz => unique index of the name and the according name
* Title.basics.tsv.gz => unique index of  film and other information (we only use the index and name of the film)
* Title.crew.tsv.gz => unique index of film and unique index of directors and writers
* Title.ratings.tsv.gz => unique index of film and ratings, number of votes
* Most of those datasets contains irrelevant information for our analysis and hence were discard. This was motivated by the fact that each dataset were larger than 500mo and some more than Go. As we did not want to fill all the available space on Github and to load each time Go of data when we will use the python file, we decided to drop entries in the different database.  

## 4. Methods :
* Mapping the whole network (different map), clustering method (Unsupervised learning, K-NN,…), Litterature review , statistics (T-test, Chi2, …)

## 5. Proposed timeline : 
Week by week till the last Milestone : 
* Week 8 : Pre-processing data, verify feasibility
* Week 9 : literature review on network, deliver milestone 2
* Week 10 : Half of the team on homework, other literature review, start mapping
* Week 11 : mapping the network, different versions
* Week 12 : Statistical hypothesis and computation
* Week 13 : Plot and redaction of Jekyll
* Week 14 : Redaction of jekyll and final touch

## 6. Organization within the team 
Faire un tableau ?
* Pierre : literature review of networks 
* Thibault : literature review of networks
* Thomas : Pre-processing of data
* Émile : 
* Alexandre :
* À faire : rédaction sur le site Jekill, homework 2 

## 7. Question TA (Optional) :
