This README describes data in the IMDb folder. Data obtained from the website IMDb which reference almost all of the film made. Their dataset contains information about film, movie director, box office, ratings, crew members, origines and much more.


Link : https://datasets.imdbws.com

Explanation : https://developer.imdb.com/non-commercial-datasets/

###
#
# DATA
#
###

1. title.crew.tsv.gz [363.3 mo]

Contains the following :

- tconst (string) - alphanumeric unique identifier of the title
- directors (array of nconsts) - director(s) of the given title
- writers (array of nconsts) – writer(s) of the given title


2. title.ratings.tsv.gz [26 mo]

Contains the following :

- tconst (string) - alphanumeric unique identifier of the title
- averageRating – weighted average of all the individual user ratings
- numVotes - number of votes the title has received