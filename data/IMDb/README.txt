This README describes data in the IMDb folder. Data obtained from the website IMDb which reference almost all of the film made. Their dataset contains information about film, movie director, box office, ratings, crew members, origines and much more.


Link of the datasets : https://datasets.imdbws.com

Link of the explanation from IMDb : https://developer.imdb.com/non-commercial-datasets/

###
#
# DATA
#
###


1. title.basics.tsv.gz [ mo]

Contains the following : 

- tconst (string) - alphanumeric unique identifier of the title
- titleType (string) – the type/format of the title (e.g. movie, short, tvseries, tvepisode, video, etc)
- primaryTitle (string) – the more popular title / the title used by the filmmakers on promotional materials at the point of release
- originalTitle (string) - original title, in the original language
- isAdult (boolean) - 0: non-adult title; 1: adult title
- startYear (YYYY) – represents the release year of a title. In the case of TV Series, it is the series start year
- endYear (YYYY) – TV Series end year. '\N' for all other title types
- runtimeMinutes – primary runtime of the title, in minutes
- genres (string array) – includes up to three genres associated with the title


2. title.crew.tsv.gz [363.3 mo]

Contains the following :

- tconst (string) - alphanumeric unique identifier of the title
- directors (array of nconsts) - director(s) of the given title
- writers (array of nconsts) – writer(s) of the given title


3. title.ratings.tsv.gz [26 mo]

Contains the following :

- tconst (string) - alphanumeric unique identifier of the title
- averageRating – weighted average of all the individual user ratings
- numVotes - number of votes the title has received


4. name.basics.tsv.gz [ mo]

Contains the following :

- nconst (string) - index unique of the name
- primaryName (string) - name of the person
- birthYear (YYYY) - year of birth
- deathYear (YYYY) - year of death
- primaryProfession (array of strings) - profession such as actor, producer, director, ....
- knownForTitles (array of tconst) - film for which the person is known for

