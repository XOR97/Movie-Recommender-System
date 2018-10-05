# Movie Recommender System using Spark MLlib

Implemented a recommender system that suggests movies to any user based on their ratings. The dataset used is MovieLens 100k dataset.

Recommender System uses Item-based Collaborative Filtering approach. The script makes use of the MovieLens dataset to recommend movies to users who liked similar movies using item-item similarity score.

### Specifications

- Python 2.7.13
- PySpark 2.1.0
- AWS EMR Cluster
- AWS S3
- Spark's Alternating Least Squares algorithm
- [MovieLens dataset](https://grouplens.org/datasets/movielens/)

### Running the application

+ Clone the repo on local machine
+ Download the datasets
```
$ sh download_datasets.sh
```
+ Put the datasets on the Hadoop fs
```
$ hadoop fs -put /user/<user_name>/datasets
```

+ Run the application:
```
$ sh run.sh
```

+ Query the application
```
# get ratings for a user
curl http://0.0.0.0:5432/<user_id>/ratings/top/<count>

# add ratings for a new user for predictions
curl -X POST http://0.0.0.0:5432/<new_user_id>/ratings --data <movie_id,ratings>

# get the recommendations for new user
curl http://0.0.0.0:5432/<new_user_id>/ratings/top/<count>

```

We have used collaborative filtering to build a movie recommendation system using the alternating least squares implementation in Spark MLlib.  We will be using Python with flask framework to build a web application that gives an UI for our Spark model. The UI page allows the user to select the movie and the system provides recommendations based on the selection. Further, we will visualize our findings using a network interconnected graph to show the predicted ratings.

Spark MLlib library for Machine Learning provides a Collaborative Filtering implementation by using Alternating Least Squares. The implementation in MLlib has the following parameters:

	•	numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
	•	rank is the number of latent factors in the model.
	•	iterations is the number of iterations to run.
	•	lambda specifies the regularization parameter in ALS.
	•	implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
	•	alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations

We are using the [MovieLens Dataset](https://grouplens.org/datasets/movielens/) for training and testing our model.
The dataset consists of a number of csv files. Some of the columns present are movieId, userId, rating , title, genre etc. It consists of a total of 20 million ratings for 27,000 movies.

### Evaluate the model using RMSE

The use of RMSE is very common and it makes an excellent general purpose error metric for numerical predictions.

#### Root Mean Squared Error (RMSE)

RMSE is the square root of the average of squared errors. The effect of each error on RMSD is proportional to the size of the squared error; thus larger errors have a disproportionately large effect on RMSD. Consequently, RMSE is sensitive to outliers.
