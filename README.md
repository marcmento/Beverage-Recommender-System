# Beverage Recommender System
In this project I created and trained multiple machine learning models that recommend beverages to certain users based on previous reviews. In this project I used 3 distinct algorithms to make 4 basic models, the algorithms are:
* SVD
* NMF 
* KNN
* Ensemble (Mix of previous three)

I then created an advanced model which used the LightFM library to make a model that takes in specific user and item feature data to help make better recommendations. For this advanced model some of the features included were:
* Reviewers’ genders
* Beverage ABV
* Beverage name
* Beverage type

## Results
After running each model on my test data I produced 5 result files that give a recommendation score from 0 - 5 with 5 being highly recommend. Results have been included. 

## Run
To run these 5 models please ensure you have training, validation and test data in the same directory as the python file. For privacy reasons, I have not included my used data sets.
