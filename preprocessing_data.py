# IMPORTING THE LIBRAIRIES
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# IMPORTING THE DATASET
movies = pd.read_csv(
    'data/ml-1m/movies.dat',
    sep='::',
    header=None,
    engine='python',
    encoding='latin-1'
)

users = pd.read_csv(
    'data/ml-1m/users.dat',
    sep='::',
    header=None,
    engine='python',
    encoding='latin-1'
)

# Ratings structure
#   - first column  = user id
#   - second column = movie id
#   - third column  = rating value
ratings = pd.read_csv(
    'data/ml-1m/ratings.dat',
    sep='::',
    header=None,
    engine='python',
    encoding='latin-1'
)

# PREPARING THE TRAINING SET AND THE TEST SET
training_set = pd.read_csv(
    'data/ml-100k/u1.base',
    delimiter='\t'
)
# Convert to an array
#   dtype='int' means that we want all values in the dataset converting into integer
training_set = np.array(
    training_set,
    dtype='int'
)

test_set = pd.read_csv(
    'data/ml-100k/u1.base',
    delimiter='\t'
)
# Convert to an array
#   dtype='int' means that we want all values in the dataset converting into integer
test_set = np.array(
    test_set    ,
    dtype='int'
)

# Getting the number of users and movies
nb_users = max(
    max(training_set[:, 0]),    #all lines, first column
    max(test_set[:, 0])
)

nb_movies = max(
    max(training_set[:, 1]),    #all lines, second column
    max(test_set[:, 1])
)

# Converting the data into an array with users in line and movies in column
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]  #Get all the movies ids
                                                        # for the user id
        id_ratings = data[:, 2][data[:, 0] == id_users] # Get all the ratings
                                                        # for the user id

        # We create now a list of 1682 elements (as there are 1682 movies)
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)
# The row i of training_set and test_set will contain a list of the ratings
# of the user id i. The index j in the list is the movie id.

# Converting the data into Torch tensors
## The test set gonna be one torch tensor, and the training set another one.
training_set = torch.FloatTensor(training_set)  #Expect a list of list, that's
                                                # why we use "list(ratings)"
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings : 1 = liked, 0 = Not Liked
training_set[training_set == 0] = -1    #Set the value of not rated movie at -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1    #Set the value of not rated movie at -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1