import RBM
from preprocessing_data import training_set
# Creation of the RBM object
# nv is a fixed parameters, here the number of movies / number of visible no-
# -des
nv = len(training_set[0])

# nh the number of hidden nodes, a number we choose.
# We have 1682 visible nodes. nh will be the possible features detected, as
# actors, oscar, director, etc. nh corresponds to the number of features we
# want to detect. 100 is a good start. nh is totally tunable.
nh = 100

# We need now the batch size. Another tunable parameter.
batch_size = 100

#Creating the RBM
rbm = RBM.RBM(nv, nh)