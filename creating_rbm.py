import torch

import RBM

from preprocessing_data import training_set, nb_users
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

# Creating the RBM
rbm = RBM.RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    # We need a loss function, compute the error between the prediction and
    # the real ratings. RMSE (most common one), simple distance, absolute dis-
    # -tance. We will use simple distance
    train_loss = 0

    # We need a counter, in order to normalize the train_loss
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)

        #Loops for the K steps of contrastive divergence
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(
            torch.abs(v0[v0>0] - vk[v0>0])
        )
        s += 1.
    print(f'epoch: {epoch}     loss: {train_loss/s}')
