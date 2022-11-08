# Testing the RBM
from creating_rbm import *
from preprocessing_data import test_set

test_loss = 0

# We need a counter, in order to normalize the train_loss
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]

    if len(vt[vt>=0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(
            torch.abs(vt[vt>0] - v[vt>0])
        )
        s += 1.
print(f'test loss: {test_loss/s}')
