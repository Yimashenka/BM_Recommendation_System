# CREATING THE ARCHITECTURE OF THE NN
import torch
class RBM():
    # nv = Number of visible nodes
    # nh = Number of hidden nodes
    def __init__(self, nv, nh):
        # Initialising the Weights randomly (normal distribution), a tensor of
        # size (nh, nv)
        self.W = torch.randn(nh, nv)
        # Initialising the biais : probability of the hidden nodes given the
        # visible ones.
        self.a = torch.randn(1, nh)
        # Initialising the biais : probability of the visible nodes given the
        # hidden ones.
        self.b = torch.randn(1, nv)