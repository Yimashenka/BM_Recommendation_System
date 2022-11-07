# CREATING THE ARCHITECTURE OF THE NN
import torch
class RBM():
    # nv = Number of visible nodes
    # nh = Number of hidden nodes
    def __init__(self, nv, nh):
        # Initialising the Weights randomly (normal distribution), a tensor of
        # size (nh, nv)
        self.W = torch.randn(nh, nv)
        # Initialising the bias : probability of the hidden nodes given the
        # visible ones.
        self.a = torch.randn(1, nh)
        # Initialising the bias : probability of the visible nodes given the
        # hidden ones.
        self.b = torch.randn(1, nv)

    # sample_h : sampling the hidden nodes according to the probabilities
    # P(h) given v; h the hidden nodes, v the visible ones. It is nothing else
    # than the sigmoid activation function.
    # Why this function ? During the training, we will approximate the log
    # likelihood gradient through the Gibbs sampling. To apply Gibbs sampling,
    # we need to compute the probabilities of the hidden nodes given the visi-
    # -ble nodes and use these probabilities to sample the activations of the
    # hidden nodes.
    # x : correponds to the visible neurons v, in the probabilities P(h) given
    # v.
    def sample_h(self, x):
        # Compute the probability of h given v : the sigmoid activation fun-
        # -ction applied to W.x
        wx = torch.mm(x, self.W.t())

        # We want to make sure that the bias is applied to each line of the
        # mini batch. We need to use a function adding a new dimension :
        # expand_as()
        activation = wx + self.a.expand_as(wx)

        # Now we can compute the activation function, here the probability
        # that the hidden node will be activated according to the value of
        # the visible node.
        p_h_given_v = torch.sigmoid(activation)

        # Final step : return not only this probability, but of course, a sam-
        # -ple of h.
        # IMPORTANT : We are making a Bernoulli RBM, because we're just predi-
        # -cting a binary outcome, that is that the user like yes or no a mo-
        # -vie, so wez predict 0 or 1.
        # p_h_given_v is a vector of nh elements, containing the probability
        # that the hidden node corresponding is activated.
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    # sample_v : the same as sample_h but for visible nodes given the hidden
    # ones.
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    # Constrative Divergence, see paper in doc, k-step constrative divergence)
    # v0 : input vectors containing the ratings of all the movies by one user
    # vk : visible nodes obtained after k samplings
    # ph0 : vector of probabilities that at the first iteration, the hidden
    # nodes equal one given the value of v0
    # phk : the probabilities of the hidden nodes after k samplings given the
    # value of the visible nodes vk
    def train(self, v0, vk, ph0, phk):
        # First update : the weights
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)

        # Second update : b
        self.b += torch.sum((v0 - vk), 0)   #just to keep the format of b

        # Third update : a
        self.a += torch.sum((ph0 - phk), 0) #just to keep the format of a
