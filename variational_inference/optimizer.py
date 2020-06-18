import numpy as np

from variational_inference.distribution import NormalDistribution, CategoricalDistribution


class SimpleCAVI:
    def __init__(self, prior_distribution, ):
        self.q_mu = NormalDistribution(mu=np.random.normal(0, 1, size), sigma_sq=np.random.normal(1, 1, size))
        self.q_c = CategoricalDistribution(probas=[1/size]*size)
