import numpy as np
import math

from variational_inference.distribution import NormalDistribution, CategoricalDistribution
from variational_inference.datagenerator.GMM import SimpleGMM


class SimpleCAVI:
    """
    Probabilistic Optimization Algorithm for the Gaussian Mixtute Model Case.
    The Algorithm tries to find the parameters of the latent Gaussian Distribution.
    """
    def __init__(self, probabilistic_model: SimpleGMM, data):
        self.probabilistic_model = probabilistic_model
        self.data = data
        # TODO: necessary parameters need to be retrieved independent of probability models
        self.cluster = len(self.probabilistic_model.mixture_distribution.mu)

        # surrogate probability
        self.q_mu = NormalDistribution(mu=np.random.normal(0, 1, self.cluster),
                                       sigma_sq=np.random.normal(1, 1, self.cluster)**2)
        # TODO:  this needs more generalization it currently also makes no sense
        self.q_c = CategoricalDistribution(probas=np.array([1/self.cluster]*self.cluster))
        phi_q_logit = np.random.uniform(0, 1, (len(data), self.cluster))
        self.phi = np.apply_along_axis(lambda x: x / x.sum(), 1, phi_q_logit)

    def infer(self):
        """
        Starts the optimization procedure. For more info take a look at: https://arxiv.org/pdf/1601.00670.pdf
        :return:
        """
        for idx, x_i in enumerate(self.data):
            q_c_assignments_row = self.phi[idx, :]
            for idk in range(self.q_c.probas.shape[0]):
                q_c_assignments_row[idk] = math.exp(self.q_mu.mu[idk] * x_i -
                                                    (self.q_mu.mu[idk] ** 2 + 2 * self.q_mu.sigma_sq[idk]) / 2)
            self.phi[idx, :] = q_c_assignments_row / q_c_assignments_row.sum()

        for idk in range(self.cluster):
            # calculate m_q for every k
            sum_nominator = sum(
                [self.phi[idx][idk] * self.data[idx] for idx, x_idx in enumerate(self.data)])  # check if convergence doesnt work
            sum_denominator = sum([self.phi[idx][idk] for idx, x in enumerate(self.data)])
            self.q_mu.mu[idk] = sum_nominator / (1 / self.probabilistic_model.x_sigma_sq + sum_denominator)

            # calculate s_q for every k
            self.q_mu.sigma_sq[idk] = 1 / (1 / self.probabilistic_model.x_sigma_sq + sum_denominator)
        return self.q_mu.mu, self.phi
