import numpy as np
import math

from variational_inference.distribution import NormalDistribution, CategoricalDistribution
from variational_inference.datagenerator.GMM import SimpleGMM


class SimpleCAVI:
    def __init__(self, probabilistic_model: SimpleGMM, data):
        """
        A frst very simple class for the CAVI Algorithm deriven by Bishop
        :param probabilistic_model:
        :param data:
        """
        self.probabilistic_model = probabilistic_model
        self.data = data
        # TODO: necessary parameters need to be retreived independend of probability models
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
        Estimates values for all given parameters.
        :return:
        """
        self.calculate_phi()
        self.calcilate_q_mu()
        return self.q_mu.mu, self.phi

    def calculate_phi(self):
        for idx, x_i in enumerate(self.data):
            q_c_assignments_row = self.phi[idx, :]
            for idk in range(self.q_c.probas.shape[0]):
                q_c_assignments_row[idk] = math.exp(self.q_mu.mu[idk] * x_i -
                                                    (self.q_mu.mu[idk] ** 2 + 2 * self.q_mu.sigma_sq[idk]) / 2)
            self.phi[idx] = q_c_assignments_row / q_c_assignments_row.sum()

    def calcilate_q_mu(self):
        for idk in range(self.cluster):
            sum_nominator = sum(
                [self.phi[idx][idk] * self.data[idx] for idx, x_idx in enumerate(self.data)])
            sum_denominator = sum([self.phi[idx][idk] for idx, x in enumerate(self.data)])
            self.q_mu.mu[idk] = sum_nominator / (1 / self.probabilistic_model.x_sigma_sq + sum_denominator)

            self.q_mu.sigma_sq[idk] = 1 / (1 / self.probabilistic_model.x_sigma_sq + sum_denominator)
