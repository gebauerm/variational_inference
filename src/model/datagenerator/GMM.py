from typing import List, Dict, Union
import numpy as np

from src.utils.descriptors import NotNoneAttribute, NonNegativeAttribute


class SimpleGMM:
    cluster = NotNoneAttribute()
    instances_num = NotNoneAttribute()
    mu_sigma_sq = NonNegativeAttribute()
    x_sigma_sq = NonNegativeAttribute()

    def __init__(self, cluster: int = None,
                 mu_sigma_sq: float = 1,
                 instances_num: int = None,
                 mu_mean: float = 0,
                 c_proba:List = None,
                 x_sigma_sq: float = 1):
        """
        This class is a data generator, which uses a gaussian mixture model. The generated data is defined as "x" and
        its probability distribution is P(x| mu, c). Values for "x" are drawn from that distribution, which is a normal
        distribution with: N(c*mu, 1). "Mu"" are mean values of size "cluster", which are drawn as well from a
        normal distribution P(mu), which is has the following hyperparameters N(mu_mean, sigma_sq). "c" is a one-hot
        encoded vector of size "cluster", which is drawn from a categorical distribution P(c). "c" manages the cluster
        assignments for each value of "x".
        For more explanation please take a look at: https://arxiv.org/pdf/1601.00670.pdf
        Args:
            cluster: this is the number of clusters k
            sigma_sq: the variance of P(mu)
            instances_num: the number of realisations for x
            mu_mean: the expected value of P(mu)
            c_proba: the probability distribution of cluster assignments
            x_sigma_sq: the variance of P(x| mu, c)
        """

        # managed attributes
        self.cluster = cluster
        self.instances_num = instances_num
        self.mu_sigma_sq = mu_sigma_sq
        self.x_sigma_sq = x_sigma_sq

        # unmanaged attributes
        self.mu_mean = mu_mean
        self.c_proba = np.array([1/self.cluster]*self.cluster) if c_proba is None else np.array(c_proba)

        # storage values
        self.mu_vals = None

    def generate_data(self) -> Dict[str, Union[List, List[str]]]:
        """
        Generates data from the given hyperparameters with a gaussian mixture model.
        Returns:

        """
        # construct probability model
        self.mu_vals = np.random.normal(self.mu_mean, self.mu_sigma_sq, self.cluster)
        c = np.random.multinomial(n=1, pvals=self.c_proba, size=self.instances_num)
        distribution_labels = [f"dist_{np.where(c[row, :] == 1)[0][0]}" for row in range(c.shape[0])]
        x_mu = c.dot(self.mu_vals)
        x = [np.random.normal(mu_value, self.x_sigma_sq, 1)[0] for mu_value in x_mu]
        generated_data = {
            "x": x,
            "distribution_labels": distribution_labels
        }
        return generated_data
