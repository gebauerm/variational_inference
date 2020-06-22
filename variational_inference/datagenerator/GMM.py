from typing import List, Dict, Union
import numpy as np

from variational_inference.utils.descriptors import NotNoneAttribute, NonNegativeAttribute
from variational_inference.distribution import CategoricalDistribution, NormalDistribution

# TODO: incorporate distribution classes and think of a interface


class SimpleGMM:
    x_sigma_sq = NonNegativeAttribute()
    mixture_distribution = NotNoneAttribute()
    mixture_assignments = NotNoneAttribute()

    def __init__(self,
                 mixture_distribution: NormalDistribution = None,
                 mixture_assignments: CategoricalDistribution = None,
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

        # latent distributions
        self.mixture_distribution = mixture_distribution
        self.mixture_assignments_dist = mixture_assignments

        # observation_distribution
        self.x_sigma_sq = x_sigma_sq

    def generate_data(self, samples: int) -> Dict[str, Union[List, List[str]]]:
        """
        Generates data from the given hyperparameters with a gaussian mixture model.
        Returns:

        """
        mu_vals = self.mixture_distribution.sample(samples=1)
        c = self.mixture_assignments_dist.sample(samples=samples)
        distribution_labels = [f"dist_{np.where(c[row, :] == 1)[0][0]}" for row in range(c.shape[0])]
        x_mu = c.dot(mu_vals.T)
        x = [np.random.normal(mu_value, self.x_sigma_sq, 1) for mu_value in x_mu]
        generated_data = {
            "x": x,
            "distribution_labels": distribution_labels
        }
        return generated_data
