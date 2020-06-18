import abc
from enum import Enum
from typing import List
import numpy as np

from variational_inference.utils.descriptors import MutableAttribute


class DistributionType(Enum):
    NORMAL_DISTRIBUTION = "normal_distribution"
    CATEGORICAL_DISTRIBUTION = "categorical_distribution"


class Distribution(abc.ABC):

    """
    An Interface for all Distributions.
    """

    @abc.abstractmethod
    def sample(self, samples: int):
        pass

    @abc.abstractmethod
    def get_parameters(self):
        pass

    @abc.abstractmethod
    def get_distribution_type(self):
        pass


class NormalDistribution(Distribution):
    sigma_sq = MutableAttribute()
    mu = MutableAttribute()

    def __init__(self, mu, sigma_sq):
        self.mu = mu
        self.sigma_sq = sigma_sq
        self.size = len(self.mu)

    def sample(self, samples: int) -> np.array:
        return np.random.normal(self.mu, self.sigma_sq, (samples, self.size))

    def get_parameters(self) -> List:
        return [self.mu, self.sigma_sq]

    def get_distribution_type(self) -> Enum:
        return DistributionType.NORMAL_DISTRIBUTION


class CategoricalDistribution(Distribution):
    probas = MutableAttribute()

    def __init__(self, probas):
        self.probas = probas

    def sample(self, samples: int) -> np.array:
        return np.random.multinomial(n=1, pvals=self.probas, size=samples)

    def get_parameters(self) -> List:
        return [self.probas]

    def get_distribution_type(self) -> Enum:
        return DistributionType.CATEGORICAL_DISTRIBUTION
