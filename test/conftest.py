import pytest

from variational_inference.distribution import CategoricalDistribution, NormalDistribution
from variational_inference.datagenerator.GMM import SimpleGMM


@pytest.fixture
def gmm_fixture():
    mixture_distribution = NormalDistribution(mu=[0, 0], sigma_sq=[1, 10])
    mixture_assignments = CategoricalDistribution(probas=[1 / 2, 1 / 2])
    simple_gmm = SimpleGMM(mixture_distribution=mixture_distribution, mixture_assignments=mixture_assignments)
    return simple_gmm
