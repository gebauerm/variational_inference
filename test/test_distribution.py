import pytest
import numpy as np

from variational_inference.distribution import NormalDistribution, CategoricalDistribution, DistributionType


class TestDistribution:

    @pytest.fixture(params=[
        (NormalDistribution, {"mu": [0], "sigma_sq": [1]}),
        (CategoricalDistribution, {"probas": [ 0.5, 0.5]})
    ])
    def get_distribution(self, request):
        distribution, parameters = request.param
        return distribution(**parameters)

# ================ Tests ==============

    def test_sample(self, get_distribution):
        distribution_obj = get_distribution
        samples = distribution_obj.sample(10)

        assert samples is not None
        assert type(samples) is type(np.zeros(1))

    def get_parameters(self, get_distribution):
        distribution_obj = get_distribution
        dist_parameters = distribution_obj.get_parameters()

        assert dist_parameters is not None
        assert type(dist_parameters) is list

    def test_distribution_type(self, get_distribution):
        distribution_obj = get_distribution
        dist_type = distribution_obj.get_distribution_type()

        assert dist_type is not None
        assert type(dist_type) is DistributionType


if __name__ == "__main__":
    pytest.main()
