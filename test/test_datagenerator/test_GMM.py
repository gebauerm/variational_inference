import pytest

from variational_inference.datagenerator.GMM import SimpleGMM
from variational_inference.distribution import NormalDistribution, CategoricalDistribution


class TestGMM:

    def test_generate_data(self, gmm_fixture):
        sample_gmm = gmm_fixture
        generated_data = sample_gmm.generate_data(samples=10)
        assert generated_data is not None
        assert type(generated_data) is dict


if __name__ == "__main__":
    pytest.main()
