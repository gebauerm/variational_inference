import pytest

from variational_inference.datagenerator.GMM import SimpleGMM
from variational_inference.distribution import NormalDistribution, CategoricalDistribution


class TestGMM:

    @pytest.fixture
    def gmm_fixture(self):
        mixture_distribution = NormalDistribution(mu=[0, 0], sigma_sq=[1])
        mixture_assignments = CategoricalDistribution(probas=[1/2, 1/2])
        simple_gmm= SimpleGMM(mixture_distribution=mixture_distribution, mixture_assignments=mixture_assignments)
        return simple_gmm

    def test_generate_data(self, gmm_fixture):
        sample_gmm = gmm_fixture
        generated_data = sample_gmm.generate_data(samples=10)
        assert generated_data is not None
        assert type(generated_data) is dict


if __name__ == "__main__":
    pytest.main()
