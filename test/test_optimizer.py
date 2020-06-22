import pytest

from variational_inference.distribution import CategoricalDistribution, NormalDistribution
from variational_inference.optimizer import SimpleCAVI
from variational_inference.datagenerator.GMM import SimpleGMM


class TestSimpleCAVI:
    def test_infer(self, gmm_fixture):
        data_gen_simpleGMM = gmm_fixture
        test_data = data_gen_simpleGMM.generate_data(1000)

        mixture_distribution = NormalDistribution(mu=[0, 0], sigma_sq=[1, 1])
        mixture_assignments = CategoricalDistribution(probas=[1 / 2, 1 / 2])
        prior_model = SimpleGMM(mixture_distribution=mixture_distribution, mixture_assignments=mixture_assignments)
        simple_cavi = SimpleCAVI(probabilistic_model=prior_model, data=test_data["x"])

        est_mu, phi = simple_cavi.infer()

        assert est_mu is not None
        assert phi is not None
        for phi_row in phi:
            assert round(sum(phi_row), 2) == 1


if __name__ == "__main__":
    pytest.main()

