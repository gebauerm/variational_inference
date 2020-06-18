import pytest

from variational_inference.model.datagenerator.GMM import SimpleGMM


class TestGMM:

    def test_initialization(self):
        sample_gmm = SimpleGMM(cluster=4,
                               instances_num=10)
        assert sample_gmm.mu_vals is None
        assert sample_gmm.c_proba is not None

    def test_generate_data(self):
        sample_gmm = SimpleGMM(cluster=4,
                               instances_num=10)
        generated_data = sample_gmm.generate_data()
        assert generated_data is not None
        assert type(generated_data) is dict
