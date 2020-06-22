from tqdm import tqdm

from variational_inference.distribution import CategoricalDistribution, NormalDistribution
from variational_inference.optimizer import SimpleCAVI
from variational_inference.datagenerator.GMM import SimpleGMM


# ========= Generating some Test Data
generative_mixture_distribution = NormalDistribution(mu=[0, 0, 0, 0], sigma_sq=[1, 10, 1, 10])
generative_mixture_assignments = CategoricalDistribution(probas=[1/4]*4)
data_gen_simpleGMM = SimpleGMM(mixture_distribution=generative_mixture_distribution,
                               mixture_assignments=generative_mixture_assignments)
data = data_gen_simpleGMM.generate_data(2000)


# ======== Defining a Prior over the given Data =====
mixture_distribution = NormalDistribution(mu=[0, 0, 0, 0], sigma_sq=[1, 1, 1, 1])
mixture_assignments = CategoricalDistribution(probas=[1/4]*4)
prior_model = SimpleGMM(mixture_distribution=mixture_distribution, mixture_assignments=mixture_assignments)

# Handing over the prior to the Optimizer with the Data
simple_cavi = SimpleCAVI(probabilistic_model=prior_model, data=data["x"])

iterations = 200
for iteration in tqdm(range(iterations)):

    q_mu, phi = simple_cavi.infer()

print(f"Real Params:\n{data_gen_simpleGMM.mu_vals}")
print(f"Estimated Params: {q_mu}")
print(phi)
