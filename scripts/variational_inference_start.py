# Reference: https://arxiv.org/pdf/1601.00670.pdf
import numpy as np

# ============================= Generate Data =============================

# ===================== Distribution P ===================
# hyperparameters of P(mu)
mu_mean = 0
mu_variance = 10
k = 4

# number of instances
i = 100
c_probas = [0.5, 0.2, 0.2, 0.1]

mu = np.random.normal(mu_mean, mu_variance, k)
c = np.random.multinomial(n=1, pvals=c_probas, size= i)
x_mu = c.dot(mu)
x = np.random.normal(x_mu, 1)
print(x)

# ================== Set initial values =================

