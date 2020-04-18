# Reference: https://arxiv.org/pdf/1601.00670.pdf
# https://ssl2.cms.fu-berlin.de/ewi-psy/einrichtungen/arbeitsbereiche/computational_cogni_neurosc/PMFN/15-Variational-inference.pdf
import numpy as np
import math
from math import log2
from scipy.stats import norm, uniform
from scripts.utils import plot_elbo

# TODO: find errors in the ELBO
# TODO: fix the variance estimation issue

# ============================= Generate Data =============================

# ===================== Distribution P ===================
# hyperparameters of P(mu)
mu_mean = 0
sigma = 1 # this variance is known
k = 4

# number of instances
i = 2000
c_probas = [1/k]*k

# construct probability model
mu = np.random.normal(mu_mean, sigma, k)
c = np.random.multinomial(n=1, pvals=c_probas, size= i)
x_mu = c.dot(mu)
x = np.random.normal(x_mu, 1)

# optimization params
iterations = 200

# ================== Initialize Variational Parameters =================
# Hypterparameter for Q(mu| m_q, s_q)
mu_est = np.random.normal(0, 1, k)
sigma_est = np.random.normal(0, 1, 1)

m_q = np.ones(k)
s_q=1


# Hyperparameter for Q(c)
phi_q_logit = np.random.uniform(0, 1, (i,k))
phi_q = np.apply_along_axis(lambda x: x/x.sum(), 1, phi_q_logit)
c_est = phi_q

# memory
elbo_vals=[]

# Start optimization
for iter in range(iterations):
    # set phi_q for every kth entry
    for idx, x_i in enumerate(x):
        phi_row = np.ones(k)
        for idk in range(k):
            phi_row[idk] = math.exp(mu_est[idk]*x_i - (mu_est[idk]**2 + sigma_est)/2)
        phi_q[idx] = phi_row/phi_row.sum()

    for idk in range(k):
        # calculate m_q for every k
        sum_nominator = sum([phi_q[idx][idk]*x[idx] for idx, x_idx in enumerate(x)]) # check if convergence doesnt work
        sum_denominator = sum([phi_q[idx][idk] for idx, x in enumerate(x)])
        m_q[idk] = sum_nominator/(1/sigma + sum_denominator)

        # calculate s_q for every k
        s_q = 1/(1/sigma + sum_denominator)

    # ==== compute elbo
    # First Component sum_k(E[log p(mu_k)| m_k, s_k])
    e_p_mu_k = sum([-mu_est[idk]**2 - sigma_est for idk in range(k)])
    # Second Component e_log_p(ci)
    e_log_p_c = -i*math.log(k)
    # Third Component e_log_p(xi|ci_mu)
    e_log_x = sum([sum([phi_q[idx][idk]*x[idx]*mu_est[idk] - phi_q[idx][idk]*mu_est[idk]**2/2 for idx, _ in enumerate(x)])
                   for idk in range(k)])
    # Forth Component e_log_q(c_i | phi_i)
    e_log_p_ci = sum([sum([phi_q[idx][idk]*math.log(phi_q[idx][idk]) for idk in range(k)]) for idx in range(i)])
    # Fith Component e_log_q_mu_k|m_k,s_k)
    e_log_mu_k_q = sum([(math.log(2*math.pi) + sigma_est)/2 for idk in range(k)]) # here it might be one according to princton
    # compute elbo
    elbo = e_p_mu_k + e_log_p_c+e_log_x - e_log_p_ci -e_log_mu_k_q
    elbo_vals.append(elbo)
    print(elbo)
    # get parameters of P
    mu_est = m_q
    sigma_est = s_q

    if iter>1:
        if np.abs(elbo_vals[iter-1] - elbo) < 0.01:
            break

fig_elbo = plot_elbo(elbo_vals)
fig_elbo.show()
print(f"Estimated Mean Vals: {mu_est}")
print(f"Actual Mean Vals: {mu}")
print(f"Estimated Sigma: {sigma_est}")
print(f"Real Sigma: {sigma}")

