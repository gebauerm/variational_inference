# Reference: https://arxiv.org/pdf/1601.00670.pdf
# https://ssl2.cms.fu-berlin.de/ewi-psy/einrichtungen/arbeitsbereiche/computational_cogni_neurosc/PMFN/15-Variational-inference.pdf
# https://zhiyzuo.github.io/VI/
# http://www.cs.uoi.gr/~arly/papers/SPM08.pdf
# import numpy as np
# import math
# from scripts.ML_Community.utils import *
# import pandas as pd
#
# import seaborn as sns
#
# # TODO: find errors in the ELBO
# # TODO: fix the variance estimation issue
#
# # ============================= Generate Data =============================
#
# # ===================== Distribution P ===================
#
# # hyperparameters of P(mu)
# k = 4
# mu_mean = 0
# sigma = np.array([5]*k) # this variance is known
#
#
# # number of instances
# i = 5000
# c_probas = np.array([1/k]*k)
#
# # construct probability model
# mu = np.random.normal(mu_mean, sigma, k)
# c = np.random.multinomial(n=1, pvals=c_probas, size= i)
# distribution_labels = [f"dist_{np.where(c[row, :] == 1)[0][0]}" for row in range(c.shape[0])]
# x_mu = c.dot(mu)
# x = np.array([np.random.normal(mu_value, 1, 1)[0] for mu_id, mu_value in enumerate(x_mu)])
#
# # ================== Initialize Variational Parameters =================
# iterations = 200
# # Define Prior for P
# mu_p = 1
# sigma_p = sigma
#
#
# # Hypterparameter for Q(mu| m_q, s_q)
# m_q = np.random.normal(0, 1, k)
# s_q = np.ones(k)
#
#
# # Hyperparameter for Q(c)
# phi_q_logit = np.random.uniform(0, 1, (i,k))
# phi_q = np.apply_along_axis(lambda x: x/x.sum(), 1, phi_q_logit)
# c_est = phi_q
#
# # memory
# elbo_vals=[]
#
# # Start optimization
# for iter in range(iterations):
#     # set phi_q for every kth entry
#     for idx, x_i in enumerate(x):
#         phi_row = np.ones(k)
#         for idk in range(k):
#             phi_row[idk] = math.exp(m_q[idk]*x_i - (m_q[idk]**2 + 2*s_q[idk])/2)
#         phi_q[idx] = phi_row/phi_row.sum()
#
#     for idk in range(k):
#         # calculate m_q for every k
#         sum_nominator = sum([phi_q[idx][idk]*x[idx] for idx, x_idx in enumerate(x)]) # check if convergence doesnt work
#         sum_denominator = sum([phi_q[idx][idk] for idx, x in enumerate(x)])
#         m_q[idk] = sum_nominator/(1/sigma_p[idk] + sum_denominator)
#
#         # calculate s_q for every k
#         s_q[idk] = 1/(1/sigma_p[idk] + sum_denominator)
#
#     # ==== compute elbo
#     # First Component sum_k(E[log p(mu_k)| m_k, s_k])
#     e_p_mu_k = sum([m_q[idk]**2/sigma_p[idk] for idk in range(k)])
#     # Second Component e_log_p(ci)
#     e_log_p_c = -i*math.log(k)
#     # Third Component e_log_p(xi|ci_mu)
#     e_log_x = sum([sum([-phi_q[idx][idk]*x[idx]**2/2*sigma_p[idk]+phi_q[idx][idk]*x[idx]*m_q[idk]/sigma_p[idk] - phi_q[idx][idk]*m_q[idk]**2/2*sigma_p[idk] for idx, _ in enumerate(x)])
#                    for idk in range(k)])
#     # Forth Component e_log_q(c_i | phi_i)
#     e_log_p_ci = sum([sum([phi_q[idx][idk]*math.log(phi_q[idx][idk]) for idk in range(k)]) for idx in range(i)])
#     # Fith Component e_log_q_mu_k|m_k,s_k)
#     e_log_mu_k_q = sum([math.log(s_q[idk])/2 for idk in range(k)]) # here it might be one according to princton
#     # compute elbo
#     elbo = e_p_mu_k + e_log_p_c+e_log_x - e_log_p_ci -e_log_mu_k_q
#     elbo_vals.append(elbo)
#     print(elbo)
#
#     if iter>1:
#         if np.abs(elbo_vals[iter-1] - elbo) < 0.01:
#             break
#
# fig_elbo = plot_elbo(elbo_vals, True, seaborn=False)
# fig_elbo.show()
# print(f"Estimated Mean Vals: {m_q}")
# print(f"Actual Mean Vals: {mu}")
#
# fig_2 = (x, targets=distribution_labels, mu_values=m_q)id
# fig_2.show()