import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import bernoulli
import pandas as pd
import pymc3 as pm
import theano.tensor as T
import operator
from scipy.stats import norm
from scipy.special import expit
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(1234)

"""Generate Obserevational and experimental mixed data X,Y binary and Z1,Z2 ~Normal Distributed.
- Find Markov Boundary from observational data by calculate Marginal Likelihood P(Do) = integral[ P(Do|a,b)*P(a,b) dadb]
- Find IMB~~=CMB in Experimental Data, by calculate the marginal likelihood with flat prior and with prior the posterior 
calculated before in observational data.The subset that will have marginal with post> marginal with flat will be IMB.
"""

"""Generate observational data"""

def Generate_Observational_Data(sample_size):

    alpha = 1
    beta = [1.3, 1.25, 1.4, 0.15]
    e = np.random.normal(1, 0, sample_size)  # noise

    Z1 = np.random.normal(0, 15, sample_size)
    Z2 = np.random.normal(0, 10, sample_size)
    μ_true = beta[0] * Z1 + beta[1] * Z2
    p_true = expit(μ_true)
    X = bernoulli.rvs(p_true)
    μ_true1 = alpha + beta[2] * X + beta[3] * Z2 + e
    p_true1 = expit(μ_true1)
    Y = bernoulli.rvs(p_true1)

    data = pd.DataFrame({"Z1": Z1, "Z2": Z2, 'X': X, "Y": Y})

    return data


No = 100
Ne = 50
mcmc_sample = 1000
data = Generate_Observational_Data(No)


def logistic(l):
    return 1 / (1 + T.exp(-l))

"1 case: Regression for (Z1,Z2,X),  Y = b0*Z1 + b1*Z2 +b2*X"

bayes_log_reg_model_C1 = pm.Model()

with bayes_log_reg_model_C1:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=3)

    # Probability of Expected value of outcome
    p = alpha + beta[0]*data.Z1 + beta[1] * data.Z2 + beta[2] * data.X

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Bernoulli('Y_obs', p=logistic(p), observed=data.Y)
    trace1 = pm.sample_smc(mcmc_sample, parallel=True) # posterior sampling f(θ|Do)
    # prior_trace1 = pm.sample_prior_predictive(mcmc_sample, random_seed=42)  #prior sampling

"""Estimate a,b0,b1,b2 to verify that our model is ok"""
map_estimate = pm.find_MAP(model=bayes_log_reg_model_C1)
print(map_estimate)
z
"""Calculate_Log_Likelihood_by_Hand1: Calculate Likelihood without pymc"""
def Calculate_Likelihood_by_Hand1(trace1,data):
    P = 0
    for dok in range(len(trace1['alpha'])):
        μ_D0_theta = trace1['alpha'][dok] + trace1['beta'][:, 0][dok] * data.Z1.values + trace1['beta'][:, 1][
            dok] * data.Z2.values \
                     + trace1['beta'][:, 2][dok] * data.X.values
        P_D0_theta = expit(μ_D0_theta)

        Y = data.Y
        for i in range(len(P_D0_theta)):
            if Y[i] == 1:
                P = P + math.log(P_D0_theta[i])
            elif Y[i] == 0:
                P = P + math.log(1 - P_D0_theta[i])
    Log_Likelihood1 = P/len(trace1['alpha'])

    return Log_Likelihood1

"""Calculate pdf of prior trace"""
f_a = norm(0, 10).logpdf(trace1['alpha'])
f_b0 = norm(0, 10).logpdf(trace1['beta'][:, 0])
f_b1 = norm(0, 10).logpdf(trace1['beta'][:, 1])
f_b2 = norm(0, 10).logpdf(trace1['beta'][:, 2])
log_prior1 = (sum(f_a) + sum(f_b0) + sum(f_b1) + sum(f_b2))/len(trace1['alpha'])

Log_Likelihood1 = Calculate_Likelihood_by_Hand1(trace1, data)
print("Likelihood1", Log_Likelihood1)
print("Prior1", log_prior1)
Posterior1 = Log_Likelihood1 + log_prior1
print("Log_Marginal1", Posterior1)

"""The Marginal Likelihood integral(P(Do|θ)*P(θ)dθ)"""
Marginal_Likelihood1 = trace1.report.log_marginal_likelihood
print("Marginal1 from pySMC", Marginal_Likelihood1)

"2 case: Regression for (Z1,X),  Y = b0*Z1 +b2*X"

bayes_log_reg_model_C2 = pm.Model()

with bayes_log_reg_model_C2:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)

    # Expected value of outcome
    mu = alpha + beta[0]*data.Z1 + beta[1] * data.X

    # Likelihood (sampling distribution) of observations
    Y_obs2 = pm.Bernoulli('Y_obs2', p=logistic(mu), observed=data.Y)
    trace2 = pm.sample_smc(mcmc_sample, parallel=True)
    # prior_trace2 = pm.sample_prior_predictive(mcmc_sample, random_seed=42)  # prior sampling

map_estimate2 = pm.find_MAP(model=bayes_log_reg_model_C2)
print(map_estimate2)

# with bayes_log_reg_model_C2:
#     az.plot_trace(trace2)
# plt.show()

def Calculate_Likelihood_by_Hand2(trace2,data):

    P = 0
    for dok in range(len(trace2['alpha'])):

        μ_D1_theta = trace2['alpha'][dok] + trace2['beta'][:, 0][dok] * data.Z1.values \
                     + trace2['beta'][:, 1][dok] * data.X.values
        P_D1_theta = expit(μ_D1_theta)

        Y = data.Y
        for i in range(len(P_D1_theta)):
            if Y[i] == 1:
                P = P + math.log(P_D1_theta[i])
            elif Y[i] == 0:
                P = P + math.log(1-P_D1_theta[i])
    Log_Likelihood2 = P/len(trace2['alpha'])

    return Log_Likelihood2

Log_Likelihood2 = Calculate_Likelihood_by_Hand2(trace2,data)
print("Likelihood2", Log_Likelihood2)

"""Calculate pdf of prior trace"""
f_a2 = norm(0, 10).logpdf(trace2['alpha'])
f_b02 = norm(0, 10).logpdf(trace2['beta'][:,0])
f_b12 = norm(0, 10).logpdf(trace2['beta'][:,1])
log_prior2 = (sum(f_a2) + sum(f_b02) + sum(f_b12))/len(trace2['alpha'])

print("Prior2", log_prior2)
Posterior2 = Log_Likelihood2 + log_prior2
print("Log_Marginal2", Posterior2)

"""The Marginal Likelihood integral(P(Do|θ)*P(θ)dθ)"""
Marginal_Likelihood2 = trace2.report.log_marginal_likelihood
print("Marginal2 from pySMC", Marginal_Likelihood2)

"3 case: Regression for (Z2,X),  Y = b1*Z2 +b2*X"

bayes_log_reg_model_C3 = pm.Model()

with bayes_log_reg_model_C3:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)

    # Expected value of outcome
    mu = alpha + beta[0]*data.Z2 + beta[1] * data.X

    # Likelihood (sampling distribution) of observations
    Y_obs3 = pm.Bernoulli('Y_obs3', p=logistic(mu), observed=data.Y)
    trace3 = pm.sample_smc(mcmc_sample, parallel=True)
    # prior_trace3 = pm.sample_prior_predictive(mcmc_sample, random_seed=42)  # prior sampling

# with bayes_log_reg_model_C3:
#     az.plot_trace(trace3)
# plt.show()

def Calculate_Likelihood_by_Hand3(trace3,data):

    P = 0
    for dok in range(len(trace3['alpha'])):

        μ_D2_theta = trace3['alpha'][dok] + trace3['beta'][:, 0][dok] * data.Z2.values \
                     + trace3['beta'][:, 1][dok] * data.X.values
        P_D2_theta = expit(μ_D2_theta)

        Y = data.Y
        for i in range(len(P_D2_theta)):
            if Y[i] == 1:
                P = P + math.log(P_D2_theta[i])
            elif Y[i] == 0:
                P = P + math.log(1 - P_D2_theta[i])
    Log_Likelihood3 = P/len(trace3['alpha'])

    return Log_Likelihood3
Log_Likelihood3 = Calculate_Likelihood_by_Hand3(trace3,data)
print("Log_Likelihood3", Log_Likelihood3)

"""Calculate pdf of prior trace"""
f_a3 = norm(0, 10).logpdf(trace3['alpha'])
f_b03 = norm(0, 10).logpdf(trace3['beta'][:, 0])
f_b13 = norm(0, 10).logpdf(trace3['beta'][:, 1])
log_prior3 = (sum(f_a3) + sum(f_b03) + sum(f_b13))/len(trace3['alpha'])

print("Prior3", log_prior3)
Posterior3 = Log_Likelihood3 + log_prior3
print("Log_Marginal3", Posterior3)

"""The Marginal Likelihood integral(P(Do|θ)*P(θ)dθ)"""
Marginal_Likelihood3 = trace3.report.log_marginal_likelihood
print("Marginal3 from pySMC",Marginal_Likelihood3)

"4 case: Regression for (X),  Y = b2*X"

bayes_log_reg_model_C4 = pm.Model()

with bayes_log_reg_model_C4:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=1)

    # Expected value of outcome
    mu = alpha + beta[0]*data.X

    # Likelihood (sampling distribution) of observations
    Y_obs4 = pm.Bernoulli('Y_obs4', p=logistic(mu), observed=data.Y)
    trace4 = pm.sample_smc(mcmc_sample, parallel=True)
    # prior_trace4 = pm.sample_prior_predictive(mcmc_sample, random_seed=42)  # prior sampling

map_estimate4 = pm.find_MAP(model=bayes_log_reg_model_C4)
print(map_estimate4)

def Calculate_Likelihood_by_Hand4(trace4,data):

    P = 0
    for dok in range(len(trace4['alpha'])):

        μ_D3_theta = trace4['alpha'][dok] + trace4['beta'][:, 0][dok] * data.X.values
        P_D3_theta = expit(μ_D3_theta)

        Y = data.Y
        for i in range(len(P_D3_theta)):
            if Y[i] == 1:
                P = P + math.log(P_D3_theta[i])
            elif Y[i] == 0:
                P = P + math.log(1 - P_D3_theta[i])
    Log_Likelihood4 = P / len(trace4['alpha'])

    return Log_Likelihood4

Log_Likelihood4 = Calculate_Likelihood_by_Hand4(trace4, data)
print("Log_Likelihood4", Log_Likelihood4)

"""Calculate pdf of prior trace"""
f_a4 = norm(0, 10).logpdf(trace4['alpha'])
f_b04 = norm(0, 10).logpdf(trace4['beta'][:, 0])
log_prior4 = (sum(f_a4) + sum(f_b04)) / len(trace4['alpha'])

print("Prior4", log_prior4)
Posterior4 = Log_Likelihood4 + log_prior4
print("Log_Marginal4", Posterior4)

"""The Marginal Likelihood integral(P(Do|θ)*P(θ)dθ)"""
Marginal_Likelihood4 = trace4.report.log_marginal_likelihood
print("Marginal4 from pySMC", Marginal_Likelihood4)

"""Summary"""
print("For set Z = (Z1,Z2,X) as MB, the Marginal Likelihood calculated by hand is P(Do) = ", Posterior1)
print("For set Z = (Z1,X) as MB, the Marginal Likelihood calculated by hand is P(Do) = ", Posterior2)
print("For set Z = (Z2,X) as MB, the Marginal Likelihood calculated by hand is P(Do) =  ", Posterior3)
print("For set Z = (X) as MB, the Marginal Likelihood calculated by hand is P(Do) =  ", Posterior4)


MB_Scores = {}
MB_Scores[('Z1', 'Z2', 'X')] = Posterior1
MB_Scores[('Z1', 'X')] = Posterior2
MB_Scores[('Z2', 'X')] = Posterior3
MB_Scores['X'] = Posterior4


MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
print('The set that is the Markov Boundary in observational data is:', MB_Do)

"""Generate Experimental data"""

def Generate_Experimental_Data(sample_size):
    alpha = 1
    beta = [1.4, 0.15]
    e = np.random.normal(1, 0, sample_size)  # noise

    Z2 = np.random.normal(0, 10, sample_size)
    X = bernoulli.rvs(0.5) #Treatment  randomized controlled trial(RTC)
    μ_true1 = alpha + beta[0] * X + beta[1] * Z2 + e
    p_true1 = expit(μ_true1)
    Y = bernoulli.rvs(p_true1)

    data_ex = pd.DataFrame({"Z2": Z2, 'X': X, "Y": Y})

    return data_ex

data_ex = Generate_Experimental_Data(Ne)

"""For every subset of MB check if it is IMB with flat and 'MB's posterior as prior in Marginal Likelihood"""

"""FOR IMB,CMB be (Z2,X)"""
bayes_log_reg_model_exp = pm.Model()

with bayes_log_reg_model_exp:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=2)

    # Expected value of outcome
    mu = alpha + beta[0]*data_ex.Z2 + beta[1] * data_ex.X

    # Likelihood (sampling distribution) of observations
    Y_obs_ex = pm.Bernoulli('Y_obs_ex', p=logistic(mu), observed=data_ex.Y)

with bayes_log_reg_model_exp:
    trace_ex = pm.sample_smc(mcmc_sample, parallel=True)
    # prior_trace_ex = pm.sample_prior_predictive(mcmc_sample, random_seed=42)  # prior sampling

Likelihood_Ex1_uniform = Calculate_Likelihood_by_Hand3(trace_ex, data_ex)
f_a2 = norm(0, 10).logpdf(trace_ex['alpha'])
f_b02 = norm(0, 10).logpdf(trace_ex['beta'][:, 0])
f_b12 = norm(0, 10).logpdf(trace_ex['beta'][:, 1])
uniform_log_prior = (sum(f_a2) + sum(f_b02) + sum(f_b12))/len(trace_ex['alpha'])
print('For the subset (Z2,X) the marginal likelihood in experimental data with uniform prior is:'
      , Likelihood_Ex1_uniform + uniform_log_prior)

Likelihood_Ex1_post = Calculate_Likelihood_by_Hand3(trace3, data_ex)

# post_prior = (sum(np.log(trace3['alpha'])) + sum(np.log(trace3['beta'][:, 0])) + sum(np.log(trace3['beta'][:, 1])))\
#              / len(trace3['alpha'])
post_prior = log_prior3

print('For the subset (Z2,X) the marginal likelihood in experimental data with posterior prior is:'
      , Likelihood_Ex1_post + post_prior)


"""FOR IMB,CMB be (X)"""
bayes_log_reg_model_exp1 = pm.Model()

with bayes_log_reg_model_exp1:
    # Priors for unknown model parameters
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10, shape=1)

    # Expected value of outcome
    mu = alpha + beta[0] * data_ex.X

    # Likelihood (sampling distribution) of observations
    Y_obs_ex = pm.Bernoulli('Y_obs_ex1', p=logistic(mu), observed=data_ex.Y)

with bayes_log_reg_model_exp1:
    trace_ex1 = pm.sample_smc(mcmc_sample, parallel=True)
    # prior_trace_ex1 = pm.sample_prior_predictive(mcmc_sample, random_seed=42)  # prior sampling

Likelihood_Ex1_uniform1 = Calculate_Likelihood_by_Hand4(trace_ex1, data_ex)
f_a21 = norm(0, 10).logpdf(trace_ex1['alpha'])
f_b021 = norm(0, 10).logpdf(trace_ex1['beta'][:, 0])
uniform_log_prior1 = (sum(f_a21) + sum(f_b021))/len(trace_ex1['alpha'])
print('For the subset (X) the marginal likelihood in experimental data with uniform prior is:'
      , Likelihood_Ex1_uniform1 + uniform_log_prior1)

Likelihood_Ex1_post1 = Calculate_Likelihood_by_Hand4(trace4, data_ex)

# post_prior1 = (sum(np.log(trace4['alpha'])) + sum(np.log(trace4['beta'][:, 0])))/len(trace4['alpha'])
post_prior1 = log_prior4
print('For the subset (X) the marginal likelihood in experimental data with posterior prior is:'
      , Likelihood_Ex1_post1 + post_prior1)
