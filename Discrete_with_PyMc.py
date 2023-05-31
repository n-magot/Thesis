"""Vriskw to Marginal me observational prior mono me to PyMc kai oxi me ton by_hand tropo"""

import arviz as az
import numpy as np
from scipy.stats import bernoulli
import pandas as pd
import pymc3 as pm
import theano.tensor as T
from itertools import combinations
import operator
from scipy.special import expit
import openturns as ot
import matplotlib.pyplot as plt

ot.Log.Show(ot.Log.NONE)
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(1234)

epss = np.finfo(np.float32).eps

"""Generate observational data"""
No_samples = 100
Ne_samples = 50

def Generate_Observational_Data(sample_size):

    alpha = 1
    beta = [0.3, 1.25, 1.4, 1.15]
    e = np.random.normal(1, 0, sample_size)  # noise

    Z1 = np.random.normal(0, 15, sample_size)
    Z2 = np.random.normal(0, 10, sample_size)
    μ_true = beta[0] * Z1 + beta[1] * Z2
    p_true = expit(μ_true)
    X = bernoulli.rvs(p_true)
    μ_true1 = alpha + beta[2] * X + beta[3] * Z2 + e
    p_true1 = expit(μ_true1)
    Y = bernoulli.rvs(p_true1)

    data = pd.DataFrame({"Y": Y, 'X': X, "Z1": Z1, "Z2": Z2})

    return data

data = Generate_Observational_Data(No_samples)

"""!!!  Oti dataframe kai na diavazoume tha vazoume stin prwti stili 
to outcome(Y) kai sthn 2h to treatment(X) !!!"""

def logistic(l):
    return 1 / (1 + T.exp(-l))

"""Take all possible combinations for regression """
#Pairnei to dataset kai ftiaxnei olous tous pithanous syndiasmous pou periexoun to X
# wste na kanw ta regression me 1 function

sample_list = data.columns.values[1:]
list_comb = []
for i in range(data.shape[1]-1):
    list_combinations = list(combinations(sample_list, data.shape[1]-i-1))
    for x in list_combinations:
        if x[0] == 'X':
            list_comb.append(x)
# print('The combinations of regression models for Y are {}'.format(list_comb))

"Function for regression model "
#data: Dataset with Y(oucome) in the first column and X(Treatment) in the second column
#reg_variables: variables for regression

def Regression_cases(data, reg_variables,N_samples):

    N_coef = len(reg_variables)

    reg_model = pm.Model()

    with reg_model:
        # Priors for unknown model parameters
        """O re file sos gia ta priors, epd mpainei sto logit transformation den vazo megalo sd=100 alla 1"""
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=N_coef)

        # Probability of Expected value of outcome
        p = alpha
        for i in range(len(reg_variables)):
            p = p + beta[i]*data['{}'.format(reg_variables[i])]

        Y_obs = pm.Bernoulli('Y_obs', p=logistic(p), observed=data.iloc[:, 0])
        trace = pm.sample_smc(1000, parallel=True)


    """Estimate a,b0,b1,b2 to verify that our model is ok"""
    map_estimate = pm.find_MAP(model=reg_model)
    print(map_estimate)

    """The Marginal Likelihood integral(P(Do|θ)*P(θ)dθ)"""
    Marginal_Likelihood = trace.report.log_marginal_likelihood

    return (Marginal_Likelihood[0] + Marginal_Likelihood[1])/2, trace

MB_Scores = {}
trace_list = {}

for i in range(len(list_comb)):
    Marginal_Likelihood, trace = Regression_cases(data, list_comb[i], No_samples)
    MB_Scores[list_comb[i]] = Marginal_Likelihood
    trace_list[list_comb[i]] = trace

    # print("For set Z ={} as MB, the Marginal Likelihood  is P(Do) = ".format(list_comb[i]), Marginal_Likelihood)
MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
# print(MB_Scores)
# print('The set that is the Markov Boundary in observational data is:', MB_Do)

"""Generate Experimental data"""

def Generate_Experimental_Data(sample_size):
    alpha = 1
    beta = [1.4, 1.15]
    e = np.random.normal(1, 0, sample_size)  # noise

    Z2 = np.random.normal(0, 10, sample_size)
    X = bernoulli.rvs(0.5)   #Treatment  randomized controlled trial(RTC)
    μ_true1 = alpha + beta[0] * X + beta[1] * Z2 + e
    p_true1 = expit(μ_true1)
    Y = bernoulli.rvs(p_true1)

    data_ex = pd.DataFrame({"Y": Y, 'X': X, "Z2": Z2})

    return data_ex

data_ex = Generate_Experimental_Data(Ne_samples)

"""Create a list with every SUBSET of MB calculated before"""

sample_list = list(MB_Do)

subset_list = []
for i in range(len(sample_list)):
    list_combinations = list(combinations(sample_list, len(sample_list)-i))
    for x in list_combinations:
        if x[0] == 'X':
            subset_list.append(x)
# print('The subsets of MB are {}'.format(subset_list))

"""Calculate Marginal Likelihood for experimental data with flat prior"""

IMB_flat_prior = {}

for i in range(len(subset_list)):
    Marginal_Likelihood_ex_flat, trace_ex_flat = Regression_cases(data_ex, subset_list[i], Ne_samples)
    IMB_flat_prior[subset_list[i]] = Marginal_Likelihood_ex_flat
    # print("For set Z ={} as IMB<>CMB, the Marginal Likelihood  is P(De|Do, Hz~) = ".format(subset_list[i]),
    #       IMB_flat_prior[subset_list[i]])

"""Calculate the marginal with prior calculated before as normal distribution with mean= mean_post_coef and
std=std_post_coef, if we assume that the new posterior distribution of the coefficients will be normal with PyMc"""

def Marginal_Ex_PyMc(data, reg_variables, trace):

    N_coef = len(reg_variables)
    func_dict = {"mean": np.mean, "std": np.std}
    post_a = az.summary(trace['alpha'], stat_funcs=func_dict, extend=False)
    mean_post_a = post_a.iloc[0]['mean']
    std_post_a = post_a.iloc[0]['std']

    post_b = {}
    dok_post_b = {}
    """Panta to prwto bo tha einai sto X:to treatment"""
    for i in range(N_coef):
        post_b["post_b{0}".format(i)] = az.summary(trace['beta'][:, i], stat_funcs=func_dict, extend=False)
        mean_post = post_b["post_b{0}".format(i)].iloc[0]['mean']
        std_post = post_b["post_b{0}".format(i)].iloc[0]['std']
        dok_post_b["post_b{0}".format(i)] = [mean_post, std_post]

    reg_model = pm.Model()

    with reg_model:
        # Priors for unknown model parameters
        alpha = pm.Normal("alpha", mu=mean_post_a, sigma=std_post_a)

        # Probability of Expected value of outcome
        p = alpha
        for i in range(len(reg_variables)):
            mean = dok_post_b["post_b{0}".format(i)][0]
            std = dok_post_b["post_b{0}".format(i)][1]
            beta = pm.Normal("beta{0}".format(i), mu=mean, sigma=std, shape=1)
            p = p + beta*data['{}'.format(reg_variables[i])]

        Y_obs_ex = pm.Bernoulli('Y_obs_ex', p=logistic(p), observed=data.iloc[:, 0])
        trace_ex = pm.sample_smc(1000, parallel=True)

    """The Marginal Likelihood integral(P(Do|θ)*P(θ)dθ)"""
    Marginal_Likelihood = trace_ex.report.log_marginal_likelihood

    return Marginal_Likelihood

IMB_post_prior_normal = {}
for i in range(len(subset_list)):
    Marginal_Likelihood_dok = Marginal_Ex_PyMc(data_ex, subset_list[i],  trace_list[subset_list[i]])
    IMB_post_prior_normal[subset_list[i]] = Marginal_Likelihood_dok

print('The combinations of regression models for Y are {}'.format(list_comb))
print(MB_Scores)
print('The set that is the Markov Boundary in observational data is:', MB_Do)
print('The subsets of MB are {}'.format(subset_list))
print("IMB<>CMB:", IMB_flat_prior)
print("IMB==CMB PYMC with normal posterior:", IMB_post_prior_normal)
