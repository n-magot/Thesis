from scipy.special import expit
import arviz as az
import jax
import matplotlib.pyplot as plt
import numpy
import numpyro.distributions as dist
import pandas as pd
from jax import numpy as np, random
import numpyro
from numpyro import sample
from numpyro.distributions import (Normal)
from numpyro.infer import MCMC, NUTS, log_likelihood, init_to_feasible, init_to_sample, init_to_uniform, init_to_value
import seaborn as sns
from numpyro import sample, handlers
from jax.scipy.special import logsumexp
from itertools import combinations
# import jax.numpy as jnp
from jax import random, vmap
import math
from scipy.stats import norm
import operator
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import openturns as ot

ot.Log.Show(ot.Log.NONE)
numpyro.enable_x64()

assert numpyro.__version__.startswith("0.11.0")
az.style.use("arviz-darkgrid")

def Generate_Observational_Data(sample_size):

    alpha = [-4, 4]
    beta = [1.2, 0.7, 1.4, 0.9]
  # [Z1, Z2, X, Z2]
    # beta = [1.3, 1.25, 1.4, 1.15]

    e = dist.Normal(0, 1).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(sample_size,))
    Z1 = dist.Normal(0, 15).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(sample_size,))
    Z2 = dist.Normal(0, 10).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(sample_size,))

    μ_true = beta[0] * Z1 + beta[1] * Z2
    p_true = expit(μ_true)
    X = dist.Bernoulli(p_true).sample(random.PRNGKey(numpy.random.randint(100)))

    logit_0 = alpha[0] + (beta[2] * X + beta[3]*Z2) + e
    logit_1 = alpha[1] + (beta[2] * X + beta[3]*Z2) + e
    q_0 = expit(logit_0)
    q_1 = expit(logit_1) #probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = np.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(1,))[0]
    data = pd.DataFrame({"Y": Y, 'X': X, "Z1": Z1, "Z2": Z2})

    return data, Y, X, Z1, Z2

def Generate_Experimental_Data(Ne):

    alpha = [-4, 4]
    beta = [1.2, 0.7, 1.4, 0.9]
  # [Z1, Z2, X, Z2]
    # beta = [1.3, 1.25, 1.4, 1.15]

    e = dist.Normal(0, 1).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(sample_size,))
    Z1 = dist.Normal(0, 15).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(sample_size,))
    Z2 = dist.Normal(0, 10).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(sample_size,))

    X = dist.Bernoulli(0.5).sample(random.PRNGKey(numpy.random.randint(100)))  #Treatment  randomized controlled trial(RTC)

    logit_0 = alpha[0] + (beta[2] * X + beta[3]*Z2) + e
    logit_1 = alpha[1] + (beta[2] * X + beta[3]*Z2) + e
    q_0 = expit(logit_0)
    q_1 = expit(logit_1) #probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = np.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(100)), sample_shape=(1,))[0]
    data = pd.DataFrame({"Y": Y, 'X': X, "Z1": Z1, "Z2": Z2})

    return data, Y, X, Z1, Z2


def Regression_cases(X,Z1,Z2,reg_variables,Y=None):

    vars_dict = {'Y': Y, 'X': X, 'Z1': Z1, 'Z2': Z2}
    beta = {}
    for i in range(len(reg_variables)):
        beta['beta_{}'.format(reg_variables[i])] = (sample('beta_{}'.format(reg_variables[i]), Normal(0, 1)))

    cutpoints = numpyro.sample(
        "cutpoints",
        dist.TransformedDistribution(
            dist.Normal(0, 1).expand([N_classes - 1]),
            dist.transforms.OrderedTransform(),
        ),
    )
    prediction = 0
    for i in range(len(reg_variables)):
        prediction = prediction + beta['beta_{}'.format(reg_variables[i])] * vars_dict['{}'.format(reg_variables[i])]

    numpyro.sample(
        "Y",
        dist.OrderedLogistic(predictor=prediction, cutpoints=cutpoints),
        obs=Y
    )

def var_combinations(data):
    sample_list = data.columns.values[1:]
    list_comb = []
    for l in range(data.shape[1]-1):
        list_combinations = list(combinations(sample_list, data.shape[1]-l-1))
        for x in list_combinations:
            if x[0] == 'X':
                list_comb.append(x)

    return list_comb


correct_MB = 0
correct_IMB = 0

sample_size = 500
Ne = 100

fb_trace = 0
num_warmup, num_samples = 1000, 1000

data, Y, X, Z1, Z2 = Generate_Observational_Data(sample_size)
print(data["Y"].value_counts())

"""Take all possible combinations for regression """
list_comb = var_combinations(data)

"""Regression for (X,Z1,Z2), (X,Z1), (X,Z2), {X}"""
MB_Scores = {}
list_of_traces = {}

for i in range(len(list_comb)):
    reg_variables = list_comb[i]

    N_classes = 3

    kernel = NUTS(Regression_cases, init_strategy=init_to_sample())
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(1670940923), X, Z1, Z2, reg_variables, Y)
    mcmc.print_summary()
    trace = mcmc.get_samples()

    list_of_traces['trace_{}'.format(reg_variables)] = trace

    Log_likelhood_dict = log_likelihood(Regression_cases, trace, X, Z1, Z2, reg_variables, Y)

    Log_likelhood = sum(sum(Log_likelhood_dict['Y']))

    Marginal_Likelihood = Log_likelhood

    fa_trace = sum(norm(0, 1).logpdf(numpy.array(trace['cutpoints'][:, 0])) +
                   norm(0, 1).logpdf(numpy.array(trace['cutpoints'][:, 1])))

    fb_trace = 0
    for k in range(len(reg_variables)):
        fb_trace = fb_trace + norm(0, 1).logpdf(numpy.array(trace['beta_{}'.format(reg_variables[k])]))

    f_prior = sum(fb_trace)

    MB_Scores[reg_variables] = Log_likelhood + f_prior + fa_trace

MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]

print(MB_Do)
print(MB_Scores)

if MB_Do == (('X', 'Z2')):
    correct_MB = correct_MB + 1

sample_list = list(MB_Do)

subset_list = []
for i in range(len(sample_list)):
    list_combinations = list(combinations(sample_list, len(sample_list)-i))
    for x in list_combinations:
        if x[0] == 'X':
            subset_list.append(x)
print('The subsets of MB are {}'.format(subset_list))

data_ex, Y, X, Z1, Z2 = Generate_Experimental_Data(Ne)

"""Calculate Marginal Likelihood for experimental data with flat prior"""
IMB_flat_prior = {}
for i in range(len(subset_list)):
    reg_variables = subset_list[i]

    N_classes = 3

    kernel = NUTS(Regression_cases, init_strategy=init_to_sample())
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(random.PRNGKey(1670940923), X, Z1, Z2, reg_variables, Y)
    mcmc.print_summary()
    trace = mcmc.get_samples()

    Log_likelhood_dict = log_likelihood(Regression_cases, trace, X, Z1, Z2, reg_variables, Y)

    Log_likelhood = sum(sum(Log_likelhood_dict['Y']))

    Marginal_Likelihood = Log_likelhood

    fa_trace = sum(norm(0, 1).logpdf(numpy.array(trace['cutpoints'][:, 0])) +
                   norm(0, 1).logpdf(numpy.array(trace['cutpoints'][:, 1])))

    fb_trace = 0
    for k in range(len(reg_variables)):
        fb_trace = fb_trace + norm(0, 1).logpdf(numpy.array(trace['beta_{}'.format(reg_variables[k])]))

    f_prior = sum(fb_trace)

    IMB_flat_prior[reg_variables] = Log_likelhood + f_prior + fa_trace
print("With flat prior:", IMB_flat_prior)


def calculate_LogPDF_of_trace(trace):
    ot.RandomGenerator.SetSeed(1000)
    sample = ot.Sample.BuildFromPoint(trace) #Convert array to openturns object
    ks = ot.KernelSmoothing()
    fittedDist = ks.build(sample)
    f_trace = fittedDist.computeLogPDF(sample)

    return f_trace

"""Calculate Marginal Likelihood for experimental data with post prior"""
IMB_post_prior = {}
for i in range(len(subset_list)):
    reg_variables = subset_list[i]

    N_classes = 3
    trace = list_of_traces['trace_{}'.format(reg_variables)]

    Log_likelhood_dict = log_likelihood(Regression_cases, trace, X, Z1, Z2, reg_variables, Y)

    Log_likelhood = sum(sum(Log_likelhood_dict['Y']))

    Marginal_Likelihood = Log_likelhood

    fa_trace = sum(calculate_LogPDF_of_trace(numpy.array(trace['cutpoints'][:, 0]))) +\
               sum(calculate_LogPDF_of_trace(numpy.array(trace['cutpoints'][:, 1])))

    fb_trace = 0
    for k in range(len(reg_variables)):
        fb_trace = fb_trace + sum(calculate_LogPDF_of_trace(numpy.array(trace['beta_{}'.format(reg_variables[k])])))

    IMB_post_prior[reg_variables] = Log_likelhood + fb_trace[0] + fa_trace[0]
if ('X', 'Z2') in IMB_post_prior:
    if IMB_post_prior[('X', 'Z2')] > IMB_flat_prior[('X', 'Z2')] and IMB_post_prior[('X',)] < IMB_flat_prior[('X',)]:
        correct_IMB = correct_IMB + 1
print("With posterior as prior:", IMB_post_prior)

