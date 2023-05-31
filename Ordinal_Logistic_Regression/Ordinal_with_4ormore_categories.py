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
from numpyro.infer import MCMC, NUTS, log_likelihood, init_to_feasible, init_to_sample, init_to_uniform
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
numpyro.enable_x64()

assert numpyro.__version__.startswith("0.11.0")
az.style.use("arviz-darkgrid")

def Generate_Observational_Data(sample_size):

    alpha = [-3, 3, 6]
    beta = [0.9, 0.5, 1.4, 1.8]
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
    logit_2 = alpha[2] + (beta[2] * X + beta[3]*Z2) + e

    q_0 = expit(logit_0)
    q_1 = expit(logit_1) #probability of class 1 or 0
    q_2 = expit(logit_2)
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = q_2 - q_1
    prob_3 = 1 - q_2
    probs = np.stack((prob_0, prob_1, prob_2, prob_3), axis=1)

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
correct_MB1 = 0

sample_size = 100
runs = 3
for p in range(runs):
    fb_trace = 0
    num_warmup, num_samples = 1000, 1000

    data, Y, X, Z1, Z2 = Generate_Observational_Data(sample_size)
    print(data["Y"].value_counts())
    # data["Y"].value_counts().plot.barh()
    # plt.show()

    """Create correlation matrix"""
    # corr_matrix = data.corr()
    # sn.heatmap(corr_matrix, annot=True)
    # plt.show()

    """Take all possible combinations for regression """
    list_comb = var_combinations(data)

    """Regression for (X,Z1,Z2), (X,Z1), (X,Z2), {X}"""
    MB_Scores = {}
    MB_Scores1 = {}

    for i in range(len(list_comb)):
        reg_variables = list_comb[i]

        N_coef, N_classes = len(reg_variables), 4

        kernel = NUTS(Regression_cases, init_strategy=init_to_sample())
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(random.PRNGKey(1670940923), X, Z1, Z2, reg_variables, Y)
        mcmc.print_summary()
        trace = mcmc.get_samples()

        # post_pred = numpyro.infer.Predictive(Regression_cases, mcmc.get_samples())
        # post_predictions = post_pred(
        #     jax.random.PRNGKey(93),
        #     X=np.array([0, 0, 1, 1]),
        #     Z1=np.array([0.56, 1.67, 6.78, -2.34]),
        #     Z2=np.array([6.56, 7.67, 1.78, -0.34]),
        #     reg_variables=reg_variables
        # )
        #
        # fig, ax = plt.subplots(figsize=(14, 8))
        # list_coef = []
        # for co in reg_variables:
        #     list_coef.append("beta_{}".format(co))
        # for coefficient in list_coef:
        #     sns.kdeplot(mcmc.get_samples()[coefficient], ax=ax, label=coefficient)
        # ax.legend()
        # plt.show()


        Log_likelhood_dict = log_likelihood(Regression_cases, trace, X, Z1, Z2, reg_variables, Y)
        # print('Likelihood gia to b1', sum(Log_likelhood_dict['Y'][0]))
        Log_likelhood = sum(sum(Log_likelhood_dict['Y']))
        # print(Log_likelhood)

        Marginal_Likelihood = Log_likelhood
        MB_Scores[reg_variables] = Marginal_Likelihood

        # fb_trace = norm(0, 1).logpdf(numpy.array(trace['cutpoints'][:, 0])) + \
        #            norm(0, 1).logpdf(numpy.array(trace['cutpoints'][:, 1]))
        fb_trace = 0
        for k in range(len(reg_variables)):
            fb_trace = fb_trace + norm(0, 1).logpdf(numpy.array(trace['beta_{}'.format(reg_variables[k])]))

        f_prior = sum(fb_trace)
        prior = f_prior
        MB_Scores1[reg_variables] = Log_likelhood + prior

    MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
    MB_Do1 = max(MB_Scores1.items(), key=operator.itemgetter(1))[0]

    print(MB_Do)
    print(MB_Do1)
    print(MB_Scores)
    print(MB_Scores1)
    if MB_Do == (('X', 'Z2')):
        correct_MB = correct_MB + 1
    if MB_Do1 == (('X', 'Z2')):
        correct_MB1 = correct_MB1 + 1
print(correct_MB / runs)
print(correct_MB1 / runs)
