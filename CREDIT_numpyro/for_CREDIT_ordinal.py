"""Mia asxeti metavliti Z3 kai na ftiaxw ta functions na einai gia kathe eisodo"""
"""den exw valei na vriskei automata ton arithmo ton klaseon kai tin arithmisi sta data, dld [X,Z1,Z2..]-[0,1,2,]"""

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import operator
from jax.scipy.special import expit, logit
from itertools import combinations
import numpy
import pandas as pd
import arviz as az
import jax
import matplotlib.pyplot as plt
from jax import numpy as np, random
import numpyro
from numpyro import sample, handlers
from numpyro.distributions import (
    Dirichlet,
    TransformedDistribution,
    transforms,
)
from numpyro.infer.reparam import TransformReparam


"""Ordinal Data"""

"""Values that we change in every run:"""
No = 1000
Ne = 300

rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)

def Generate_Observational_Data(sample_size):

    #beta_X > Z1*X +Z2*X
    beta_X = jnp.array([1.5, 1.5])
    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 1.2])

    e = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    Z1 = 0 + 15 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z3 = dist.Bernoulli(probs=0.7).sample(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    data_Z1_Z2 = jnp.concatenate((Z1, Z2), axis=1)

    logits_X = jnp.sum(beta_X * data_Z1_Z2, axis=-1)
    X = dist.Bernoulli(logits=logits_X).sample(random.PRNGKey(numpy.random.randint(1000)))
    X = X.reshape(-1, 1)
    data_X_Z2 = jnp.concatenate((X, Z2), axis=1)
    logit_0 = -5 + jnp.sum(beta_Y * data_X_Z2 + e, axis=-1)
    logit_1 = 5 + jnp.sum(beta_Y * data_X_Z2 + e, axis=-1)
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)  # probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = jnp.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(1000)), sample_shape=(1,))[0]
    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((X, Z1, Z2, Z3), axis=1)

    return data, labels


def Generate_Experimental_Data(sample_size):

    #beta_Y > X*Y + Z2*Y
    beta_Y = jnp.array([1.4, 1.2])

    e = 0 + 1 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    Z1 = 0 + 15 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z2 = 0 + 10 * random.normal(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    Z3 = dist.Bernoulli(probs=0.7).sample(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))

    X = dist.Bernoulli(probs=0.5).sample(random.PRNGKey(numpy.random.randint(1000)), (sample_size, 1))
    data_X_Z2 = jnp.concatenate((X, Z2), axis=1)
    logit_0 = -5 + jnp.sum(beta_Y * data_X_Z2 + e, axis=-1)
    logit_1 = 5 + jnp.sum(beta_Y * data_X_Z2 + e, axis=-1)
    q_0 = expit(logit_0)
    q_1 = expit(logit_1)  # probability of class 1 or 0
    prob_0 = q_0
    prob_1 = q_1 - q_0
    prob_2 = 1 - q_1
    probs = jnp.stack((prob_0, prob_1, prob_2), axis=1)

    Y = dist.Categorical(probs=probs).sample(random.PRNGKey(numpy.random.randint(1000)), sample_shape=(1,))[0]

    labels = Y
    # data pane X,Z1,Z2
    data = jnp.concatenate((X, Z1, Z2, Z3), axis=1)

    return data, labels

def ordinal_logistic_regression(data, labels):
    D = data.shape[1]
    N_classes = 3
    concentration = np.ones((N_classes,)) * 10.0
    anchor_point = 0.0

    # mporw na paix kai me: dist.StudentT(5, jnp.zeros(D), 1*jnp.ones(D))
    coefs = numpyro.sample('coefs', dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)))

    with handlers.reparam(config={"cutpoints": TransformReparam()}):
        cutpoints = sample(
            "cutpoints",
            TransformedDistribution(
                Dirichlet(concentration),
                transforms.SimplexToOrderedTransform(anchor_point),
            ),
        )
    logits = jnp.sum(coefs * data, axis=-1)

    return numpyro.sample('obs', dist.OrderedLogistic(predictor=logits, cutpoints=cutpoints), obs=labels)

def log_likelihood_calculation(cutpoints, coefs, data, obs):

    logits = jnp.dot(data, coefs)
    log_likelihood = dist.OrderedLogistic(predictor=logits, cutpoints=cutpoints).log_prob(obs)
    return log_likelihood.sum()

def sample_prior(data):
    prior_samples = {}
    D = data.shape[1]
    N_classes = 3
    concentration = np.ones((N_classes,)) * 10.0
    anchor_point = 0.0

    coefs = dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))

    cutpoints = (TransformedDistribution(Dirichlet(concentration), transforms.SimplexToOrderedTransform(anchor_point), )
                 .sample(random.PRNGKey(0), (num_samples,)))

    prior_samples['coefs'] = coefs
    prior_samples['cutpoints'] = cutpoints

    return prior_samples


def sample_posterior(data, observed_data):
    kernel = NUTS(ordinal_logistic_regression, init_strategy=numpyro.infer.init_to_sample)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(jax.random.PRNGKey(42), data, observed_data)

    # Get the posterior samples
    posterior_samples = mcmc.get_samples()
    data_plot = az.from_numpyro(mcmc)
    az.plot_trace(data_plot, compact=True, figsize=(15, 25))
    # plt.show()
    return posterior_samples

def calculate_log_marginal(num_samples, samples, data, observed_data):
    log_likelihoods = jnp.zeros(num_samples)

    for i in range(num_samples):
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_calculation(samples["cutpoints"][i],
                                                                               samples["coefs"][i], data, observed_data))

    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    return log_marginal_likelihood

def var_combinations(data):
    #how many variables we have
    num_variables = data.shape[1]
    column_list = list(map(lambda var: var, range(0, num_variables)))

    df = pd.DataFrame(data, columns=column_list)
    sample_list = df.columns.values[0:]
    list_comb = []
    for l in range(df.shape[1]-1):
        list_combinations = list(combinations(sample_list, df.shape[1]-l))
        for x in list_combinations:
            if x[0] == 0:
                list_comb.append(x)
    return list_comb


correct_MB = 0
correct_IMB = 0
runs = 1
for i in range(runs):
    num_samples = 1000

    data, labels = Generate_Observational_Data(No)
    # get unique values and counts of each value
    unique, counts = numpy.unique(labels, return_counts=True)

    # display unique values and counts side by side
    print('Observational data', numpy.asarray((unique, counts)).T)

    exp_data, exp_labels = Generate_Experimental_Data(Ne)
    # get unique values and counts of each value
    unique, counts = numpy.unique(exp_labels, return_counts=True)

    # display unique values and counts side by side
    print('Experimental data', numpy.asarray((unique, counts)).T)

    MB_Scores = {}
    IMB_Scores_obs = {}
    IMB_Scores_exp = {}

    list_comb = var_combinations(data)
    print(list_comb)

    for comb in range(len(list_comb)):
        reg_variables = list_comb[comb]

        sub_data = data[:, reg_variables]

        prior_samples = sample_prior(sub_data)

        marginal = calculate_log_marginal(num_samples, prior_samples, sub_data, labels)

        MB_Scores[reg_variables] = marginal

    """Dictionary of marginals from prior sampling"""

    MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
    print(MB_Scores)
    print(MB_Do)
    if MB_Do == (0, 2):
        correct_MB = correct_MB + 1

    sample_list = list(MB_Do)

    """Searching for subsets of MB"""
    subset_list = []
    for s in range(len(sample_list)):
        list_combinations = list(combinations(sample_list, len(sample_list)-s))
        for x in list_combinations:
            if x[0] == 0:
                subset_list.append(x)
    print('The subsets of MB are {}'.format(subset_list))

    """For subsets of MB sample from experimental and observational data"""
    for j in range(len(subset_list)):

        reg_variables = subset_list[j]
        sub_data = data[:, reg_variables]
        exp_sub_data = exp_data[:, reg_variables]

        posterior_samples = sample_posterior(sub_data, labels)

        prior_samples = sample_prior(exp_sub_data)

        marginal = calculate_log_marginal(num_samples, prior_samples, exp_sub_data, exp_labels)
        print('Marginal {} from experimental sampling:'.format(reg_variables), marginal)
        IMB_Scores_exp[reg_variables] = marginal


        marginal = calculate_log_marginal(num_samples, posterior_samples, exp_sub_data, exp_labels)
        print('Marginal {} from observational sampling:'.format(reg_variables), marginal)

        IMB_Scores_obs[reg_variables] = marginal

    if IMB_Scores_obs[(0, 2)] > IMB_Scores_exp[(0, 2)] and IMB_Scores_obs[(0,)] < IMB_Scores_exp[(0,)]:
        correct_IMB = correct_IMB + 1

print(correct_MB)
print(correct_IMB)


#Afou vrw to MB kai CMB tha exw se ena array ta onomata kai tha kanw variables_array(np.array(MB_Do))
