import operator
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import jax
import jax.numpy as jnp
from jax import random
import time
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("/home/nandia/PycharmProjects/remote_project/example_run/kaggle/weatherAUS.csv")

#delete rows with missing values on outcome==RainTomorrow

df.dropna(subset=['RainTomorrow'], inplace=True)

col_names = df.columns

# print(col_names)

# find categorical variables
categorical = [var for var in df.columns if df[var].dtype=='O']

# print('There are {} categorical variables\n'.format(len(categorical)))
# print('The categorical variables are :', categorical)

# check missing values in categorical variables
df[categorical].isnull().sum()
# print categorical variables containing missing values
cat1 = [var for var in categorical if df[var].isnull().sum()!=0]

# parse the dates, currently coded as strings, into datetime format
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# drop the original Date variable

df.drop('Date', axis=1, inplace = True)

# find categorical variables
categorical = [var for var in df.columns if df[var].dtype=='O']

#in location variable there are 49 labels and we will do OnehotEncoding
pd.get_dummies(df.Location, drop_first=True).head()
#oneHot encoding for WindGustDir etc..

pd.get_dummies(df.WindGustDir, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindDir9am, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.WindDir3pm, drop_first=True, dummy_na=True).head()
pd.get_dummies(df.RainToday, drop_first=True, dummy_na=True).head()

#Explore Numerical Variables
numerical = [var for var in df.columns if df[var].dtype!='O']

# print('There are {} numerical variables\n'.format(len(numerical)))
# print('The numerical variables are :', numerical)

# check missing values in numerical variables
df[numerical].isnull().sum()

X = df.drop(['RainTomorrow'], axis=1)

df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Engineering

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']


"""Assuming MCAT,  I will use median imputation because median imputation is robust to outliers. I will impute missing 
values with the appropriate statistical measures of the data, in this case median. Imputation should be done over the
training set, and then propagated to the test set. It means that the statistical measures to be used to fill missing
values both in train and test set, should be extracted from the train set only. This is to avoid overfitting."""

# impute missing values in X_train and X_test with respective column median in X_train
#Numerical variables
for df1 in [X_train, X_test]:
    for col in numerical:
        col_median = X_train[col].median()
        df1[col].fillna(col_median, inplace=True)

#Categorical variables

for df2 in [X_train, X_test]:
    df2['WindGustDir'].fillna(X_train['WindGustDir'].mode()[0], inplace=True)
    df2['WindDir9am'].fillna(X_train['WindDir9am'].mode()[0], inplace=True)
    df2['WindDir3pm'].fillna(X_train['WindDir3pm'].mode()[0], inplace=True)
    df2['RainToday'].fillna(X_train['RainToday'].mode()[0], inplace=True)

#Engineering outliers in numerical variables
def max_value(df3, variable, top):
    return numpy.where(df3[variable]>top, top, df3[variable])

for df3 in [X_train, X_test]:
    df3['Rainfall'] = max_value(df3, 'Rainfall', 3.2)
    df3['Evaporation'] = max_value(df3, 'Evaporation', 21.8)
    df3['WindSpeed9am'] = max_value(df3, 'WindSpeed9am', 55)
    df3['WindSpeed3pm'] = max_value(df3, 'WindSpeed3pm', 57)

# print(X_train[numerical].describe())
# encode RainToday variable

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location),
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_test.Location),
                     pd.get_dummies(X_test.WindGustDir),
                     pd.get_dummies(X_test.WindDir9am),
                     pd.get_dummies(X_test.WindDir3pm)], axis=1)

#Feature scaling

cols = X_train.columns
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

"""Now we keep only 1000 sample from 142.193 and 5 variables from 23 so to run our code"""
X_sk = X_train.iloc[0:1000, 0:5]
y_sk = y_train.iloc[0:1000]
# print(X_sk, y_sk)

#apply SelectKBest class to extract top 3 best features
bestfeatures = SelectKBest(score_func=chi2, k=3)
fit = bestfeatures.fit(X_sk, y_sk)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X_sk.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print('SelectkBest:')
print(featureScores.nlargest(3,'Score'))  #print 10 best features

# Import your necessary dependencies
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Feature extraction
model = LogisticRegression()
rfe = RFE(model, n_features_to_select=1, step=1)
fit = rfe.fit(X_sk, y_sk)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

"""From now on we start our code. We need to do some changes to the type of the data"""
num_samples = 1000
X_train = (X_sk.to_numpy())
y = (y_sk.to_numpy())

data = jnp.array(X_train)
labels = jnp.array(y)
#Used as observational data
data = jnp.array(X_train[0:1000, 0:5])
labels = jnp.array(y[0:1000])

#Used as experimental data
exp_data = jnp.array(X_train[0:100, 0:5])
exp_labels = jnp.array(y[0:100])

unique, counts = numpy.unique(labels, return_counts=True)

# display unique values and counts side by side
print('Unique counts on the outcome', numpy.asarray((unique, counts)).T)


st = time.time()

import itertools
def var_combinations(data):
    """function for finding the combinations of independent variables
    I assume that you star your counting from the first independent variable"""
    #how many variables we have
    num_variables = data.shape[1]
    column_list = (map(lambda var: var, range(0, num_variables)))

    df = pd.DataFrame(data, columns=column_list)
    sample_list = df.columns.values[0:]
    list_comb = []
    for r in range(1, len(sample_list) + 1):
        # to generate combination
        list_comb.extend(itertools.combinations(sample_list, r))
    return list_comb


rng_key = random.PRNGKey(numpy.random.randint(100))
rng_key, rng_key_ = random.split(rng_key)

def logistic_regression_model(data, labels):
    """Create the logistic regression model"""
    D = data.shape[1]
    alpha = numpyro.sample("alpha", dist.Cauchy(0, 10))
    beta = numpyro.sample("beta", dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)))
    logits = alpha + jnp.dot(data, beta)
    return numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=labels)

def log_likelihood_calculation(alpha, beta, data, obs):
    """function for calculate log likelihood"""
    logits = alpha + jnp.dot(data, beta)
    log_likelihood = dist.Bernoulli(logits=logits).log_prob(obs)
    return log_likelihood.sum()

def sample_prior(data):
    """Sample from prior distribution"""
    prior_samples = {}
    D = data.shape[1]
    #paizei kai i paralagi me Cauchy(0, 2.5) gia tous coefs kai Cauchy(0, 10 ) gia to intercept
    # coefs_samples = dist.Normal(jnp.zeros(D), 100*jnp.ones(D)).sample(random.PRNGKey(0), (num_samples,))
    coefs_samples = dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)).sample(rng_key, (num_samples,))
    # intecept_samples = dist.Normal(jnp.zeros(1), 100*jnp.ones(1)).sample(random.PRNGKey(0), (num_samples,))
    intecept_samples = dist.Cauchy(jnp.zeros(1), 10 * jnp.ones(1)).sample(rng_key, (num_samples,))

    prior_samples["beta"] = coefs_samples
    prior_samples["alpha"] = intecept_samples
    pdf_priors = (((dist.Cauchy(jnp.zeros(D), 2.5 * jnp.ones(D)).log_prob(coefs_samples)).sum() +
                   dist.Cauchy(jnp.zeros(1), 10 * jnp.ones(1)).log_prob(intecept_samples).sum())) / num_samples
    return prior_samples, pdf_priors

def sample_posterior(data, observed_data):
    """Posterior sampling using mcmc"""
    kernel = NUTS(logistic_regression_model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(jax.random.PRNGKey(42), data, observed_data)
    data_plot = az.from_numpyro(mcmc)
    az.plot_trace(data_plot, compact=True, figsize=(15, 25))

    """Uncomment it if you want to see the posterior distribution of samples"""
    # plt.show()

    # Get the posterior samples
    posterior_samples = mcmc.get_samples()

    return posterior_samples

# Calculate log likelihood for each posterior sample
def calculate_log_marginal(num_samples, samples, data, observed_data):
    """function for calclulate log marginal likelihood"""
    log_likelihoods = jnp.zeros(num_samples)

    for i in range(num_samples):
        log_likelihoods = log_likelihoods.at[i].set(log_likelihood_calculation(samples["alpha"][i], samples["beta"][i],
                                                                               data, observed_data))
    # Estimate the log marginal likelihood using the log-sum-exp trick
    log_marginal_likelihood = jax.scipy.special.logsumexp(log_likelihoods) - jnp.log(num_samples)
    return log_marginal_likelihood

"""Find MB"""
MB_Scores = {}

list_comb = var_combinations(data)
print('The combination of variables:', list_comb)

for comb in range(len(list_comb)):
    reg_variables = list_comb[comb]

    sub_data = data[:, reg_variables]

    prior_samples, pdf_prior = sample_prior(sub_data)
    marginal = calculate_log_marginal(num_samples, prior_samples, sub_data, labels)

    MB_Scores[reg_variables] = marginal + pdf_prior
MB_Do = max(MB_Scores.items(), key=operator.itemgetter(1))[0]
print('For every subset the log marginal likelihood:', MB_Scores)
print('Markov boundary:', MB_Do)

"""Searching for subsets of MB"""
sample_list = [i for i in MB_Do]
subset_list = []
for r in range(1, len(sample_list) + 1):
    # to generate combination
    subset_list.extend(itertools.combinations(sample_list, r))
print('The subsets of MB are {}'.format(subset_list))

"""Calculate marginals for subsets of MB sample from experimental and observational data"""
IMB_Scores_exp = {}
IMB_Scores_obs = {}
for j in range(len(subset_list)):

    reg_variables = subset_list[j]
    sub_data = data[:, reg_variables]
    exp_sub_data = exp_data[:, reg_variables]

    posterior_samples = sample_posterior(sub_data, labels)

    prior_samples, pdf_prior = sample_prior(exp_sub_data)

    marginal = calculate_log_marginal(num_samples, prior_samples, exp_sub_data, exp_labels)
    print('Marginal {} from experimental sampling:'.format(reg_variables), marginal)
    IMB_Scores_exp[reg_variables] = marginal

    marginal = calculate_log_marginal(num_samples, posterior_samples, exp_sub_data, exp_labels)
    print('Marginal {} from observational sampling:'.format(reg_variables), marginal)

    IMB_Scores_obs[reg_variables] = marginal

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
