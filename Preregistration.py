######Preregistration file


############################################
#Step 1: Define the data generating process#
############################################

#%%load required packages
import numpyro
import jax
print('Numpyro Version: ',numpyro.__version__)
assert numpyro.__version__.startswith('0.15.2')
print('Jax Version: ',jax.__version__)
assert jax.__version__.startswith('0.4.31')
#assert jax.__version__.startswith('0.4.26')
import math
import os
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jax.numpy as jnp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from jax import lax, random
from jax.experimental.ode import odeint
from jax.scipy.special import logsumexp
from jax.scipy.special import expit
from numpyro.distributions.transforms import OrderedTransform
from numpyro.distributions.truncated import TruncatedNormal
import jax.random as random
import seaborn as sns
from scipy.stats import norm
import seaborn as sb
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, Predictive
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
#if "SVG" in os.environ:
 #   %config InlineBackend.figure_formats = ["svg"]
az.style.use("arviz-darkgrid")
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
#%%set random key
rng_key = random.PRNGKey(1)


#%%Defining the model
def model(DSTExperience, TechEngagement, RiskPref, Age, Farmsize, AdvisorExperience, #exogenous cahracteristics
          True_DeltaPerformance_Q0, True_DeltaPerformance_Q1, True_DeltaPerformance_Q2,#treatment
          AIA_1=None, AIA_2=None, AIA_3=None, AIA_4=None, AIA_5=None, AIA_6=None, AIA_7=None, AIA_8=None, AIA_9=None, AIA_10=None, AIA_11=None, AIA_12=None, AIA_13=None, AIA_14=None, AIA_15=None, AIA_16=None, #outcome of interest (AI Anxiety)
          SI_1=None, SI_2=None, 
          PE_1=None, PE_2=None, PE_3=None, PE_4=None,
          EE_1=None, EE_2=None, EE_3=None,
          BI_1=None, BI_2=None, BI_3=None, #outcome of interest (Behavioral Intention)
          Delta_WTP_Q0_relTo90=None, #outcome of interest (Delta WTP_Q0)
          Delta_WTP_Q1_relTo90=None, #outcome of interest (Delta WTP_Q1)
          Delta_WTP_Q2_relTo90=None, #outcome of interest (Delta WTP_Q2)
           ):
  
   #AIAnxiety
    alpha_idvConst_AIA = numpyro.sample('alpha_idvConst_AIA', dist.Normal (0,1))
    beta_DSTExperience_AIA = numpyro.sample('beta_DSTExperience_AIA', dist.Normal (0,0.5)) # we start with quite broad, uninformed priors
    beta_TechEngagement_AIA = numpyro.sample('beta_TechEngagement_AIA', dist.Normal (0,0.5))
    beta_RiskPref_AIA = numpyro.sample('beta_RiskPref_AIA', dist.Normal (0,0.5))
    beta_Age_AIA = numpyro.sample('beta_Age_AIA', dist.Normal (0,0.5))
    beta_Farmsize_AIA = numpyro.sample('beta_Farmsize_AIA', dist.Normal (0,0.5))
    beta_AdvisorExperience_AIA = numpyro.sample('beta_AdvisorExperience_AIA', dist.Normal (0,0.5))
    LatentAIAnxiety = numpyro.deterministic("LatentAIAnxiety",
                                            alpha_idvConst_AIA+
                                            beta_DSTExperience_AIA*DSTExperience + 
                                            beta_TechEngagement_AIA*TechEngagement + 
                                            beta_RiskPref_AIA*RiskPref + 
                                            beta_Age_AIA*Age +
                                            beta_Farmsize_AIA*Farmsize + 
                                            beta_AdvisorExperience_AIA*AdvisorExperience)
    #try individual cutpoints
    cutpoints_AIA_1 = numpyro.sample("cutpoints_AIA_1",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]), 
                                    OrderedTransform()))-2.5
    cutpoints_AIA_2 = numpyro.sample("cutpoints_AIA_2",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_3 = numpyro.sample("cutpoints_AIA_3",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_4 = numpyro.sample("cutpoints_AIA_4",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_5 = numpyro.sample("cutpoints_AIA_5",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_6 = numpyro.sample("cutpoints_AIA_6",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_7 = numpyro.sample("cutpoints_AIA_7",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_8 = numpyro.sample("cutpoints_AIA_8",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_9 = numpyro.sample("cutpoints_AIA_9",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_10 = numpyro.sample("cutpoints_AIA_10",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_11 = numpyro.sample("cutpoints_AIA_11",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_12 = numpyro.sample("cutpoints_AIA_12",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_13 = numpyro.sample("cutpoints_AIA_13",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_14 = numpyro.sample("cutpoints_AIA_14",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_15 = numpyro.sample("cutpoints_AIA_15",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
    cutpoints_AIA_16 = numpyro.sample("cutpoints_AIA_16",
                                    dist.TransformedDistribution(dist.Normal(0, 0.3).expand([6]),
                                    OrderedTransform()))-2.5
                                        
    AIA_1 = numpyro.sample('AIA_1', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_1), obs = AIA_1) 
    AIA_2 = numpyro.sample('AIA_2', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_2), obs = AIA_2)
    AIA_3 = numpyro.sample('AIA_3', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_3), obs = AIA_3)
    AIA_4 = numpyro.sample('AIA_4', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_4), obs = AIA_4)
    AIA_5 = numpyro.sample('AIA_5', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_5), obs = AIA_5)
    AIA_6 = numpyro.sample('AIA_6', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_6), obs = AIA_6)
    AIA_7 = numpyro.sample('AIA_7', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_7), obs = AIA_7)
    AIA_8 = numpyro.sample('AIA_8', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_8), obs = AIA_8)
    AIA_9 = numpyro.sample('AIA_9', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_9), obs = AIA_9)
    AIA_10 = numpyro.sample('AIA_10', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_10), obs = AIA_10)
    AIA_11 = numpyro.sample('AIA_11', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_11), obs = AIA_11)
    AIA_12 = numpyro.sample('AIA_12', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_12), obs = AIA_12)
    AIA_13 = numpyro.sample('AIA_13', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_13), obs = AIA_13)
    AIA_14 = numpyro.sample('AIA_14', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_14), obs = AIA_14)
    AIA_15 = numpyro.sample('AIA_15', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_15), obs = AIA_15)
    AIA_16 = numpyro.sample('AIA_16', dist.OrderedLogistic(LatentAIAnxiety, cutpoints_AIA_16), obs = AIA_16)

    # ####Part 1: Definition of UTAUT statements###

    # # # #Social Influence
    # # #set up similar to AI Anxiety
    beta_DSTExperience_SI = numpyro.sample('beta_DSTExperience_SI', dist.Normal (0,0.5)) 
    beta_TechEngagement_SI = numpyro.sample('beta_TechEngagement_SI', dist.Normal (0,0.5))
    beta_RiskPref_SI = numpyro.sample('beta_RiskPref_SI', dist.Normal (0,0.5))
    beta_Age_SI = numpyro.sample('beta_Age_SI', dist.Normal (0,0.5))
    beta_Farmsize_SI = numpyro.sample('beta_Farmsize_SI', dist.Normal (0,0.5))
    beta_AdvisorExperience_SI = numpyro.sample('beta_AdvisorExperience_SI', dist.Normal (0,0.5))
    mu_LatentSocialInfluence = numpyro.deterministic("mu_LatentSocialInfluence", 
                                                     beta_DSTExperience_SI*DSTExperience + 
                                                     beta_TechEngagement_SI*TechEngagement + 
                                                     beta_RiskPref_SI*RiskPref + 
                                                     beta_Age_SI*Age + 
                                                     beta_Farmsize_SI*Farmsize + 
                                                     beta_AdvisorExperience_SI*AdvisorExperience)
    cutpoints_SI_1 = numpyro.sample("cutpoints_SI_1", 
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]), 
                                  OrderedTransform()),)-2.5
    cutpoints_SI_2 = numpyro.sample("cutpoints_SI_2",
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]),
                                  OrderedTransform()),)-2.5
    SI_1 = numpyro.sample('SI_1', dist.OrderedLogistic(mu_LatentSocialInfluence, cutpoints_SI_1), obs=SI_1)
    SI_2 = numpyro.sample('SI_2', dist.OrderedLogistic(mu_LatentSocialInfluence, cutpoints_SI_2), obs=SI_2)
   
 
    # # #PerformanceExpectancy
    # # #set up similar to Social Influence except for additional effect of AI Anxiety on Performance Expectancy
    beta_DSTExperience_PE = numpyro.sample('beta_DSTExperience_PE', dist.Normal (0,0.5)) 
    beta_TechEngagement_PE = numpyro.sample('beta_TechEngagement_PE', dist.Normal (0,0.5))
    beta_RiskPref_PE = numpyro.sample('beta_RiskPref_PE', dist.Normal (0,0.5))
    beta_Age_PE = numpyro.sample('beta_Age_PE', dist.Normal (0,0.5))
    beta_Farmsize_PE = numpyro.sample('beta_Farmsize_PE', dist.Normal (0,0.5))
    beta_AdvisorExperience_PE = numpyro.sample('beta_AdvisorExperience_PE', dist.Normal (0,0.5))
    beta_AIA_PE = numpyro.sample('beta_AIA_PE', dist.Normal (0,0.5)) #effect of AIA on Performance Expectancy
    mu_LatentPerformanceExpectancy = numpyro.deterministic("mu_LatentPerformanceExpectancy", 
                                                           beta_DSTExperience_PE*DSTExperience + 
                                                           beta_TechEngagement_PE*TechEngagement + 
                                                           beta_RiskPref_PE*RiskPref + 
                                                           beta_Age_PE*Age +
                                                           beta_Farmsize_PE*Farmsize + 
                                                           beta_AdvisorExperience_PE*AdvisorExperience +
                                                           beta_AIA_PE*LatentAIAnxiety)
    cutpoints_PE_1 = numpyro.sample("cutpoints_PE_1", 
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]), 
                                  OrderedTransform()),)-2.5
    cutpoints_PE_2 = numpyro.sample("cutpoints_PE_2",
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]),
                                  OrderedTransform()),)-2.5
    cutpoints_PE_3 = numpyro.sample("cutpoints_PE_3",
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]),
                                  OrderedTransform()),)-2.5
    cutpoints_PE_4 = numpyro.sample("cutpoints_PE_4",
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]),
                                  OrderedTransform()),)-2.5
    PE_1 = numpyro.sample('PE_1', dist.OrderedLogistic(mu_LatentPerformanceExpectancy, cutpoints_PE_1), obs=PE_1)
    PE_2 = numpyro.sample('PE_2', dist.OrderedLogistic(mu_LatentPerformanceExpectancy, cutpoints_PE_2), obs=PE_2)
    PE_3 = numpyro.sample('PE_3', dist.OrderedLogistic(mu_LatentPerformanceExpectancy, cutpoints_PE_3), obs=PE_3)
    PE_4 = numpyro.sample('PE_4', dist.OrderedLogistic(mu_LatentPerformanceExpectancy, cutpoints_PE_4), obs=PE_4)
    

    # # #EffortExpectancy
    # # #same set up as Performance Expectancy
    beta_DSTExperience_EE = numpyro.sample('beta_DSTExperience_EE', dist.Normal (0,0.5))
    beta_TechEngagement_EE = numpyro.sample('beta_TechEngagement_EE', dist.Normal (0,0.5))
    beta_RiskPref_EE = numpyro.sample('beta_RiskPref_EE', dist.Normal (0,0.5))
    beta_Age_EE = numpyro.sample('beta_Age_EE', dist.Normal (0,0.5))
    beta_Farmsize_EE = numpyro.sample('beta_Farmsize_EE', dist.Normal (0,0.5))
    beta_AdvisorExperience_EE = numpyro.sample('beta_AdvisorExperience_EE', dist.Normal (0,0.5))
    beta_AIA_EE = numpyro.sample('beta_AIA_EE', dist.Normal (0,0.5)) #effect of AIA on Effort Expectancy
    mu_LatentEffortExpectancy = numpyro.deterministic("mu_LatentEffortExpectancy", 
                                                      beta_DSTExperience_EE*DSTExperience +
                                                      beta_TechEngagement_EE*TechEngagement + 
                                                      beta_RiskPref_EE*RiskPref + 
                                                      beta_Age_EE*Age +
                                                      beta_Farmsize_EE*Farmsize + 
                                                      beta_AdvisorExperience_EE*AdvisorExperience + 
                                                      beta_AIA_EE*LatentAIAnxiety)
    cutpoints_EE_1 = numpyro.sample("cutpoints_EE_1", 
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]), 
                                  OrderedTransform()),)-2.5
    cutpoints_EE_2 = numpyro.sample("cutpoints_EE_2",
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]),
                                  OrderedTransform()),)-2.5
    cutpoints_EE_3 = numpyro.sample("cutpoints_EE_3",
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]),
                                  OrderedTransform()),)-2.5
    EE_1 = numpyro.sample('EE_1', dist.OrderedLogistic(mu_LatentEffortExpectancy, cutpoints_EE_1), obs=EE_1)
    EE_2 = numpyro.sample('EE_2', dist.OrderedLogistic(mu_LatentEffortExpectancy, cutpoints_EE_2), obs=EE_2)
    EE_3 = numpyro.sample('EE_3', dist.OrderedLogistic(mu_LatentEffortExpectancy, cutpoints_EE_3), obs=EE_3)

    # # #BehavioralIntention
    beta_SI_BI = numpyro.sample('beta_SI_BI', dist.Normal (0,0.5)) #effect of Social Influence on Behavioral Intention
    beta_EE_BI = numpyro.sample('beta_EE_BI', dist.Normal (0,0.5)) #effect of Effort Expectancy on Behavioral Intention
    beta_PE_BI = numpyro.sample('beta_PE_BI', dist.Normal (0,0.5)) #effect of Performance Expectancy on Behavioral Intention
    beta_AIA_BI = numpyro.sample('beta_AIA_BI', dist.Normal (0,0.5)) #our parameter of interest: effect of AI Anxiety on Behavioral Intention
    mu_LatentBehavioralIntention = numpyro.deterministic("mu_LatentBehavioralIntention",
                                   (beta_SI_BI*mu_LatentSocialInfluence + 
                                    beta_EE_BI*mu_LatentEffortExpectancy + 
                                    beta_PE_BI*mu_LatentPerformanceExpectancy + 
                                    beta_AIA_BI*LatentAIAnxiety))
    cutpoints_BI_1 = numpyro.sample("cutpoints_BI_1", 
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]), 
                                  OrderedTransform()),)-2.5
    cutpoints_BI_2 = numpyro.sample("cutpoints_BI_2",
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]),
                                  OrderedTransform()),)-2.5
    cutpoints_BI_3 = numpyro.sample("cutpoints_BI_3",
                                  dist.TransformedDistribution(dist.Normal(0, 0.5).expand([6]),
                                  OrderedTransform()),)-2.5
    BI_1 = numpyro.sample('BI_1', dist.OrderedLogistic(mu_LatentBehavioralIntention, cutpoints_BI_1), obs=BI_1)
    BI_2 = numpyro.sample('BI_2', dist.OrderedLogistic(mu_LatentBehavioralIntention, cutpoints_BI_2), obs=BI_2)
    BI_3 = numpyro.sample('BI_3', dist.OrderedLogistic(mu_LatentBehavioralIntention, cutpoints_BI_3), obs=BI_3)    

    # ####Part 2: Definition of the Experiment###   
    beta_AA = numpyro.sample('beta_AA', dist.Normal(0, 0.5)) # our parameter of interest = Algorithm Aversion
    beta_deltaTrue = numpyro.sample('beta_deltaTrue', dist.Normal(0, 0.5))  # Coef determining how percentage delta performance is mapped in WTP
    Delta_WTP_Q0_relTo90 = numpyro.sample("Delta_WTP_Q0_relTo90", dist.Normal(beta_AA*LatentAIAnxiety+beta_deltaTrue*True_DeltaPerformance_Q0,0.2), obs=Delta_WTP_Q0_relTo90) #set SD to 0.3 to ensure it only ranges from -1 to 1
    Delta_WTP_Q1_relTo90 = numpyro.sample("Delta_WTP_Q1_relTo90", dist.Normal(beta_AA*LatentAIAnxiety+beta_deltaTrue*True_DeltaPerformance_Q1,0.2), obs=Delta_WTP_Q1_relTo90)
    Delta_WTP_Q2_relTo90 = numpyro.sample("Delta_WTP_Q2_relTo90", dist.Normal(beta_AA*LatentAIAnxiety+beta_deltaTrue*True_DeltaPerformance_Q2,0.2), obs=Delta_WTP_Q2_relTo90)
       

#%%
#####################################################
#Step 2: Create synthetic data with H1 true/not true#
#####################################################

#%%creating synthetic data with same sample size as real data of 250
n_samples = 250

#%%add true_deltaperformance as a vector taking randomly one of the possible values as defined in experiment
possible_values_DeltaTrue = np.array([0, -0.05, -0.10, 0.05, 0.10])
True_DeltaPerformance_Q0 = np.random.choice(possible_values_DeltaTrue, size=(n_samples), replace=True)
True_DeltaPerformance_Q1 = np.random.choice(possible_values_DeltaTrue, size=(n_samples), replace=True)
True_DeltaPerformance_Q2 = np.random.choice(possible_values_DeltaTrue, size=(n_samples), replace=True)          
#same for other characteristics thata re obtained via likert-scale
possible_values_TechEngagement = np.array([1, 2, 3, 4, 5, 6, 7])
TechEngagement = np.random.choice(possible_values_TechEngagement, size=(n_samples), replace=True)
possible_values_DSTExperience = np.array([0, 1, 2, 3, 4, 5, 6, 7])
DSTExperience = np.random.choice(possible_values_DSTExperience, size=(n_samples), replace=True)
possible_values_RiskPref = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
RiskPref = np.random.choice(possible_values_RiskPref, size=(n_samples), replace=True)
possible_values_AdvisorExperience = np.array([0, 1, 2, 3, 4, 5, 6, 7])
AdvisorExperience = np.random.choice(possible_values_AdvisorExperience, size=(n_samples), replace=True)
possible_values_Farmsize = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
Farmsize = np.random.choice(possible_values_Farmsize, size=(n_samples), replace=True)

#standardize categorical variables
TechEngagement = (TechEngagement - np.mean(TechEngagement))/np.std(TechEngagement)
DSTExperience = (DSTExperience - np.mean(DSTExperience))/np.std(DSTExperience)
RiskPref = (RiskPref - np.mean(RiskPref))/np.std(RiskPref)
AdvisorExperience = (AdvisorExperience - np.mean(AdvisorExperience))/np.std(AdvisorExperience)
Farmsize = (Farmsize - np.mean(Farmsize))/np.std(Farmsize)


#%% create dictionary with "exogenous" data
dict_characteristics = dict(
  Age = np.random.normal(0, 1, n_samples),  # Normally distributed with mean at 0
  Farmsize = Farmsize, #categorical
  TechEngagement = TechEngagement, #categorical
  RiskPref = RiskPref, #categorical
  DSTExperience = DSTExperience, #categorical
  AdvisorExperience = AdvisorExperience, #categorical
  True_DeltaPerformance_Q0 = True_DeltaPerformance_Q0,
  True_DeltaPerformance_Q1 = True_DeltaPerformance_Q1,
  True_DeltaPerformance_Q2 = True_DeltaPerformance_Q2)

print(dict_characteristics) 

#%% sample from prior and use coefficients for conditioning (or set manually)

prior_predictive = Predictive(model, num_samples=1)
prior_sample_for_shape = prior_predictive(rng_key, **dict_characteristics)

#%%conditioning
#-------------------------------------------------------
#2.1. Create Synthetic data where H1 is true (AA exists)
#-------------------------------------------------------

#setting coefficients in order to make our Hypothesis true (and create the synthetic data as expected)
#in addition we condition all betas to ensure our posterior predicitve only shows the AIA effect by using a prior sample to "choose" the value to condition on
coefTrue_H1 = { 'beta_deltaTrue': 1, #True Delta Performance has a positive effect on Perceived Delta Performance
                'beta_AA': -0.5, #AI Anxiety has a negative effect on Perceived Delta Performance = Algorithm Aversion
                'beta_Age_AIA': prior_sample_for_shape['beta_Age_AIA'][0],
                'beta_Farmsize_AIA': prior_sample_for_shape['beta_Farmsize_AIA'][0],
                'beta_RiskPref_AIA': prior_sample_for_shape['beta_RiskPref_AIA'][0],
                'beta_TechEngagement_AIA': prior_sample_for_shape['beta_TechEngagement_AIA'][0],
                'beta_DSTExperience_AIA': prior_sample_for_shape['beta_DSTExperience_AIA'][0],
                'beta_AdvisorExperience_AIA': prior_sample_for_shape['beta_AdvisorExperience_AIA'][0],
                'cutpoints_AIA_1': prior_sample_for_shape['cutpoints_AIA_1'][0],
                'cutpoints_AIA_2': prior_sample_for_shape['cutpoints_AIA_2'][0],
                'cutpoints_AIA_3': prior_sample_for_shape['cutpoints_AIA_3'][0],
                'cutpoints_AIA_4': prior_sample_for_shape['cutpoints_AIA_4'][0],
                'cutpoints_AIA_5': prior_sample_for_shape['cutpoints_AIA_5'][0],
                'cutpoints_AIA_6': prior_sample_for_shape['cutpoints_AIA_6'][0],
                'cutpoints_AIA_7': prior_sample_for_shape['cutpoints_AIA_7'][0],
                'cutpoints_AIA_8': prior_sample_for_shape['cutpoints_AIA_8'][0],
                'cutpoints_AIA_9': prior_sample_for_shape['cutpoints_AIA_9'][0],
                'cutpoints_AIA_10': prior_sample_for_shape['cutpoints_AIA_10'][0],
                'cutpoints_AIA_11': prior_sample_for_shape['cutpoints_AIA_11'][0],
                'cutpoints_AIA_12': prior_sample_for_shape['cutpoints_AIA_12'][0],
                'cutpoints_AIA_13': prior_sample_for_shape['cutpoints_AIA_13'][0],
                'cutpoints_AIA_14': prior_sample_for_shape['cutpoints_AIA_14'][0],
                'cutpoints_AIA_15': prior_sample_for_shape['cutpoints_AIA_15'][0],
                'cutpoints_AIA_16': prior_sample_for_shape['cutpoints_AIA_16'][0],
                'alpha_idvConst_AIA': prior_sample_for_shape['alpha_idvConst_AIA'][0], 
                'beta_AIA_BI': -0.5, #AI Anxiety has a negative effect on Behavioral Intention= Algorithm Aversion
                'beta_PE_BI': 1  #Performance Expectancy has a positive effect on Behavioral Intention               
            }


#%% Create the Predictive object for prior predictive sampling when H1 is true
condition_model_H1 = numpyro.handlers.condition(model, data=coefTrue_H1)
#%%with conditioning
cond_prior_predictive_H1 = Predictive(condition_model_H1, num_samples=1)

#%%Prior predicitve distribution
rng_key, rng_key_ = random.split(rng_key)
prior_predictions_H1 = cond_prior_predictive_H1(rng_key, **dict_characteristics)


#-------------------------------------------------
##Run some prior predicitve checks for H1 is true  
#-------------------------------------------------

#%% create Plot for the prior predictive distribution of Delta WTP when H1 is true as in pregresitration (Figure 4)
colors = [(0, "green"), (0.5, "blue"), (1, "orange")]  # Define the colors at normalized points (0, 0.5, 1)
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

latent_anxiety_values = np.mean(prior_predictions_H1['LatentAIAnxiety'], axis=0)
norm = mcolors.Normalize(vmin=np.min(latent_anxiety_values), vmax=np.max(latent_anxiety_values))

fig, ax = plt.subplots()

# Plot for each sample of LatentAIAnxiety
for i in range(prior_predictions_H1['LatentAIAnxiety'].shape[1]):
    # Compute dWtpHat
    dWtpHat = (np.mean(prior_predictions_H1['beta_AA']) * latent_anxiety_values[i] 
               + possible_values_DeltaTrue * np.mean(prior_predictions_H1['beta_deltaTrue']))
    
    # Get a color for this line based on the normalized LatentAIAnxiety value
    color = cmap(norm(latent_anxiety_values[i]))
    
    # Plot with the corresponding color
    ax.plot(possible_values_DeltaTrue, dWtpHat, color=color, alpha=1)

# Create a dummy mappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array needed for ScalarMappable

# Add colorbar explicitly to the figure
plt.colorbar(sm, ax=ax, label='LatentAIAnxiety')
plt.ylim(-3, 3)  # Set the y-axis limits to match the range of dWtpHat
plt.title('Prior Predictive Distribution of Delta WTP when H1 is true')
plt.xlabel('Delta True')
plt.ylabel('Delta WTP')
# Set the x-ticks to show only -0.1, -0.05, 0, 0.05, and 0.1
ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
ax.set_xticklabels(['-0.1', '-0.05', '0', '0.05', '0.1'])  # Optional, you can also customize the labels

# Show the plot
plt.show()


#%%we save dict characteristics and the prior predicitve samples for H1 is true in one dictionary
keys_to_take = ['AIA_1', 'AIA_2', 'AIA_3', 'AIA_4', 'AIA_5', 'AIA_6', 'AIA_7', 'AIA_8', 'AIA_9', 
                   'AIA_10', 'AIA_11', 'AIA_12', 'AIA_13', 'AIA_14', 'AIA_15', 'AIA_16', 'SI_1', 'SI_2', 
                   'PE_1', 'PE_2', 'PE_3', 'PE_4', 'EE_1', 'EE_2', 'EE_3', 'BI_1', 'BI_2', 'BI_3', 
                   'Delta_WTP_Q0_relTo90', 'Delta_WTP_Q1_relTo90', 'Delta_WTP_Q2_relTo90']

dict_synthetic_H1True ={**dict_characteristics, **{k: v for k, v in prior_predictions_H1.items() if k in keys_to_take}}
print(dict_synthetic_H1True)

# %%save dictionary also as data frame in csv
# Flatten the dictionary values if they are multi-dimensional arrays
dict_synthetic_H1True_flattened = {key: np.ravel(value) for key, value in dict_synthetic_H1True.items()}
# Convert the flattened dictionary to a DataFrame
df_synthetic_H1True = pd.DataFrame(dict_synthetic_H1True_flattened)
#%% Save the DataFrame as a CSV file for use in R code
# Ensure the directory exists
os.makedirs('SyntheticData', exist_ok=True)
df_synthetic_H1True.to_csv('SyntheticData/synthetic_data_H1True.csv', index=False)



#-----------------------------------------------------------------
#2.2. Create Synthetic data where H1 is not true (AA doesn't exists)
#-----------------------------------------------------------------
#%%
coefTrue_NoH1 = { 'beta_deltaTrue': 1, #True Delta Performance has a positive effect on Perceived Delta Performance
                'beta_AA': 0, #AI Anxiety has a negative effect on Perceived Delta Performance = Algorithm Aversion
                'beta_Age_AIA': prior_sample_for_shape['beta_Age_AIA'][0],
                'beta_Farmsize_AIA': prior_sample_for_shape['beta_Farmsize_AIA'][0],
                'beta_RiskPref_AIA': prior_sample_for_shape['beta_RiskPref_AIA'][0],
                'beta_TechEngagement_AIA': prior_sample_for_shape['beta_TechEngagement_AIA'][0],
                'beta_DSTExperience_AIA': prior_sample_for_shape['beta_DSTExperience_AIA'][0],
                'beta_AdvisorExperience_AIA': prior_sample_for_shape['beta_AdvisorExperience_AIA'][0],
                'cutpoints_AIA_1': prior_sample_for_shape['cutpoints_AIA_1'][0],
                'cutpoints_AIA_2': prior_sample_for_shape['cutpoints_AIA_2'][0],
                'cutpoints_AIA_3': prior_sample_for_shape['cutpoints_AIA_3'][0],
                'cutpoints_AIA_4': prior_sample_for_shape['cutpoints_AIA_4'][0],
                'cutpoints_AIA_5': prior_sample_for_shape['cutpoints_AIA_5'][0],
                'cutpoints_AIA_6': prior_sample_for_shape['cutpoints_AIA_6'][0],
                'cutpoints_AIA_7': prior_sample_for_shape['cutpoints_AIA_7'][0],
                'cutpoints_AIA_8': prior_sample_for_shape['cutpoints_AIA_8'][0],
                'cutpoints_AIA_9': prior_sample_for_shape['cutpoints_AIA_9'][0],
                'cutpoints_AIA_10': prior_sample_for_shape['cutpoints_AIA_10'][0],
                'cutpoints_AIA_11': prior_sample_for_shape['cutpoints_AIA_11'][0],
                'cutpoints_AIA_12': prior_sample_for_shape['cutpoints_AIA_12'][0],
                'cutpoints_AIA_13': prior_sample_for_shape['cutpoints_AIA_13'][0],
                'cutpoints_AIA_14': prior_sample_for_shape['cutpoints_AIA_14'][0],
                'cutpoints_AIA_15': prior_sample_for_shape['cutpoints_AIA_15'][0],
                'cutpoints_AIA_16': prior_sample_for_shape['cutpoints_AIA_16'][0],
                'alpha_idvConst_AIA': prior_sample_for_shape['alpha_idvConst_AIA'][0], 
                'beta_AIA_BI': 0, #AI Anxiety has no effect on Behavioral Intention
                'beta_PE_BI': 1 #Performance Expectancy has a positive effect on Behavioral Intention                
            }


#%% Create the Predictive object for prior predictive sampling when H1 is true
condition_model_NoH1 = numpyro.handlers.condition(model, data=coefTrue_NoH1)
#%%with conditioning
cond_prior_predictive_NoH1 = Predictive(condition_model_NoH1, num_samples=1)

#%%Prior predicitve distribution
rng_key, rng_key_ = random.split(rng_key)
prior_predictions_NoH1 = cond_prior_predictive_NoH1(rng_key, **dict_characteristics)


#-------------------------------------------------
##Run some prior predicitve checks for H1 is NOT true
#-------------------------------------------------

#%% create Plot for the prior predictive distribution of Delta WTP when H1 is true as in pregresitration (Figure 4)
colors = [(0, "green"), (0.5, "blue"), (1, "orange")]  # Define the colors at normalized points (0, 0.5, 1)
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

latent_anxiety_values = np.mean(prior_predictions_NoH1['LatentAIAnxiety'], axis=0)
norm = mcolors.Normalize(vmin=np.min(latent_anxiety_values), vmax=np.max(latent_anxiety_values))

fig, ax = plt.subplots()

# Plot for each sample of LatentAIAnxiety
for i in range(prior_predictions_NoH1['LatentAIAnxiety'].shape[1]):
    # Compute dWtpHat
    dWtpHat = (np.mean(prior_predictions_NoH1['beta_AA']) * latent_anxiety_values[i] 
               + possible_values_DeltaTrue * np.mean(prior_predictions_NoH1['beta_deltaTrue']))
    
    # Get a color for this line based on the normalized LatentAIAnxiety value
    color = cmap(norm(latent_anxiety_values[i]))
    
    # Plot with the corresponding color
    ax.plot(possible_values_DeltaTrue, dWtpHat, color=color, alpha=1)

# Create a dummy mappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array needed for ScalarMappable

# Add colorbar explicitly to the figure
plt.colorbar(sm, ax=ax, label='LatentAIAnxiety')
plt.ylim(-3, 3)  # Set the y-axis limits to match the range of dWtpHat
plt.title('Prior Predictive Distribution of Delta WTP when H1 is not true')
plt.xlabel('Delta True')
plt.ylabel('Delta WTP')
# Set the x-ticks to show only -0.1, -0.05, 0, 0.05, and 0.1
ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
ax.set_xticklabels(['-0.1', '-0.05', '0', '0.05', '0.1'])  # Optional, you can also customize the labels

# Show the plot
plt.show()


#%%we save dict characteristics and the prior predicitve samples for H1 is true in one dictionary
keys_to_take = ['AIA_1', 'AIA_2', 'AIA_3', 'AIA_4', 'AIA_5', 'AIA_6', 'AIA_7', 'AIA_8', 'AIA_9', 
                   'AIA_10', 'AIA_11', 'AIA_12', 'AIA_13', 'AIA_14', 'AIA_15', 'AIA_16', 'SI_1', 'SI_2', 
                   'PE_1', 'PE_2', 'PE_3', 'PE_4', 'EE_1', 'EE_2', 'EE_3', 'BI_1', 'BI_2', 'BI_3', 
                   'Delta_WTP_Q0_relTo90', 'Delta_WTP_Q1_relTo90', 'Delta_WTP_Q2_relTo90']

dict_synthetic_NoH1 ={**dict_characteristics, **{k: v for k, v in prior_predictions_NoH1.items() if k in keys_to_take}}
print(dict_synthetic_NoH1)

#%%save as csv as well
# Flatten the dictionary values if they are multi-dimensional arrays
dict_synthetic_NoH1_flattened = {key: np.ravel(value) for key, value in dict_synthetic_NoH1.items()}
# Convert the flattened dictionary to a DataFrame
df_synthetic_NoH1 = pd.DataFrame(dict_synthetic_NoH1_flattened)
#%% Save the DataFrame as a CSV file
# Ensure the directory exists
os.makedirs('SyntheticData', exist_ok=True)
df_synthetic_NoH1.to_csv('SyntheticData/synthetic_data_NoH1.csv', index=False)


############################################
#Step 3: Run mcmc with synthetic data      #
############################################

#-----------------------------------------------------
#3.1. With Synthetic data where H1 is true (AA exists)
#-----------------------------------------------------


# %%Instantiate a `MCMC` object using a NUTS sampler
# Set kernel
kernel = NUTS(model, init_strategy = numpyro.infer.init_to_feasible)
mcmc_synthetic_H1True = MCMC(sampler=kernel,
            num_warmup=1000,
            num_samples=1000,
            num_chains=2)
# Run the MCMC sampler and collect samples
rng_key, rng_key_ = random.split(rng_key)
mcmc_synthetic_H1True.run(rng_key=rng_key,**dict_synthetic_H1True)
#%%
mcmc_synthetic_H1True.print_summary()
samples_mcmc_H1 = mcmc_synthetic_H1True.get_samples(group_by_chain=False)
print(samples_mcmc_H1.keys())

#------------------------------------------------------------------
#3.2. With Synthetic data where H1 is not true (AA doesn't exist)
#------------------------------------------------------------------

# %%Instantiate a `MCMC` object using a NUTS sampler
# Set kernel
kernel = NUTS(model, init_strategy = numpyro.infer.init_to_feasible)
mcmc_synthetic_NoH1 = MCMC(sampler=kernel,
            num_warmup=1000,
            num_samples=1000,
            num_chains=2)
# Run the MCMC sampler and collect samples
rng_key, rng_key_ = random.split(rng_key)
mcmc_synthetic_NoH1.run(rng_key=rng_key,**dict_synthetic_NoH1)
#%%
mcmc_synthetic_NoH1.print_summary()
samples_mcmc_NoH1 = mcmc_synthetic_NoH1.get_samples(group_by_chain=False)
print(samples_mcmc_NoH1.keys())



#-----------------------------------------------------
##3.3. run posterior predicitve and model checks#
#-----------------------------------------------------

#get posterior predictive for both data sets

#%%1. H1 true
rng_key, rng_key_ = random.split(rng_key)
posterior_predictive_H1 = Predictive(model, samples_mcmc_H1)
posterior_predictions_H1 = posterior_predictive_H1(rng_key, **dict_characteristics)

#%%2. H1 not true
rng_key, rng_key_ = random.split(rng_key)
posterior_predictive_NoH1 = Predictive(model, samples_mcmc_NoH1)
posterior_predictions_NoH1 = posterior_predictive_NoH1(rng_key, **dict_characteristics)


#%%plot the posterior predictive checks to create plots as in preregsitration (Figure 4)

#%%1. for H1 true
colors = [(0, "green"), (0.5, "blue"), (1, "orange")]  # Define the colors at normalized points (0, 0.5, 1)
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

latent_anxiety_values = np.mean(samples_mcmc_NoH1['LatentAIAnxiety'], axis=0)
norm = mcolors.Normalize(vmin=np.min(latent_anxiety_values), vmax=np.max(latent_anxiety_values))

fig, ax = plt.subplots()

# Plot for each sample of LatentAIAnxiety
for i in range(samples_mcmc_H1['LatentAIAnxiety'].shape[1]):
    # Compute dWtpHat
    dWtpHat = (np.mean(samples_mcmc_H1['beta_AA']) * latent_anxiety_values[i] 
               + possible_values_DeltaTrue * np.mean(samples_mcmc_H1['beta_deltaTrue']))
        # Get a color for this line based on the normalized LatentAIAnxiety value
    color = cmap(norm(latent_anxiety_values[i]))
        # Plot with the corresponding color
    ax.plot(possible_values_DeltaTrue, dWtpHat, color=color, alpha=1)
# Create a dummy mappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array needed for ScalarMappable

plt.colorbar(sm, ax=ax, label='LatentAIAnxiety')
plt.ylim(-3, 3)  # Set the y-axis limits to match the range of dWtpHat
plt.title('Posterior Predictive Distribution of Delta WTP when H1 is true')
plt.xlabel('Delta True')
plt.ylabel('Delta WTP')
# Set the x-ticks to show only -0.1, -0.05, 0, 0.05, and 0.1
ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
ax.set_xticklabels(['-0.1', '-0.05', '0', '0.05', '0.1'])  # Optional, you can also customize the labels

plt.show()

# %%2. for H1 not true
# Define custom colormap: green for negative, blue around 0, orange for positive
colors = [(0, "green"), (0.5, "blue"), (1, "orange")]  # Define the colors at normalized points (0, 0.5, 1)
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

latent_anxiety_values = np.mean(samples_mcmc_NoH1['LatentAIAnxiety'], axis=0)
norm = mcolors.Normalize(vmin=np.min(latent_anxiety_values), vmax=np.max(latent_anxiety_values))

fig, ax = plt.subplots()

# Plot for each sample of LatentAIAnxiety
for i in range(samples_mcmc_NoH1['LatentAIAnxiety'].shape[1]):
    # Compute dWtpHat
    dWtpHat = (np.mean(samples_mcmc_NoH1['beta_AA']) * latent_anxiety_values[i] 
               + possible_values_DeltaTrue * np.mean(samples_mcmc_NoH1['beta_deltaTrue']))
        # Get a color for this line based on the normalized LatentAIAnxiety value
    color = cmap(norm(latent_anxiety_values[i]))
        # Plot with the corresponding color
    ax.plot(possible_values_DeltaTrue, dWtpHat, color=color, alpha=1)
# Create a dummy mappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([]) 
plt.colorbar(sm, ax=ax, label='LatentAIAnxiety')
plt.ylim(-3, 3)  # Set the y-axis limits to match the range of dWtpHat
plt.title('Posterior Predictive Distribution of Delta WTP when H1 is not true')
plt.xlabel('Delta True')
plt.ylabel('Delta WTP')
# Set the x-ticks to show only -0.1, -0.05, 0, 0.05, and 0.1
ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
ax.set_xticklabels(['-0.1', '-0.05', '0', '0.05', '0.1'])  # Optional, you can also customize the labels

plt.show()

# %%plot betas
#therefore we need to get the samples from the mcmc grouped by chain

#%%1. H1 true (Figure 5&7 in Prergistration)
samples_mcmc_H1_grouped = mcmc_synthetic_H1True.get_samples(group_by_chain=True)
print(samples_mcmc_H1_grouped.keys())

numPostSamples_H1 = samples_mcmc_H1_grouped[list(samples_mcmc_H1_grouped.keys())[0]].shape[1]
prior_sampling_forPlot = Predictive(model, num_samples=numPostSamples_H1)
prior_predictions_H1_forPlot = prior_sampling_forPlot(rng_key,**dict_characteristics)
print(prior_predictions_H1_forPlot.keys())
#%% Add one dimension for num_chains
for key in prior_predictions_H1_forPlot.keys():
    prior_predictions_H1_forPlot[key] = prior_predictions_H1_forPlot[key][None,:]
prior_predictions_H1_forPlot['AIA_1'].shape
# %%# Prepare arviz dataset for plotting
dsPrior = az.convert_to_dataset(prior_predictions_H1_forPlot,
                    group='prior',)
dsPost = az.convert_to_dataset(samples_mcmc_H1_grouped,
                    group='posterior',)
xrr = az.convert_to_inference_data(dsPost)
xrr.add_groups(prior=dsPrior)
xrr


#%%for preregistration: only Algorithm Aversion betas (Figre 5)

ax = az.plot_posterior(
    xrr,
    coords={},
    group='posterior',
    var_names=['beta_AIA_BI','beta_AA'],
    label='posterior',
    ref_val=[
        prior_predictions_H1['beta_AIA_BI'].flatten().tolist()[0],
        prior_predictions_H1['beta_AA'].flatten().tolist()[0]
    ],
)
ax = az.plot_posterior(
    xrr,
    coords={},
    group='prior',
    var_names=['beta_AIA_BI','beta_AA'],
    hdi_prob='hide',
    point_estimate=None,
    color='green',
    label='prior',
    ax=ax
)

#%%trace plot for selected vars (Figure 7 in Preregistration)
az.plot_trace(mcmc_synthetic_H1True,
              var_names=['beta_AIA_BI','beta_AA',
                          'beta_deltaTrue', 'alpha_idvConst_AIA'],
              figsize=(5,5));

plt.suptitle('Trace Plot from syntehtic data where H1 is true for selected variables', y=1.09)
plt.show()


# %%same for H1 not true (Figure 6&7 in Preregistration)
samples_mcmc_NoH1_grouped = mcmc_synthetic_NoH1.get_samples(group_by_chain=True)
print(samples_mcmc_NoH1_grouped.keys())

numPostSamples_NoH1 = samples_mcmc_NoH1_grouped[list(samples_mcmc_NoH1_grouped.keys())[0]].shape[1]
prior_sampling_forPlot = Predictive(model, num_samples=numPostSamples_NoH1)
prior_predictions_NoH1_forPlot = prior_sampling_forPlot(rng_key,**dict_characteristics)
print(prior_predictions_NoH1_forPlot.keys())
#%% Add one dimension for num_chains
for key in prior_predictions_NoH1_forPlot.keys():
    prior_predictions_NoH1_forPlot[key] = prior_predictions_NoH1_forPlot[key][None,:]
prior_predictions_NoH1_forPlot['AIA_1'].shape
# %%# Prepare arviz dataset for plotting
dsPrior = az.convert_to_dataset(prior_predictions_NoH1_forPlot,
                    group='prior',)
dsPost = az.convert_to_dataset(samples_mcmc_NoH1_grouped,
                    group='posterior',)
xrr = az.convert_to_inference_data(dsPost)
xrr.add_groups(prior=dsPrior)
xrr


#%%for preregistration: only Algorithm Aversion betas (Figure 6)

ax = az.plot_posterior(
    xrr,
    coords={},
    group='posterior',
    var_names=['beta_AIA_BI','beta_AA'],
    label='posterior',
    ref_val=[
        prior_predictions_NoH1['beta_AIA_BI'].flatten().tolist()[0],
        prior_predictions_NoH1['beta_AA'].flatten().tolist()[0]
    ],
)
ax = az.plot_posterior(
    xrr,
    coords={},
    group='prior',
    var_names=['beta_AIA_BI','beta_AA'],
    hdi_prob='hide',
    point_estimate=None,
    color='green',
    label='prior',
    ax=ax
)


#%%trace plot for selected vars (Figure 7 in Preregistration)
az.plot_trace(mcmc_synthetic_NoH1,
              var_names=['beta_AIA_BI','beta_AA',
                          'beta_deltaTrue', 'alpha_idvConst_AIA'],
              figsize=(5,5));

plt.suptitle('Trace Plot from synthetic data where H1 is not true for selected variables', y=1.09)
plt.show()

#%%