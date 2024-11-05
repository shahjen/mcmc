#%%
import numpy as np
from mhmcmc import MHMCMC
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf as autocorr

#%%
def random_walk():
    pass
def logbanana(x, a=3):
    return - (a *((x[1]-x[0]**2)**2+(1-x[0])**2))
def random_walk_rng(rng, theta, sigma_sc):
    if(type(theta)==int):
        return rng.multivariate_normal(np.array(theta), sigma_sc)[0]
    return rng.multivariate_normal(np.array(theta), sigma_sc)#multivariate_normal(theta, sigma, seed=seed).rvs(1)

#%%
def run_mcmc_trial(sigmas, sampling_dist, is_random_walk=True, dim=2, num_samples=100000):
    
    sampler = MHMCMC(sampling_dist_log=sampling_dist, proposal_dist_rng=random_walk_rng)
    sampler2 = MHMCMC(sampling_dist_log=sampling_dist, proposal_dist_rng=random_walk_rng, seed=43)
    # stores = np.zeros((num_samples, dim, 4))
    for i, sigma in enumerate(sigmas):
        
        if is_random_walk:
            print("Randomly waling")
            sampler.set_proposal_dist_rng(lambda rng, theta:random_walk_rng(rng, theta, sigma_sc=sigma*np.identity(len(theta))))
            sampler2.set_proposal_dist_rng(lambda rng, theta:random_walk_rng(rng, theta, sigma_sc=sigma*np.identity(len(theta))))
        else:
            hastings_ratio_fn = lambda past, possible_future, dim:(multivariate_normal(possible_future, sigma*np.identity(dim)).logpdf(past) - multivariate_normal(past, sigma*np.identity(dim)).logpdf(possible_future))
            sampler.set_proposal_dist_rng(lambda rng, theta:random_walk_rng(rng, theta, sigma_sc=sigma*np.identity(dim)), use_hastings_ratio=True, hastings_ratio_fn=hastings_ratio_fn)
            sampler2.set_proposal_dist_rng(lambda rng, theta:random_walk_rng(rng, theta, sigma_sc=sigma*np.identity(dim)), use_hastings_ratio=True, hastings_ratio_fn=hastings_ratio_fn)

        sampler.sample_from_dist(np.zeros((dim,)), num_samples)
        sampler2.sample_from_dist(np.zeros((dim,)), num_samples)
        div = (np.arange(len(sampler.get_store()))+1)
        # print(len(sampler.get_store()))
        # print(len(div))
        # print(np.cumsum(sampler.get_store(), axis=0).shape)
        # print(np.cumsum(sampler2.get_store()).shape)
        # plt.plot(np.cumsum(sampler.get_store(), axis=0)/div)
        # plt.plot(np.cumsum(sampler2.get_store(), axis=0)/div)
        # plt.show()
        plt.plot(np.linalg.norm(np.cumsum(sampler.get_store()-sampler2.get_store(), axis=0), 2, axis=1)/div)
        # plt.show()
        print(sampler.get_acceptance_ratio(), sampler2.get_acceptance_ratio())
        sampler.refresh()
        sampler2.refresh()
        
        
  
  
#%%
## Random walk
# Part a
dim=2
sigmas = [0.1, 0.5, 1.0, 2.0]
run_mcmc_trial(sigmas = sigmas, sampling_dist=lambda x:multivariate_normal(np.zeros(dim,), np.identity(dim)).logpdf(x), is_random_walk=True, num_samples=100000, dim=dim)

run_mcmc_trial(sigmas = sigmas, sampling_dist=lambda x:multivariate_normal(np.zeros(dim,), np.identity(dim)).logpdf(x), is_random_walk=False, num_samples=100000, dim=dim)

#%%
# Part b
run_mcmc_trial(sigmas = sigmas, sampling_dist=lambda x:multivariate_normal(np.zeros(dim), np.array([[10, -1], [-1, 1]])).logpdf(x), is_random_walk=True, num_samples=100000, dim=dim)

run_mcmc_trial(sigmas = sigmas, sampling_dist=lambda x:multivariate_normal(np.zeros(dim), np.array([[10, -1], [-1, 1]])).logpdf(x), is_random_walk=False, num_samples=100000, dim=dim)

#%%
# Part c
def logbanana(x, a=3):
    return - (a *((x[1]-x[0]**2)**2+(1-x[0])**2))

run_mcmc_trial(sigmas = [sigmas[1]], sampling_dist=logbanana, is_random_walk=True, num_samples=100000, dim=dim)
run_mcmc_trial(sigmas = [sigmas[1]], sampling_dist=logbanana, is_random_walk=False, num_samples=100000, dim=dim)
plt.show()

# %%
