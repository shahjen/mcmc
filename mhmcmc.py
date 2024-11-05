#%%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.stattools import acf as autocorr

#%%
class MHMCMC:
    def __init__(self, sampling_dist_log, proposal_dist_rng, use_hastings_ratio=False, seed=42):
        """_summary_

        Args:
            sampling_dist_log (function): distribution from which you want to sample form. Should take one arg only. (f(x): R^n -> R^n)
            proposal_dist_rng (function):
        """
        self.sampling_dist_log = sampling_dist_log
        self._sample_store = []
        self._rng= np.random.default_rng(seed)
        self.proposal_dist_rng = lambda x:proposal_dist_rng(self._rng, x)
        self._sample_size=0
        self._accepted=0
        self.use_hastings_ratio=use_hastings_ratio
        
    def set_proposal_dist_rng(self, proposal_dist_rng, use_hastings_ratio=False, hastings_ratio_fn=None):
        self.proposal_dist_rng=lambda x:proposal_dist_rng(self._rng, x)
        self.use_hastings_ratio=use_hastings_ratio
        self.hastings_ratio_fn=hastings_ratio_fn
            
    def _sample(self):
        past = np.copy(self._sample_store[self._sample_size-1])
        future = np.copy(self._sample_store[self._sample_size-1])

        possible_future = self.proposal_dist_rng(past)
        
        acceptance_check = min(self.sampling_dist_log(possible_future) - self.sampling_dist_log(past), 0) #r
        
        
        if(self.use_hastings_ratio):
            acceptance_check+=self.hastings_ratio_fn(possible_future, past, len(past))
            
        acceptance_threshold = np.log(self._rng.uniform(0,1)) #a
        if(acceptance_check > acceptance_threshold): #a<r
            future=np.copy(possible_future)
            self._accepted+=1
        self._sample_store[self._sample_size]=np.copy(future)
        self._sample_size+=1
        
    
    def sample_from_dist(self, init_val, num_samples=1000):
        self._num_samples=num_samples
        self._sample_store=np.zeros((num_samples,np.shape(init_val)[0]))
        self._sample_store[0] = init_val
        self._sample_size+=1
        self._accepted+=1
        for _ in tqdm(range(num_samples-1)):
            self._sample()
    
    def get_store(self):
        return self._sample_store[self._num_samples//10:]
        
    def refresh(self):
        self._sample_store=[]
        self._sample_size=0
        self._accepted=0
        
    def get_acceptance_ratio(self):
        return self._accepted/self._sample_size
    
    def get_mean(self):
        return np.mean(self._sample_store[self._num_samples//10:], axis=0)
    
    def get_cov(self):
        return np.cov(self._sample_store[self._num_samples//10:].T)
    
    def neff(self):
        n = self._sample_size - self._sample_size//10
        acf = autocorr(self._sample_store, nlags=n, fft=True)
        sums = 0
        for k in range(1, len(acf)):
            sums = sums + (n-k)*acf[k]/n

        return n/(1+2*sums)
    
# %%
