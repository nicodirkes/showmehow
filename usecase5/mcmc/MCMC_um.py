import numpy as np
import emcee
import umbridge
import argparse


def log_prior_parameters(theta):
    """
    Log prior for the 3 model parameters: A, alpha, beta
    """
    # Bounds for the parameters A, alpha and beta
    bounds = np.array([
        [0.0, 1e-3],  # A in [0, 1e-3]
        [0.0, 3.0],   # alpha in [0, 3]
        [0.0, 1.0]    # beta in [0, 1]
    ])
    
    # uniform prior over the intervals given in the bounds
    if not np.all(bounds[:, 0] <= theta) or not np.all(theta <= bounds[:, 1]):
        return -np.inf
    
    # Calculate the "volume" of the uniform distribution
    # as the product of (upper_bound - lower_bound) for each parameter
    volume = np.product(bounds[:, 1] - bounds[:, 0])
    
    # Return the negative log of this volume (equivalent to log of uniform prior)
    return -np.log(volume)


def log_prior_parameters_with_sigma(theta):
    """
    Log prior for the 4 parameters: A, alpha, beta, sigma^2
    """
    # Bounds for the parameters A, alpha, beta, and sigma^2
    bounds = np.array([
        [0.0, 1e-3],  # A in [0, 1e-3]
        [0.0, 3.0],   # alpha in [0, 3]
        [0.0, 1.0],   # beta in [0, 1]
        [1e-6, 10.0]  # sigma^2 in [1e-6, 10.0] - adjust bounds as needed
    ])
    
    # uniform prior over the intervals given in the bounds
    if not np.all(bounds[:, 0] <= theta) or not np.all(theta <= bounds[:, 1]):
        return -np.inf
    
    # Calculate the "volume" of the uniform distribution
    volume = np.product(bounds[:, 1] - bounds[:, 0])
    
    # Return the negative log of this volume
    return -np.log(volume)

def safe_model_evaluation(model, theta):
    """
    Safely evaluate the model and handle potential errors
    """
    try:
        model_output = np.asarray(model([theta.tolist()])[0], dtype='float64')
        if not np.all(np.isfinite(model_output)):
            return None
        return model_output
    except Exception:
        return None
    
def log_likelihood_homoscedastic(theta, data, model):
    """
    Homoscedastic likelihood without calibrating the model parameter
    Expects data = [mean, std_dev]
    """
    mean = data[0]  # compute the mean 
    std_dev = data[1]  # compute the standard deviation
    
    model_output = safe_model_evaluation(model, theta)
    if model_output is None:
        return -np.inf
    
    variance = std_dev ** 2
    
    # Handle case where variance is zero or very small
    if np.any(variance <= 0):
        return -np.inf
    
    # Sum over all data points
    log_lik = -0.5 * np.sum(np.log(2 * np.pi * variance) + 
                           (mean - model_output) ** 2 / variance)
    
    return log_lik
    
def log_likelihood_homoscedastic_calibrating_sigma(theta, data, model):
    """
    Homoscedastic likelihood with calibrating the model parameter
    Expects data = [mean] (only mean, sigma is calibrated)
    """
    mean = data[0]  # compute the mean 

    # Extract model parameters and noise variance
    model_params = theta[:-1]  # All parameters except the last one
    sigma_squared = theta[-1]  # Last parameter is σ²
    
    if sigma_squared <= 0:
        return -np.inf
    
    model_output = safe_model_evaluation(model, model_params)
    if model_output is None:
        return -np.inf
    
    # Sum over all data points
    log_lik = -0.5 * np.sum(np.log(2 * np.pi * sigma_squared) + 
                           (mean - model_output) ** 2 / sigma_squared)
    
    return log_lik

def prepare_heteroscedastic_data(replicates_data):
    """
    Prepare data for heteroscedastic likelihood from raw replicates
    
    Parameters:
    -----------
    replicates_data : array-like, shape (n_data, 3)
        Raw replicate measurements (3 measurements per data point)
    
    Returns:
    --------
    tuple : (mean_observations, empirical_variances)
    """
    # Calculate mean and variance for each data point
    mean_obs = np.mean(replicates_data, axis=1)
    
    # Sample variance (using N-1 denominator)
    empirical_var = np.var(replicates_data, axis=1, ddof=1)
    
    # Handle cases where variance is very small (numerical stability)
    min_var = 1e-10
    empirical_var = np.maximum(empirical_var, min_var)
    
    return mean_obs, empirical_var

def log_likelihood_heteroscedastic(theta, data, model):
    """
    Heteroscedastic likelihood function
    Expects data to be raw replicates or processed (mean, variance) data
    """
    
    # Data is raw replicates, need to process
    mean_obs, empirical_var = prepare_heteroscedastic_data(data)
    
    model_output = safe_model_evaluation(model, theta)
    if model_output is None:
        return -np.inf
    
    # Handle cases where empirical variance is zero or negative
    if np.any(empirical_var <= 0):
        return -np.inf
    
    # Sum over all data points
    log_lik = -0.5 * np.sum(np.log(2 * np.pi * empirical_var) + 
                           (mean_obs - model_output) ** 2 / empirical_var)
    
    return log_lik

class BayesianInference:
    """
    Unified interface for different likelihood functions
    """
    
    def __init__(self, likelihood_type='homoscedastic'):
        """
        Initialize the Bayesian inference with specified likelihood type
        
        Parameters:
        -----------
        likelihood_type : str
            One of 'homoscedastic', 'homoscedastic_calibrating_sigma', 'heteroscedastic'
        """
        self.likelihood_type = likelihood_type
        self.likelihood_functions = {
            'homoscedastic': log_likelihood_homoscedastic,
            'homoscedastic_calibrating_sigma': log_likelihood_homoscedastic_calibrating_sigma,
            'heteroscedastic': log_likelihood_heteroscedastic
        }
        
        if likelihood_type not in self.likelihood_functions:
            raise ValueError(f"Unknown likelihood type: {likelihood_type}")
    
    def log_posterior(self, theta, data, model):
        """
        Compute log posterior probability
        """
        # Choose appropriate prior based on likelihood type
        if self.likelihood_type == 'homoscedastic_calibrating_sigma':
            log_prior = log_prior_parameters_with_sigma(theta)
        else:
            log_prior = log_prior_parameters(theta)
        
        if log_prior == -np.inf:
            return -np.inf
        
        # Compute likelihood
        log_lik = self.likelihood_functions[self.likelihood_type](theta, data, model)
        
        return log_prior + log_lik
    
    def compute_mcmc(self, data, model, nwalkers=20, nburn=500, nsteps=1000):
        """
        Runs an MCMC sampler with appropriate parameter bounds based on likelihood type
        
        Parameters
        ----------
        data : any
            Data relevant for evaluating the model/posterior.
        model : any
            Model definition or function relevant for evaluating the posterior.
        nwalkers : int, optional
            Number of walkers in the MCMC ensemble. Default is 20.
        nburn : int, optional
            Number of burn-in steps. Default is 500.
        nsteps : int, optional
            Number of production steps (post burn-in). Default is 1000.
        
        Returns
        -------
        trace : ndarray
            Flattened samples after the production phase,
            shaped (nwalkers * nsteps, ndim).
        sampler : emcee.EnsembleSampler
            The emcee sampler object.
        full_chain : ndarray
            Full samples (unflattened) from the production phase,
            shaped (nsteps, nwalkers, ndim).
        """
        # Determine number of parameters and bounds based on likelihood type
        if self.likelihood_type == 'homoscedastic_calibrating_sigma':
            ndim = 4
            bounds = np.array([
                [0.0, 1e-3],  # A
                [0.0, 3.0],   # alpha
                [0.0, 1.0],   # beta
                [1e-6, 10.0]  # sigma^2
            ])
        else:
            ndim = 3
            bounds = np.array([
                [0.0, 1e-3],  # A
                [0.0, 3.0],   # alpha
                [0.0, 1.0]    # beta
            ])
        
        # Generate initial positions for each walker
        starting_guesses = np.zeros((nwalkers, ndim))
        for i in range(nwalkers):
            for j in range(ndim):
                low, high = bounds[j]
                starting_guesses[i, j] = np.random.uniform(low, high)
        
        # Initialize the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_posterior, args=(data, model)
        )
        
        # ---------------------
        # BURN-IN PHASE
        # ---------------------
        print("Running burn-in...")
        state = sampler.run_mcmc(starting_guesses, nburn, progress=True)
        # Reset sampler to remove burn-in samples from memory
        sampler.reset()
        
        # ---------------------
        # PRODUCTION PHASE
        # ---------------------
        print("Running production...")
        sampler.run_mcmc(state, nsteps, progress=True)
        
        # Retrieve the chains
        full_chain = sampler.get_chain()           # unflattened
        trace = sampler.get_chain(flat=True)       # flattened
        
        return trace, sampler, full_chain
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_filename')
    parser.add_argument('surrogate_model_name')

    args = parser.parse_args()
    data = np.loadtxt(args.data_filename, delimiter=',')
    
    
    # Set up a model by connecting to URL and selecting the "forward" model
    model = umbridge.HTTPModel("http://localhost:4242", args.surrogate_model_name)
    
    inference_hom = BayesianInference('homoscedastic')
    
    trace, sampler, full_chain = inference_hom.compute_mcmc(data, model)

   
    np.save("mcmc_trace.npy", trace)
    np.save("mcmc_full_chain.npy", full_chain) 

