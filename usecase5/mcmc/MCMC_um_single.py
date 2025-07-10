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


def log_posterior(theta, data, model):
    return log_prior_parameters(theta) + log_likelihood_homoscedastic(theta, data, model)


def compute_mcmc(log_posterior, data, model,
                   nwalkers=20, nburn=500, nsteps=1000):
    #TODO These need to be passed as arguments to the function

    ndim = 2  # this determines the model
    
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
        
 
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(data, model,))
    sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    full_chain = sampler.chain[:, nburn:, :]
    
    return trace, sampler, full_chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_filename')
    parser.add_argument('surrogate_model_name')

    args = parser.parse_args()
    data = np.loadtxt(args.data_filename, delimiter=',')
    
    
    # Set up a model by connecting to URL and selecting the "forward" model
    model = umbridge.HTTPModel("http://localhost:4242", args.surrogate_model_name)

    trace, sampler, full_chain = compute_mcmc(log_posterior=log_posterior, data=data, model=model)
    np.save("mcmc_trace.npy", trace)
    np.save("mcmc_full_chain.npy", full_chain)

