import numpy as np
import emcee
import umbridge
import argparse


def log_prior(theta):
    #TODO These need to be passed as arguments to the function
    a1 = 0.02
    b1 = 0.3
    a2 = 100
    b2 = 2200
    # uniform prior over the intervals [a1, b1] and [a2, b2] for each parameter in theta
    if np.all(a1 <= theta[0]) and np.all(theta[0] <= b1) and \
       np.all(a2 <= theta[1]) and np.all(theta[1] <= b2):
        return -np.log((b1 - a1) * (b2 - a2))
    else:
        return -np.inf


def log_likelihood_Umax(theta, data, model):
    mean = data[0]
    std_dev = data[1]
    model_output = np.asarray(model([theta.tolist()])[0], dtype='float64')
    variance = std_dev ** 2
    return -0.5 *(np.log(2 * np.pi * variance)
                         + (mean - model_output) ** 2 / variance)


def log_posterior_Umax(theta, data, model):
    return log_prior(theta) + log_likelihood_Umax(theta, data, model)


def compute_mcmc(log_posterior, data, model,
                   nwalkers=20, nburn=500, nsteps=1000):
    #TODO These need to be passed as arguments to the function

    ndim = 2  # this determines the model
    
    a1, b1 = 0.02, 0.3 # Range for parameter A 
    a2, b2 = 100, 2200 # Range for parameter B
   


    # Generate starting guesses for each walker
    starting_guesses = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        starting_guesses[i, 0] = np.random.uniform(a1, b1)
        starting_guesses[i, 1] = np.random.uniform(a2, b2)
        
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

    trace, sampler, full_chain = compute_mcmc(log_posterior=log_posterior_Umax, data=data, model=model)
    np.save("mcmc_trace.npy", trace)
    np.save("mcmc_full_chain.npy", full_chain)

