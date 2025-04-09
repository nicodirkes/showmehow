import numpy as np
import emcee
import os
import argparse
import subprocess


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

def model_prediction(testing_input):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    load_predict_file_path = os.path.join(dir_path, "..", "surrogate", "Load_Predict.R")
    subprocess.run(f"Rscript {load_predict_file_path} {args.model_filename} {testing_input[0]} {testing_input[1]}", shell=True, check=True)


def log_likelihood_Umax(theta):
    mean = data[0]
    std_dev = data[1]
    model_prediction(theta)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_prediction_path = os.path.join(dir_path, "..", "surrogate", "model_prediction")
    model_output = np.loadtxt(model_prediction_path)
    return -0.5 *(np.log(2 * np.pi * std_dev ** 2)
                         + (mean - model_output) ** 2 / std_dev ** 2)




def log_posterior_Umax(theta):
    theta = np.asarray(theta)
    return log_prior(theta) + log_likelihood_Umax(theta)


def compute_mcmc(log_posterior,
                   nwalkers=10, nburn=250, nsteps=500):
    #TODO These need to be passed as arguments to the function
    ndim = 2  # this determines the model
    
    a1, b1 = 0.02, 0.3 # Range for parameter A
    a2, b2 = 100, 2200 # Range for parameter B
   
  

    # Generate starting guesses for each walker
    starting_guesses = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        starting_guesses[i, 0] = np.random.uniform(a1, b1)
        starting_guesses[i, 1] = np.random.uniform(a2, b2)
        
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    sampler.run_mcmc(starting_guesses, nsteps, progress=True)
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    full_chain = sampler.chain[:, nburn:, :]
    return trace, sampler, full_chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_filename')
    parser.add_argument('model_filename')

    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, args.data_filename)
    data = np.loadtxt(data_path, delimiter=',')
    
    trace, sampler, full_chain = compute_mcmc(log_posterior=log_posterior_Umax)
    trace_path = os.path.join(dir_path, "mcmc_trace.npy")
    full_chain_path = os.path.join(dir_path, "mcmc_full_chain.npy")
    np.save(trace_path, trace)
    np.save(full_chain_path, full_chain)

