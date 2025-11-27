import yaml
import numpy as np
import emcee
import pandas as pd
from scipy.stats import uniform, norm, truncnorm
import umbridge
import argparse
import os
import corner



def get_distribution(prior_config:dict):
    name = prior_config["name"]
    distribution = prior_config["distribution"]

    if distribution["type"] not in ["uniform", "normal", "truncated_normal"]:
        print(f"Distribution type {distribution["type"]} not yet supported")
        print("Aborting MCMC Calibration")
        exit(1)

    if distribution["type"] == "uniform":
        if not all(attribute in distribution["attribute"] for attribute in ("upper_bound", "lower_bound")):
            print(f"Incorrect definition of prior for parameter '{name}'",
                f"\nA uniform prior requires `lower_bound` and `upper_bound` attributes")
            print("Aborting MCMC Calibration")
            exit(1)
        else:
            loc=distribution["attribute"]["lower_bound"],
            scale=distribution["attribute"]["upper_bound"]-distribution["attribute"]["lower_bound"]
            return uniform(loc, scale)
    elif distribution["type"] == "normal":
        if not all(attribute in distribution["attribute"] for attribute in ("loc", "scale")):
            print(f"Incorrect definition of prior for parameter '{name}'",
                f"\nA {distribution["type"]} prior requires `loc` and `scale` attributes")
            print("Aborting MCMC Calibration")
            exit(1)
        else: 
            loc=distribution["attribute"]["loc"],
            scale=distribution["attribute"]["scale"]
            return norm(loc, scale)
    elif distribution["type"] == "truncated_normal":
        if not (all(attribute in distribution["attribute"] for attribute in ("loc", "scale"))
                and any(attribute in distribution["attribute"] for attribute in ("lower_bound", "upper_bound"))):
            print(f"Incorrect definition of prior for parameter '{name}'",
                f"\nA {distribution["type"]} prior requires `loc`, `scale`, `lower_bound` and/or `upper_bound` attributes")
            print("Aborting MCMC Calibration")
            exit(1)
        else: 
            lower_bound = distribution["attribute"].get("lower_bound", -np.inf)
            upper_bound = distribution["attribute"].get("upper_bound", np.inf)
            loc=distribution["attribute"]["loc"]
            scale=distribution["attribute"]["scale"]
            a, b = (lower_bound - loc) / scale, (upper_bound - loc) / scale
            return truncnorm(a, b, loc, scale)

def get_prior_distributions(priors:dict, parameters, noise_parameters, calibrate_noise=False):
    if calibrate_noise==True:
        parameters += noise_parameters
    prior_names = [prior["name"] for prior in priors]
    distributions = []
    for parameter in parameters:
        distributions.append(get_distribution(priors[prior_names.index(parameter)]))
    return distributions

def get_log_prior(prior_distributions: list):
    def log_prior(parameters) -> float:
        nonlocal prior_distributions
        log_prior=0.0
        for i, theta in enumerate(parameters):
            support = prior_distributions[i].support()
            if not (support[0] <= theta <= support[1]):
                return -np.inf
            else:
                log_prior+= prior_distributions[i].logpdf(theta)
        return log_prior
    return log_prior

def get_log_likelihood(model: callable, data : np.ndarray, n_noise_parameters=1, calibrate_noise=True, noise_sigma=None, distribution_type="normal") -> float:
    
    if calibrate_noise and n_noise_parameters > 1:
        print(f"Only 1 noise parameter supported for log_likelihood")
        print("Aborting MCMC Calibration")
        exit(1)
    if not calibrate_noise and noise_sigma==None:
        print(f"log_likelihood function requires `noise_sigma` to be provided")
        print("Aborting MCMC Calibration")
        exit(1)
    def log_likelihood(parameters):
        nonlocal model, data, compute_log_likelihood, n_noise_parameters, calibrate_noise, noise_sigma
        if calibrate_noise:
            noise_sigma = np.asarray(parameters[-n_noise_parameters:])
            model_parameters = parameters[:-n_noise_parameters]
            if any(sigma <= 0.0 for sigma in noise_sigma):
                return -np.inf

        try:
            prediction_mean = np.asarray(model([[*model_parameters]]))
        except Exception as e:
            return -np.inf
        
        if prediction_mean.shape != data.shape:
            raise ValueError("shape of model predictions does not match observations")

        return compute_log_likelihood(data, prediction_mean, noise_sigma)
    
    if distribution_type == "normal":
        def compute_log_likelihood(data, prediction_mean, noise_sigma):
            # Handwritten function to efficiently compute "normal" log likelihood
            # Only handles scalar noise_sigma
            variance = noise_sigma * noise_sigma            
            return -0.5 * (np.log(2.0 * np.pi * variance) + ((data - prediction_mean) ** 2) / variance).sum()
        return log_likelihood

    else: 
        print(f"Distribution type {distribution_type} not yet supported for log_likelihood")
        print("Aborting MCMC Calibration")
        exit(1)

def get_log_posterior(log_prior: callable, log_likelihood: callable) -> np.ndarray:

    def log_posterior(parameters):
        log_posterior = log_prior(parameters) + log_likelihood(parameters)
        return log_posterior
    return log_posterior

def initialize_walkers(nwalkers: int, prior_distributions) -> np.ndarray:
    nparameters = len(prior_distributions)
    initial_positions = np.zeros((nwalkers, nparameters))

    for i in range(nparameters):
        initial_positions[:, i] = prior_distributions[i].rvs(size=nwalkers)

    return initial_positions

def perform_mcmc(prior_distributions, log_posterior, nwalkers=50, nburn=2500, nsteps=5000):
    
    # Initialize Walkers        
    initial_positions = initialize_walkers(nwalkers, prior_distributions)
    
    # Setup Sampler
    ndim = len(prior_distributions)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
    
    # Run Calibration
    sampler.run_mcmc(initial_positions, nsteps, progress=True)

    # Extract Results (post burn-in)
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    lnprob = sampler.lnprobability[:, nburn:]
    samples = sampler.chain[:, nburn:, :]
    
    return trace, sampler, lnprob, samples

def parse_arguments():
    parser = argparse.ArgumentParser(description='MCMC Calibration with `emcee`')
    parser.add_argument('--config', type=str,
                        help='YAML file for Configuration Parameters')
    parser.add_argument('--data', type=str,
                        help='Path to Data File')
    parser.add_argument('--port', type=int, default=4242,
                       help='Server port')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config_file = args.config
    data_file = args.data
    port = args.port

    with open(config_file) as f:
        config = yaml.safe_load(f)


    # Load Data 
    required_columns = config["calibration"]["data"]
    df = pd.read_csv(
        data_file,
        usecols = required_columns
        )
    data = df[required_columns].to_numpy()

    # Connect Model
    try:
        model_name = config["model"]["name"]
        model = umbridge.HTTPModel(f"http://localhost:{args.port}", model_name)
        print(f"Connected to model: {model_name}")
    except Exception as e:
        print(f"Error connecting to model: {e}")
        exit(1)

    # Define Prior Distributions
    prior_distributions = get_prior_distributions(config["calibration"]["priors"],
                                                parameters=config["calibration"]["parameters"],
                                                noise_parameters=config["calibration"]["noise_parameters"],
                                                calibrate_noise=config["calibration"]["calibrate_noise"])

    # Get Prior and Likelihood Functions
    log_prior_func=get_log_prior(prior_distributions)
    log_likelihood_func=get_log_likelihood(model,data,
                                    calibrate_noise=config["calibration"]["calibrate_noise"],
                                    n_noise_parameters=len(config["calibration"]["noise_parameters"]),
                                    noise_sigma = config["calibration"].get("noise_sigma", None))

    # Define Log Posterior Funciton
    log_posterior_func= get_log_posterior(
        log_prior=log_prior_func, 
        log_likelihood=log_likelihood_func
        )
    

    # Perform MCMC Calibration
    trace, sampler, lnprob, samples = perform_mcmc(prior_distributions, log_posterior_func, 
                nwalkers=config["calibration"]["nwalkers"], 
                nburn=config["calibration"]["nburn"], 
                nsteps=config["calibration"]["nsteps"])
    
    
    
    print(f"MCMC completed. Trace shape: {trace.shape}")

    # Save results
    data_basename, ext = os.path.splitext(args.data)
    np.savez(f"calibration_output_{model_name}_{data_basename}.npz", 
                trace=trace, samples=samples, lnprob=lnprob)
    print(f"Results saved to calibration_output_{model_name}_{data_basename}.npz")

    corner_plot = corner.corner(trace, labels=[prior["name"] for prior in config["calibration"]["priors"]], show_titles=True)
    corner_plot.savefig(f"corner_plot_{model_name}_{data_basename}")
    print(f"Corner Plot saved to corner_plot_{model_name}_{data_basename}.png")


