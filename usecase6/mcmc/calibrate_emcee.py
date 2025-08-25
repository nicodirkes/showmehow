import numpy as np
import pandas as pd
import emcee
import umbridge
import argparse
from scipy.stats import qmc


def log_prior(theta):
    """Uniform prior over specified parameter bounds"""
    #TODO Move hard coded variables into workflow 
    a1, b1 = 1e-10, 20.0
    a2, b2 = 0.5, 2.5
    a3, b3 = 1e-2, 1.0
    
    if (a1 <= theta[0] <= b1 and 
        a2 <= theta[1] <= b2 and 
        a3 <= theta[2] <= b3):
        return -np.log((b1 - a1) * (b2 - a2) * (b3 - a3))
    else:
        return -np.inf


def log_likelihood(theta, model, data, noise_model):
    """
    Computes the log-likelihood for a Gaussian error model.

    Parameters
    ----------
    theta : array-like, shape (3,)
        Model parameters [theta0, theta1, theta2].
    model : callable
        Function model(t_exp, shear_stress, theta0, theta1, theta2) -> predictions.
    data : pandas.DataFrame
        Must contain columns: 'shear_stress', 'exposure_time', 'Mean', 'SD'.
    noise_model : str
        "homo" for homoscedastic (constant) variance,
        "hetero" for heteroscedastic variance.

    Returns
    -------
    float
        The log-likelihood value.
    """
    theta = np.asarray(theta, dtype=float)
    if len(theta) != 3:
        raise ValueError("theta must have exactly 3 elements")

    required_cols = {"shear_stress", "exposure_time", "fHb_mean", "fHb_std"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"data must contain columns: {required_cols}")

    noise_model = noise_model.lower()
    if noise_model not in {"homo", "hetero"}:
        raise ValueError("noise_model must be either 'homo' or 'hetero'")

    # Extract observed data
    y_obs = data["fHb_mean"].values
    t_exp = data["exposure_time"].values
    shear_stress = data["shear_stress"].values
    sd_obs = data["fHb_std"].values

    # Get model predictions
    try:
        # umbridge models expect parameters as a list of lists
        result = model([theta.tolist()])
        y_pred = np.array(result[0])  # Extract predictions from umbridge format
    except Exception as e:
        print(f"Model evaluation failed: {e}")
        return -np.inf

    # Handle noise model
    if noise_model == "homo":
        sigma = np.sqrt(np.mean(sd_obs**2))
        sigma_vec = np.full_like(y_obs, sigma)
    else:
        sigma_vec = sd_obs

    # Check for invalid predictions or variances
    if np.any(~np.isfinite(y_pred)) or np.any(sigma_vec <= 0):
        return -np.inf

    # Calculate log-likelihood
    residuals = y_obs - y_pred
    log_lik = -0.5 * np.sum(
        np.log(2 * np.pi * sigma_vec**2) + (residuals**2) / (sigma_vec**2)
    )

    return log_lik


def log_posterior(theta, model, data, noise_model):
    """Log posterior = log prior + log likelihood"""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, model, data, noise_model)


def initialize_walkers(nwalkers, bounds):
    """
    Initialize walker positions using Latin Hypercube Sampling
    
    Args:
        nwalkers: Number of walkers
        bounds: List of [low, high] bounds for each parameter
    
    Returns:
        Initial positions array of shape (nwalkers, ndim)
    """
    ndim = len(bounds)
    
    # Use Latin Hypercube Sampling for better coverage
    sampler = qmc.LatinHypercube(d=ndim)
    unit_samples = sampler.random(n=nwalkers)
    
    # Transform to parameter bounds
    initial_positions = np.zeros((nwalkers, ndim))
    for i, (low, high) in enumerate(bounds):
        initial_positions[:, i] = unit_samples[:, i] * (high - low) + low
    
    return initial_positions


def compute_mcmc(log_posterior, model, data, noise_model, nwalkers=50, nburn=2500, nsteps=5000):
    """
    Run MCMC sampling
    
    Returns:
        trace: Flattened samples after burn-in
        sampler: emcee sampler object
        lnprob: Log probabilities after burn-in
        samples: Samples after burn-in (nwalkers, nsteps-nburn, ndim)
    """
    ndim = 3
    #TODO remove repeated bounds. DRY
    bounds = [[1e-10, 20.0], [0.5, 2.5], [1e-2, 1.0]]

    initial_positions = initialize_walkers(nwalkers, bounds)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, 
        args=(model, data, noise_model)
    )
    
    print("Running MCMC...")
    sampler.run_mcmc(initial_positions, nsteps, progress=True)

    # Extract results after burn-in
    trace = sampler.chain[:, nburn:, :].reshape(-1, ndim)
    lnprob = sampler.lnprobability[:, nburn:]
    samples = sampler.chain[:, nburn:, :]
    
    return trace, sampler, lnprob, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MCMC parameter estimation')
    parser.add_argument('--data', help='CSV file with data')
    parser.add_argument('--model_name', help='Name of the HTTP model')
    parser.add_argument('--noise_model', choices=['homo', 'hetero'], 
                       help='Noise model type')
    parser.add_argument('--port', type=int, default=4242, help='Port for the HTTP Model', )

    args = parser.parse_args()
    
    # Load data
    try:
        data = pd.read_csv(args.data)
        print(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Set up model connection
    try:
        model = umbridge.HTTPModel(f"http://localhost:{args.port}", args.model_name)
        print(f"Connected to model: {args.model_name}")
    except Exception as e:
        print(f"Error connecting to model: {e}")
        exit(1)

    # Run MCMC
    try:
        trace, sampler, lnprob, samples = compute_mcmc(
            log_posterior=log_posterior, 
            model=model, 
            data=data, 
            noise_model=args.noise_model
        )
        
        print(f"MCMC completed. Trace shape: {trace.shape}")
        
        # Save results
        np.savez(f"calibration_output_{args.model_name}_{args.data}.npz", 
                 trace=trace, samples=samples, lnprob=lnprob)
        print(f"Results saved to calibration_output_{args.model_name}_{args.data}.npz")
        
    except Exception as e:
        print(f"Error during MCMC: {e}")
        exit(1)