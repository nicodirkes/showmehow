import numpy as np 
import emcee
import umbridge
import argparse
import pandas as pd
import logging
from scipy.stats import qmc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCMCConfig:
    """Centralized configuration for MCMC sampling"""
    nwalkers: int = 50
    nburn: int = 2500
    nsteps: int = 5000
    convergence_threshold: float = 1.1  # R-hat threshold
    min_eff_samples: int = 100  # Minimum effective sample size
    
    def validate_nwalkers(self, model_name: str):
        """Validate number of walkers after model is known"""
        model_config = get_model_config(model_name)
        ndim = len(model_config["params"])
        if self.nwalkers < 2 * ndim:
            logger.warning(f"nwalkers ({self.nwalkers}) should be >= 2 * ndim ({ndim}) for emcee")

# Model configurations - centralized parameter definitions
MODEL_CONFIGS = {
    "IH_powerLaw_stressBased": {
        "params": ["A", "alpha", "beta"],
        "bounds": [[0.0, 1e-3], [0.0, 3.0], [0.0, 1.0]]
    },
    "IH_powerLaw_strainBased": {
        "params": ["A", "alpha", "beta"], 
        "bounds": [[0.0, 1e-3], [0.0, 3.0], [0.0, 1.0]]
    },
    "IH_powerLaw_algebraic": {
        "params": ["A", "alpha", "beta"],
        "bounds": [[0.0, 1e-3], [0.0, 3.0], [0.0, 1.0]]
    }
}

def get_model_config(model_name: str) -> Dict:
    """Get configuration for a specific model"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name]

def validate_data(data: pd.DataFrame) -> None:
    """Validate that data has required columns and reasonable values"""
    required_columns = ["Mean", "SD"]
    
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if data["Mean"].isna().any():
        raise ValueError("NaN values found in Mean column")
    
    if data["SD"].isna().any():
        raise ValueError("NaN values found in SD column")
    
    if (data["SD"] <= 0).any():
        raise ValueError("Non-positive values found in SD column")
    
    logger.info(f"Data validation passed: {len(data)} observations")

def log_prior_theta(theta: np.ndarray, bounds: np.ndarray) -> float:
    """
    Uniform log prior for model parameters
    
    Args:
        theta: Parameter vector
        bounds: Array of [lower, upper] bounds for each parameter
    
    Returns:
        Log prior probability
    """
    # Check if parameters are within bounds
    if not np.all(bounds[:, 0] <= theta) or not np.all(theta <= bounds[:, 1]):
        return -np.inf
    
    # Calculate log of uniform prior: -log(volume)
    # Volume = product of (upper - lower) for each parameter
    volume = np.prod(bounds[:, 1] - bounds[:, 0])
    return -np.log(volume)

def safe_model_evaluation(model, theta: np.ndarray) -> Optional[np.ndarray]:
    """
    Safely evaluate model with error handling
    
    Args:
        model: UM-Bridge model
        theta: Parameter vector
    
    Returns:
        Model output or None if evaluation failed
    """
    try:
        model_output = np.asarray(model([theta.tolist()])[0], dtype='float64')
        
        if not np.all(np.isfinite(model_output)):
            return None
            
        return model_output
        
    except Exception as e:
        logger.debug(f"Model evaluation failed: {e}")
        return None

def log_likelihood_homoscedastic(theta: np.ndarray, data: pd.DataFrame, model) -> float:
    """
    Homoscedastic Gaussian likelihood
    
    Args:
        theta: Parameter vector
        data: DataFrame with Mean and SD columns
        model: UM-Bridge model
    
    Returns:
        Log likelihood
    """
    mean_obs = data["Mean"].to_numpy()
    std_obs = data["SD"].to_numpy()
    
    model_output = safe_model_evaluation(model, theta)
    if model_output is None:
        return -np.inf
    
    variance = std_obs ** 2
    
    # Calculate log likelihood
    log_lik = -0.5 * np.sum(
        np.log(2 * np.pi * variance) + 
        (mean_obs - model_output) ** 2 / variance
    )
    
    return log_lik

def log_posterior(theta: np.ndarray, data: pd.DataFrame, model, bounds: np.ndarray) -> float:
    """Combined log posterior = log prior + log likelihood"""
    return log_prior_theta(theta, bounds) + log_likelihood_homoscedastic(theta, data, model)

def initialize_walkers(nwalkers: int, bounds: np.ndarray) -> np.ndarray:
    """
    Initialize walker positions using Latin Hypercube Sampling
    
    Args:
        nwalkers: Number of walkers
        bounds: Parameter bounds
    
    Returns:
        Initial positions array
    """
    ndim = len(bounds)
    
    # Use Latin Hypercube Sampling for better coverage
    sampler = qmc.LatinHypercube(d=ndim)
    unit_samples = sampler.random(n=nwalkers)
    
    # Transform to parameter bounds
    initial_positions = np.zeros((nwalkers, ndim))
    for i in range(ndim):
        low, high = bounds[i]
        initial_positions[:, i] = unit_samples[:, i] * (high - low) + low
    
    return initial_positions

def run_burnin(initial_positions: np.ndarray, config: MCMCConfig, 
               data: pd.DataFrame, model, bounds: np.ndarray) -> np.ndarray:
    """
    Run burn-in phase
    
    Returns:
        Final walker positions after burn-in
    """
    nwalkers, ndim = initial_positions.shape
    
    logger.info(f"Starting burn-in: {config.nburn} steps with {nwalkers} walkers")
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, 
        args=(data, model, bounds)
    )
    
    sampler.run_mcmc(initial_positions, config.nburn, progress=True)
    
    logger.info("Burn-in completed")
    return sampler.get_last_sample().coords

def run_production(burnin_positions: np.ndarray, config: MCMCConfig,
                  data: pd.DataFrame, model, bounds: np.ndarray) -> Tuple[np.ndarray, emcee.EnsembleSampler]:
    """
    Run production phase
    
    Returns:
        (samples, sampler) tuple
    """
    nwalkers, ndim = burnin_positions.shape
    
    logger.info(f"Starting production: {config.nsteps} steps")
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior,
        args=(data, model, bounds)
    )
    
    sampler.run_mcmc(burnin_positions, config.nsteps, progress=True)
    
    # Reshape samples
    samples = sampler.get_chain(flat=True)
    
    logger.info(f"Production completed: {len(samples)} total samples")
    return samples, sampler

def check_convergence(sampler: emcee.EnsembleSampler, config: MCMCConfig) -> bool:
    """
    Basic convergence diagnostics
    
    Returns:
        True if converged according to basic criteria
    """
    try:
        # R-hat (Gelman-Rubin statistic)
        rhat = sampler.get_autocorr_time(quiet=True)
        max_rhat = np.max(rhat)
        
        # Effective sample size
        eff_samples = sampler.get_chain(flat=True).shape[0] / (2 * max_rhat)
        
        logger.info(f"Max R-hat estimate: {max_rhat:.3f}")
        logger.info(f"Effective sample size estimate: {eff_samples:.0f}")
        
        converged = (max_rhat < config.convergence_threshold * config.nsteps and 
                    eff_samples > config.min_eff_samples)
        
        if not converged:
            logger.warning("Convergence criteria not met!")
            if max_rhat >= config.convergence_threshold:
                logger.warning(f"R-hat too high: {max_rhat:.3f} >= {config.convergence_threshold}")
            if eff_samples <= config.min_eff_samples:
                logger.warning(f"Effective sample size too low: {eff_samples:.0f} <= {config.min_eff_samples}")
        
        return converged
        
    except Exception as e:
        logger.warning(f"Could not compute convergence diagnostics: {e}")
        return False

def run_mcmc_pipeline(data: pd.DataFrame, model, model_name: str, config: MCMCConfig) -> Tuple[np.ndarray, emcee.EnsembleSampler]:
    """
    Full MCMC pipeline: initialization -> burn-in -> production -> diagnostics
    
    Returns:
        (samples, sampler) tuple
    """
    # Get model configuration
    model_config = get_model_config(model_name)
    bounds = np.array(model_config["bounds"])
    
    logger.info(f"Running MCMC for model: {model_name}")
    logger.info(f"Parameters: {model_config['params']}")
    logger.info(f"Bounds: {bounds.tolist()}")
    
    # Initialize walkers
    initial_positions = initialize_walkers(config.nwalkers, bounds)
    
    # Run burn-in
    burnin_positions = run_burnin(initial_positions, config, data, model, bounds)
    
    # Run production
    samples, sampler = run_production(burnin_positions, config, data, model, bounds)
    
    # Check convergence
    converged = check_convergence(sampler, config)
    
    if converged:
        logger.info("MCMC completed successfully with good convergence")
    else:
        logger.warning("MCMC completed but convergence is questionable")
    
    return samples, sampler

def main():
    parser = argparse.ArgumentParser(description="MCMC parameter inference")
    parser.add_argument('data_filename', help='Path to CSV data file')
    parser.add_argument('model_name', help='Model name', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--nwalkers', type=int, default=20, help='Number of walkers')
    parser.add_argument('--nburn', type=int, default=500, help='Burn-in steps')
    parser.add_argument('--nsteps', type=int, default=1000, help='Production steps')
    parser.add_argument('--server', default='http://localhost:4242', help='UM-Bridge server URL')
    
    args = parser.parse_args()
    
    # Create configuration
    config = MCMCConfig(
        nwalkers=args.nwalkers,
        nburn=args.nburn,
        nsteps=args.nsteps
    )
    
    # Validate walker count for this model
    config.validate_nwalkers(args.model_name)
    
    # Load and validate data
    logger.info(f"Loading data from: {args.data_filename}")
    data = pd.read_csv(args.data_filename)
    validate_data(data)
    
    # Connect to model
    logger.info(f"Connecting to model: {args.model_name} at {args.server}")
    model = umbridge.HTTPModel(args.server, args.model_name)
    
    # Run MCMC
    samples, sampler = run_mcmc_pipeline(data, model, args.model_name, config)
    
    # Save results
    logger.info("Saving results...")
    np.save("mcmc_samples.npy", samples)
    np.save("mcmc_full_chain.npy", sampler.get_chain())
    
    logger.info(f"Results saved. Final sample shape: {samples.shape}")

if __name__ == "__main__":
    main()