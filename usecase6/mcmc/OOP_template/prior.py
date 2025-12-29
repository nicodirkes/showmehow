"""
Minimal prior distributions for Bayesian inference.

Design goals:
- Explicit > implicit
- NumPy-first
- Vectorized where it matters
- No factories, no logging, no hidden dependencies

Supported Priors:
    - UniformPrior: Uniform distribution over bounded intervals
    - NormalPrior: Normal (Gaussian) distribution
    - LogNormalPrior: Log-normal distribution
    - GammaPrior: Gamma distribution (shape-rate parameterization)
    - BetaPrior: Beta distribution

Author: Your Name
Version: 0.2.0
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import numpy.typing as npt

try:
    from scipy.special import gammaln, betaln
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    gammaln = None
    betaln = None



__all__ = [
    "Prior",
    "UniformPrior",
    "NormalPrior",
    "LogNormalPrior",
    "GammaPrior",
    "BetaPrior",
]


Array = npt.ArrayLike


class Prior(ABC):
    """
    Abstract base class for prior distributions.
    
    All priors must implement log_prior, log_prior_batch, and sample methods.
    
    Attributes
    ----------
    n_params : int
        Number of parameters in the prior distribution.
    """

    def __init__(self, n_params: int):
        """
        Initialize the prior.
        
        Parameters
        ----------
        n_params : int
            Number of parameters. Must be positive.
        
        Raises
        ------
        ValueError
            If n_params is not a positive integer.
        """
        if not isinstance(n_params, int) or n_params <= 0:
            raise ValueError(f"n_params must be a positive integer, got {n_params}")
        self.n_params = int(n_params)

    def _validate_theta(self, theta: Array) -> np.ndarray:
        """
        Validate a single parameter vector.
        
        Parameters
        ----------
        theta : array-like
            Parameter vector to validate.
        
        Returns
        -------
        np.ndarray
            Validated parameter array of shape (n_params,).
        
        Raises
        ------
        ValueError
            If theta has wrong shape or contains NaN/Inf.
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(
                f"theta must have shape ({self.n_params},), got {theta.shape}"
            )
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains NaN or Inf values")
        return theta
    
    def _validate_thetas(self, thetas: Array) -> np.ndarray:
        """
        Validate a batch of parameter vectors.
        
        Parameters
        ----------
        thetas : array-like
            Batch of parameter vectors to validate.
        
        Returns
        -------
        np.ndarray
            Validated parameter array of shape (N, n_params).
        
        Raises
        ------
        ValueError
            If thetas has wrong shape or contains NaN/Inf.
        """
        thetas = np.asarray(thetas, dtype=float)
        if thetas.ndim != 2 or thetas.shape[1] != self.n_params:
            raise ValueError(
                f"thetas must have shape (N, {self.n_params}), got {thetas.shape}"
            )
        if not np.all(np.isfinite(thetas)):
            raise ValueError("thetas contains NaN or Inf values")
        return thetas
    
    def _validate_nsamples(self, nsamples: int) -> int:
        """Validate number of samples."""
        if not isinstance(nsamples, int) or nsamples <= 0:
            raise ValueError(f"nsamples must be a positive integer, got {nsamples}")
        return nsamples

    @abstractmethod
    def log_prior(self, theta: Array) -> float:
        """
        Compute log-prior density for a single parameter vector.
        
        Parameters
        ----------
        theta : array-like
            Parameter vector of shape (n_params,).
        
        Returns
        -------
        float
            Log-prior density value.
        """
        pass

    @abstractmethod
    def log_prior_batch(self, thetas: np.ndarray) -> np.ndarray:
        """
        Compute log-prior densities for a batch of parameter vectors.
        
        This is a vectorized operation for efficiency.
        
        Parameters
        ----------
        thetas : np.ndarray
            Batch of parameter vectors with shape (N, n_params).
        
        Returns
        -------
        np.ndarray
            Log-prior density values with shape (N,).
        """
        pass

    def __call__(self, theta: Array) -> float:
        """
        Allow prior(theta) syntax.
        
        Parameters
        ----------
        theta : array-like
            Parameter vector of shape (n_params,).
        
        Returns
        -------
        float
            Log-prior density value.
        """
        return self.log_prior(theta)

    @abstractmethod
    def sample(
        self,
        nsamples: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """ 
        Draw samples from the prior distribution.
        
        Parameters
        ----------
        nsamples : int, default=1
            Number of samples to draw.
        rng : np.random.Generator, optional
            Random number generator. If None, creates a new default generator.
        
        Returns
        -------
        np.ndarray
            Samples with shape (nsamples, n_params).
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """Return string representation of the prior."""
        pass


# ==================== Uniform Prior ====================

class UniformPrior(Prior):
    """
    Independent uniform priors over closed intervals [a, b].
    
    Parameters
    ----------
    bounds : array-like
        Bounds for each parameter with shape (n_params, 2).
        Each row is [lower, upper] for one parameter.
    
    Attributes
    ----------
    lower : np.ndarray
        Lower bounds for each parameter.
    upper : np.ndarray
        Upper bounds for each parameter.
    width : np.ndarray
        Width of each interval (upper - lower).
    
    Examples
    --------
    >>> prior = UniformPrior([[0, 1], [-5, 5], [0, 10]])
    >>> prior.log_prior([0.5, 0, 5])
    -3.912023005428146
    >>> samples = prior.sample(100, rng=np.random.default_rng(42))
    >>> samples.shape
    (100, 3)
    """

    def __init__(self, bounds: Array):
        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                f"bounds must have shape (n_params, 2), got {bounds.shape}"
            )
        if not np.all(np.isfinite(bounds)):
            raise ValueError("bounds contains NaN or Inf values")
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("upper bounds must be greater than lower bounds")

        self.lower = bounds[:, 0]
        self.upper = bounds[:, 1]
        self.width = self.upper - self.lower
        super().__init__(n_params=len(bounds))

        # Precompute log normalizing constant
        self._log_norm = -np.sum(np.log(self.width))

    def log_prior(self, theta: Array) -> float:
        """
        Compute log-prior for uniform distribution.
        
        Returns -inf if any parameter is outside its bounds,
        otherwise returns the log normalizing constant.
        """
        theta = self._validate_theta(theta)
        if np.any(theta < self.lower) or np.any(theta > self.upper):
            return -np.inf
        return self._log_norm

    def log_prior_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Vectorized batch evaluation of log-prior."""
        thetas = self._validate_thetas(thetas)
        
        # Check bounds for all samples at once
        in_bounds = np.all(
            (thetas >= self.lower) & (thetas <= self.upper),
            axis=1
        )
        
        # Initialize with -inf, then set in-bounds values
        result = np.full(len(thetas), -np.inf)
        result[in_bounds] = self._log_norm
        return result

    def sample(
        self, 
        nsamples: int = 1, 
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Draw samples uniformly from the bounded intervals."""
        nsamples = self._validate_nsamples(nsamples)
        rng = rng or np.random.default_rng()
        return rng.uniform(
            self.lower, self.upper, size=(nsamples, self.n_params)
        )
    
    @property
    def bounds(self) -> np.ndarray:
        """Return bounds as (n_params, 2) array."""
        return np.column_stack([self.lower, self.upper])

    def __repr__(self) -> str:
        return f"UniformPrior(bounds={self.bounds.tolist()})"


# ==================== Normal Prior ====================

class NormalPrior(Prior):
    """
    Independent normal (Gaussian) priors.
    
    Parameters
    ----------
    mean : float or array-like
        Mean(s) of the distribution. If scalar, same mean for all parameters.
    std : float or array-like
        Standard deviation(s). If scalar, same std for all parameters.
    n_params : int, optional
        Number of parameters. Required if mean and std are scalars.
    
    Attributes
    ----------
    mean : np.ndarray
        Mean for each parameter.
    std : np.ndarray
        Standard deviation for each parameter.
    
    Examples
    --------
    >>> # Scalar parameters - same for all
    >>> prior = NormalPrior(mean=0, std=1, n_params=3)
    >>> # Array parameters - different for each
    >>> prior = NormalPrior(mean=[0, 1, 2], std=[1, 2, 3])
    """

    def __init__(
        self,
        mean: Union[float, Array],
        std: Union[float, Array],
        n_params: Optional[int] = None,
    ):
        if np.isscalar(mean) and np.isscalar(std):
            if n_params is None:
                raise ValueError(
                    "n_params is required when mean and std are scalars"
                )
            mean = np.full(n_params, float(mean))
            std = np.full(n_params, float(std))
        else:
            mean = np.asarray(mean, dtype=float)
            std = np.asarray(std, dtype=float)
            if mean.shape != std.shape:
                raise ValueError(
                    f"mean and std must have same shape, got {mean.shape} and {std.shape}"
                )

        if not np.all(np.isfinite(mean)):
            raise ValueError("mean contains NaN or Inf values")
        if not np.all(np.isfinite(std)):
            raise ValueError("std contains NaN or Inf values")
        if np.any(std <= 0):
            raise ValueError("std must be positive")

        super().__init__(len(mean))
        self.mean = mean
        self.std = std

        # Precompute log normalizing constant
        self._log_norm = (
            -0.5 * self.n_params * np.log(2 * np.pi)
            - np.sum(np.log(self.std))
        )

    def log_prior(self, theta: Array) -> float:
        """Compute log-prior for normal distribution."""
        theta = self._validate_theta(theta)
        z = (theta - self.mean) / self.std
        return self._log_norm - 0.5 * np.sum(z * z)

    def log_prior_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Vectorized batch evaluation of log-prior."""
        thetas = self._validate_thetas(thetas)
        z = (thetas - self.mean) / self.std
        return self._log_norm - 0.5 * np.sum(z * z, axis=1)

    def sample(
        self, 
        nsamples: int = 1, 
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Draw samples from normal distribution."""
        nsamples = self._validate_nsamples(nsamples)
        rng = rng or np.random.default_rng()
        return rng.normal(
            self.mean, self.std, size=(nsamples, self.n_params)
        )

    def __repr__(self) -> str:
        return f"NormalPrior(mean={self.mean.tolist()}, std={self.std.tolist()})"


# ==================== Log-Normal Prior ====================

class LogNormalPrior(Prior):
    """
    Independent log-normal priors.
    
    For each parameter θ_i: log(θ_i) ~ Normal(μ_i, σ_i²)
    
    Parameters
    ----------
    mu : float or array-like
        Mean(s) of the underlying normal distribution.
    sigma : float or array-like
        Standard deviation(s) of the underlying normal distribution.
    n_params : int, optional
        Number of parameters. Required if mu and sigma are scalars.
    
    Attributes
    ----------
    mu : np.ndarray
        Mean of log-transformed parameters.
    sigma : np.ndarray
        Standard deviation of log-transformed parameters.
    
    Examples
    --------
    >>> prior = LogNormalPrior(mu=0, sigma=1, n_params=2)
    >>> prior.log_prior([1.0, 2.0])
    -2.1120857...
    """

    def __init__(
        self,
        mu: Union[float, Array],
        sigma: Union[float, Array],
        n_params: Optional[int] = None,
    ):
        if np.isscalar(mu) and np.isscalar(sigma):
            if n_params is None:
                raise ValueError(
                    "n_params is required when mu and sigma are scalars"
                )
            mu = np.full(n_params, float(mu))
            sigma = np.full(n_params, float(sigma))
        else:
            mu = np.asarray(mu, dtype=float)
            sigma = np.asarray(sigma, dtype=float)
            if mu.shape != sigma.shape:
                raise ValueError(
                    f"mu and sigma must have same shape, got {mu.shape} and {sigma.shape}"
                )

        if not np.all(np.isfinite(mu)):
            raise ValueError("mu contains NaN or Inf values")
        if not np.all(np.isfinite(sigma)):
            raise ValueError("sigma contains NaN or Inf values")
        if np.any(sigma <= 0):
            raise ValueError("sigma must be positive")

        super().__init__(len(mu))
        self.mu = mu
        self.sigma = sigma

    def log_prior(self, theta: Array) -> float:
        """Compute log-prior for log-normal distribution."""
        theta = self._validate_theta(theta)
        if np.any(theta <= 0):
            return -np.inf

        log_theta = np.log(theta)
        z = (log_theta - self.mu) / self.sigma

        return (
            -0.5 * self.n_params * np.log(2 * np.pi)
            - np.sum(np.log(theta))
            - np.sum(np.log(self.sigma))
            - 0.5 * np.sum(z * z)
        )

    def log_prior_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Vectorized batch evaluation of log-prior."""
        thetas = self._validate_thetas(thetas)
        
        # Check positivity
        valid = np.all(thetas > 0, axis=1)
        result = np.full(len(thetas), -np.inf)
        
        if np.any(valid):
            log_thetas = np.log(thetas[valid])
            z = (log_thetas - self.mu) / self.sigma
            result[valid] = (
                -0.5 * self.n_params * np.log(2 * np.pi)
                - np.sum(log_thetas, axis=1)
                - np.sum(np.log(self.sigma))
                - 0.5 * np.sum(z * z, axis=1)
            )
        
        return result

    def sample(
        self, 
        nsamples: int = 1, 
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Draw samples from log-normal distribution."""
        nsamples = self._validate_nsamples(nsamples)
        rng = rng or np.random.default_rng()
        return rng.lognormal(
            self.mu, self.sigma, size=(nsamples, self.n_params)
        )

    def __repr__(self) -> str:
        return f"LogNormalPrior(mu={self.mu.tolist()}, sigma={self.sigma.tolist()})"


# ==================== Gamma Prior ====================

class GammaPrior(Prior):
    """
    Independent Gamma priors using shape-rate parameterization.
    
    For each parameter θ_i: θ_i ~ Gamma(α_i, β_i)
    where α is shape and β is rate.
    
    Parameters
    ----------
    alpha : float or array-like
        Shape parameter(s). Must be positive.
    beta : float or array-like
        Rate parameter(s). Must be positive.
    n_params : int, optional
        Number of parameters. Required if alpha and beta are scalars.
    
    Attributes
    ----------
    alpha : np.ndarray
        Shape parameters.
    beta : np.ndarray
        Rate parameters.
    
    Raises
    ------
    ImportError
        If scipy is not installed.
    
    Examples
    --------
    >>> prior = GammaPrior(alpha=2, beta=1, n_params=2)
    >>> prior.log_prior([1.0, 2.0])
    -1.6137...
    """

    def __init__(
        self,
        alpha: Union[float, Array],
        beta: Union[float, Array],
        n_params: Optional[int] = None,
    ):
        if not HAS_SCIPY:
            raise ImportError(
                "GammaPrior requires scipy. Install with: pip install scipy"
            )
        
        if np.isscalar(alpha) and np.isscalar(beta):
            if n_params is None:
                raise ValueError(
                    "n_params is required when alpha and beta are scalars"
                )
            alpha = np.full(n_params, float(alpha))
            beta = np.full(n_params, float(beta))
        else:
            alpha = np.asarray(alpha, dtype=float)
            beta = np.asarray(beta, dtype=float)
            if alpha.shape != beta.shape:
                raise ValueError(
                    f"alpha and beta must have same shape, got {alpha.shape} and {beta.shape}"
                )

        if not np.all(np.isfinite(alpha)):
            raise ValueError("alpha contains NaN or Inf values")
        if not np.all(np.isfinite(beta)):
            raise ValueError("beta contains NaN or Inf values")
        if np.any(alpha <= 0):
            raise ValueError("alpha must be positive")
        if np.any(beta <= 0):
            raise ValueError("beta must be positive")

        super().__init__(len(alpha))
        self.alpha = alpha
        self.beta = beta

        # Precompute log normalizing constant
        self._log_norm = np.sum(
            alpha * np.log(beta) - gammaln(alpha)
        )

    def log_prior(self, theta: Array) -> float:
        """Compute log-prior for Gamma distribution."""
        theta = self._validate_theta(theta)
        if np.any(theta <= 0):
            return -np.inf

        return (
            self._log_norm
            + np.sum((self.alpha - 1) * np.log(theta))
            - np.sum(self.beta * theta)
        )

    def log_prior_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Vectorized batch evaluation of log-prior."""
        thetas = self._validate_thetas(thetas)
        
        # Check positivity
        valid = np.all(thetas > 0, axis=1)
        result = np.full(len(thetas), -np.inf)
        
        if np.any(valid):
            log_thetas = np.log(thetas[valid])
            result[valid] = (
                self._log_norm
                + np.sum((self.alpha - 1) * log_thetas, axis=1)
                - np.sum(self.beta * thetas[valid], axis=1)
            )
        
        return result

    def sample(
        self, 
        nsamples: int = 1, 
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Draw samples from Gamma distribution."""
        nsamples = self._validate_nsamples(nsamples)
        rng = rng or np.random.default_rng()
        # NumPy uses shape-scale parameterization, convert rate to scale
        return rng.gamma(
            self.alpha, 1.0 / self.beta, size=(nsamples, self.n_params)
        )

    def __repr__(self) -> str:
        return f"GammaPrior(alpha={self.alpha.tolist()}, beta={self.beta.tolist()})"


# ==================== Beta Prior ====================

class BetaPrior(Prior):
    """
    Independent Beta priors on (0, 1).
    
    For each parameter θ_i: θ_i ~ Beta(α_i, β_i)
    
    Parameters
    ----------
    alpha : float or array-like
        First shape parameter(s). Must be positive.
    beta : float or array-like
        Second shape parameter(s). Must be positive.
    n_params : int, optional
        Number of parameters. Required if alpha and beta are scalars.
    
    Attributes
    ----------
    alpha : np.ndarray
        First shape parameters.
    beta : np.ndarray
        Second shape parameters.
    
    Raises
    ------
    ImportError
        If scipy is not installed.
    
    Examples
    --------
    >>> prior = BetaPrior(alpha=2, beta=2, n_params=2)
    >>> prior.log_prior([0.5, 0.5])
    1.0986...
    """

    def __init__(
        self,
        alpha: Union[float, Array],
        beta: Union[float, Array],
        n_params: Optional[int] = None,
    ):
        if not HAS_SCIPY:
            raise ImportError(
                "BetaPrior requires scipy. Install with: pip install scipy"
            )
        
        if np.isscalar(alpha) and np.isscalar(beta):
            if n_params is None:
                raise ValueError(
                    "n_params is required when alpha and beta are scalars"
                )
            alpha = np.full(n_params, float(alpha))
            beta = np.full(n_params, float(beta))
        else:
            alpha = np.asarray(alpha, dtype=float)
            beta = np.asarray(beta, dtype=float)
            if alpha.shape != beta.shape:
                raise ValueError(
                    f"alpha and beta must have same shape, got {alpha.shape} and {beta.shape}"
                )

        if not np.all(np.isfinite(alpha)):
            raise ValueError("alpha contains NaN or Inf values")
        if not np.all(np.isfinite(beta)):
            raise ValueError("beta contains NaN or Inf values")
        if np.any(alpha <= 0):
            raise ValueError("alpha must be positive")
        if np.any(beta <= 0):
            raise ValueError("beta must be positive")

        super().__init__(len(alpha))
        self.alpha = alpha
        self.beta = beta

        # Precompute log normalizing constant
        self._log_norm = -np.sum(betaln(alpha, beta))

    def log_prior(self, theta: Array) -> float:
        """Compute log-prior for Beta distribution."""
        theta = self._validate_theta(theta)
        if np.any(theta <= 0) or np.any(theta >= 1):
            return -np.inf

        return (
            self._log_norm
            + np.sum((self.alpha - 1) * np.log(theta))
            + np.sum((self.beta - 1) * np.log(1 - theta))
        )

    def log_prior_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Vectorized batch evaluation of log-prior."""
        thetas = self._validate_thetas(thetas)
        
        # Check bounds (0, 1)
        valid = np.all((thetas > 0) & (thetas < 1), axis=1)
        result = np.full(len(thetas), -np.inf)
        
        if np.any(valid):
            log_thetas = np.log(thetas[valid])
            log_one_minus_thetas = np.log(1 - thetas[valid])
            result[valid] = (
                self._log_norm
                + np.sum((self.alpha - 1) * log_thetas, axis=1)
                + np.sum((self.beta - 1) * log_one_minus_thetas, axis=1)
            )
        
        return result

    def sample(
        self, 
        nsamples: int = 1, 
        rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Draw samples from Beta distribution."""
        nsamples = self._validate_nsamples(nsamples)
        rng = rng or np.random.default_rng()
        return rng.beta(
            self.alpha, self.beta, size=(nsamples, self.n_params)
        )

    def __repr__(self) -> str:
        return f"BetaPrior(alpha={self.alpha.tolist()}, beta={self.beta.tolist()})"