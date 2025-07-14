import numpy as np
from scipy.integrate import solve_ivp
import csv

def computeGeff_shear(t, G, f1=5.0, n=100):
    t0 = t[0]
    t1 = t[1]
    ti = np.linspace(t0, t1, n)
    Geff = G * (1 - np.exp(-f1 * ti))
    return ti, Geff

def dlIHdt_powerLaw(t, sigma_all, t_all, A, alpha, beta):
    # interpolate vector sigma to current time t
    sigmai = np.interp(t, t_all, sigma_all)
    return (A * sigmai**alpha) ** (1 / beta)

def computeIH_powerLaw(t_all, sigma_all, parameters):
    A = parameters[0]
    alpha = parameters[1]
    beta = parameters[2]

    # solve ODE
    dlIHdt = lambda t, y : dlIHdt_powerLaw(t, sigma_all, t_all, A, alpha, beta)
    sol = solve_ivp(dlIHdt, [t_all[0], t_all[-1]], [0], t_eval=t_all)
    IH = (sol.y[0][-1])**(beta)
    return IH

def computeIH_powerLaw_algebraic(t_exp, sigma_exp, parameters):
    A = parameters[0]
    alpha = parameters[1]
    beta = parameters[2]

    # Calculate IH using the algebraic formula
    IH = A * (sigma_exp ** alpha) * (t_exp ** beta)
    return IH


def IH_powerLaw_strainBased(x, p):
    """
    Model #1: Compute IH with power-law model based on strain-based morphology:
    - x = [t_exp, sigma] (control variables)
    - p = [A, alpha, beta] (model parameters)
    """

    t_exp = x[0] # exposure time
    sigma = x[1] # stress

    t_arr = np.array([0, t_exp])
    t_strain, sigma_eff = computeGeff_shear(t_arr, sigma)
    return computeIH_powerLaw(t_strain, sigma_eff, p) * 100  # convert to percentage

def IH_powerLaw_stressBased(x, p):
    """
    Model #2: Compute IH with power-law model based on stress-based morphology:
    - x = [t_exp, sigma] (control variables)
    - p = [A, alpha, beta] (model parameters)
    """

    t_exp = x[0]  # exposure time
    sigma = x[1]  # stress

    t_arr = np.array([0, t_exp])
    sigmai = np.ones_like(t_arr) * sigma  # constant shear stress for power-law
    return computeIH_powerLaw(t_arr, sigmai, p) * 100  # convert to percentage

def IH_powerLaw_algebraic(x, p):
    """
    Model #3: Compute IH with algebraic power-law model based on stress-based morphology:
    - x = [t_exp, sigma] (control variables)
    - p = [A, alpha, beta] (model parameters)
    """

    # Unpack values
    t_exp = x[0]  # exposure time
    sigma = x[1]  # stress

    return computeIH_powerLaw_algebraic(t_exp, sigma, p) * 100  # convert to percentage

def get_input_data(fname):
    """
    Load input data from a CSV file.
    The CSV file should contain columns for 'exposure_time' and 'shear_stress'.
    Returns a 2D numpy array with columns [exposure_time, shear_stress].
    """
    data = csv.reader(open(fname, 'r'))
    header = next(data)
    data = np.array(list(data), dtype=float)

    # get data by header
    t = data[:, header.index('exposure_time')]
    sigma_exp = data[:, header.index('shear_stress')]

    return np.array([t, sigma_exp]).T

def get_IH_model(model_name):
    """
    Returns the appropriate IH model function based on the model name.
    """
    if model_name == 'IH_powerLaw_strainBased':
        return IH_powerLaw_strainBased
    elif model_name == 'IH_powerLaw_stressBased':
        return IH_powerLaw_stressBased
    elif model_name == 'IH_powerLaw_algebraic':
        return IH_powerLaw_algebraic
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def evaluate_model(parameters, fname='data.csv', model_name='IH_powerLaw_strainBased'):
    """
    Computes IH using the specified model with given parameters.
    """

    IH_model = get_IH_model(model_name)
    input_data = get_input_data(fname)

    return [ IH_model(x, parameters) for x in input_data ]

def main():
    """Example usage of the IH power-law models."""

    # example parameters for the power-law model
    parameters = [1.228e-7, 1.9918, 0.6606]
    
    fname_data = 'data.csv'
    output_data_strainBased = evaluate_model(parameters, fname=fname_data, model_name='IH_powerLaw_strainBased')
    output_data_stressBased = evaluate_model(parameters, fname=fname_data, model_name='IH_powerLaw_stressBased')

    print("Output data (strain-based):", output_data_strainBased)
    print("Output data (stress-based):", output_data_stressBased)

    # IH_strainBased = [ IH_powerLaw_strainBased(x, parameters) for x in input_data ]
    # IH_stressBased = [ IH_powerLaw_stressBased(x, parameters) for x in input_data ]
    # IH_stressBased_algebraic = [ IH_powerLaw_algebraic(x, parameters) for x in input_data ]


if __name__ == "__main__":
    main()