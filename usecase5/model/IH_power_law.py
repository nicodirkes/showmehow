import numpy as np
from scipy.integrate import solve_ivp
import csv
from IH_pore import IH_poreFormation

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

def get_IH_model(model_name, mu, f1, f2):
    """
    Returns the appropriate IH model function based on the model name.
    """
    if model_name == 'IH_powerLaw_strainBased':
        return IH_powerLaw_strainBased
    elif model_name == 'IH_powerLaw_stressBased':
        return IH_powerLaw_stressBased
    elif model_name == 'IH_powerLaw_algebraic':
        return IH_powerLaw_algebraic
    elif model_name == 'IH_poreFormation':
        return lambda x, p: IH_poreFormation(x, p, mu=mu, f1=f1, f2=f2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def evaluate_model(parameters, fname_controlVars='data.csv', model_name='IH_powerLaw_strainBased',
                   mu=0.0035, f1=5.0, f2=4.2298e-4):
    """
    Computes IH using the specified model with given parameters.
    Inputs:
    - parameters: list of model parameters [A, alpha, beta]
    - fname_controlVars: path to the CSV file containing control variables
    - model_name: name of the model to use ('IH_powerLaw_strainBased', 'IH_powerLaw_stressBased', 'IH_powerLaw_algebraic')
    - mu: viscosity (default 0.0035), only used for pore formation model
    - f1, f2: parameters for pore formation model (default 5.0, 4.2298e-4), only used for pore formation model
    Returns:
    - A list of computed IH values for each control variable point.
    """

    IH_model = get_IH_model(model_name, mu, f1, f2)
    input_data = get_input_data(fname_controlVars)

    return [ IH_model(x, parameters) for x in input_data ]

def main():
    """Example usage of the IH power-law models."""

    # example parameters for the power-law model
    parameters = [1.228e-7, 1.9918, 0.6606]

    # example parameters for the pore formation model
    parameters_pore = [1e-4, 0.8]
    
    fname_data = 'data.csv'
    output_data_strainBased = evaluate_model(parameters, fname_controlVars=fname_data, model_name='IH_powerLaw_strainBased')
    output_data_stressBased = evaluate_model(parameters, fname_controlVars=fname_data, model_name='IH_powerLaw_stressBased')
    output_data_pore = evaluate_model(parameters_pore, fname_controlVars=fname_data, model_name='IH_poreFormation', mu=0.00424, f1=5.0, f2=4.2298e-4)


    print("Output data (strain-based):", output_data_strainBased)
    print("Output data (stress-based):", output_data_stressBased)
    print("Output data (pore formation):", output_data_pore)


if __name__ == "__main__":
    main()