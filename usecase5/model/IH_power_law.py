import numpy as np
from scipy.integrate import quad
import csv
from IH_pore import IH_poreFormation

def computeIH_powerLaw_strainBased(t, sigma, parameters, f1 = 5.0, log=False):
    A = parameters[0]
    alpha = parameters[1]
    beta = parameters[2]

    # solve ODE
    sigma_eff = lambda t: sigma * (1.0 - np.exp(-f1 * t))
    sigma_int, _ = quad(lambda t: sigma_eff(t) ** (alpha / beta), 0, t)
    if log:
        return A + beta * np.log(sigma_int) + np.log(100)
    else:
        return A * sigma_int ** beta * 100

def computeIH_powerLaw_stressBased(t_exp, sigma_exp, parameters, log=False):
    A = parameters[0]
    alpha = parameters[1]
    beta = parameters[2]

    if log:
        return A + alpha * np.log(sigma_exp) + beta * np.log(t_exp) + np.log(100)
    else:
        return A * (sigma_exp ** alpha) * (t_exp ** beta) * 100  # convert to percentage


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

def get_IH_model(model_name, mu, f1, f2, log=False):
    """
    Returns the appropriate IH model function based on the model name.
    """
    if model_name == 'IH_powerLaw_strainBased':
        return lambda x, p: computeIH_powerLaw_strainBased(x[0], x[1], p, f1, log)
    elif model_name == 'IH_powerLaw_stressBased':
        return lambda x, p: computeIH_powerLaw_stressBased(x[0], x[1], p, log)
    elif model_name == 'IH_poreFormation':
        return lambda x, p: IH_poreFormation(x, p, log, mu=mu, f1=f1, f2=f2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def evaluate_model(parameters, fname_controlVars='data.csv', model_name='IH_powerLaw_strainBased',
                   mu=0.0035, f1=5.0, f2=4.2298e-4, log=False):
    """
    Computes IH using the specified model with given parameters.
    Inputs:
    - parameters: list of model parameters [A, alpha, beta]
    - fname_controlVars: path to the CSV file containing control variables
    - model_name: name of the model to use ('IH_powerLaw_strainBased', 'IH_powerLaw_stressBased', 'IH_powerLaw_algebraic')
    - mu: viscosity (default 0.0035), only used for pore formation model
    - f1, f2: parameters for pore formation model (default 5.0, 4.2298e-4), only used for pore formation model
    Returns:
    - log (bool): whether to use logarithmic scaling, leading to log(IH) output
    - A list of computed IH values for each control variable point.
    """

    IH_model = get_IH_model(model_name, mu, f1, f2, log)
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
    output_data_pore_log = evaluate_model([np.log(parameters_pore[0]), parameters_pore[1]], fname_controlVars=fname_data, model_name='IH_poreFormation', mu=0.00424, f1=5.0, f2=4.2298e-4, log=True)


    print("Output data (strain-based):", output_data_strainBased)
    print("Output data (stress-based):", output_data_stressBased)
    print("Output data (pore formation):", output_data_pore)
    print("Output data (pore formation, log):", np.exp(output_data_pore_log))


if __name__ == "__main__":
    main()