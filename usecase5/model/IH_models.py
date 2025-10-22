import numpy as np
from scipy.integrate import quad
import csv

def computeEffShearStrainBased(t, G, f1):
    return G * (1 - np.exp(-f1 * t))

def computePoreAreaInterpolated(G):
    if G < 3740:
        return 0.0
    elif G > 42000:
        return 6.1932
    else:
        # Coefficients of interpolation polynomial
        p = [ 4.06157696e-02, -2.83266089e-05,  2.25830588e-08, -8.49450091e-13, 1.32415867e-17, -8.23845340e-23]
        return p[0] + p[1]*G + p[2]*G**2 + p[3]*G**3 + p[4]*G**4 + p[5]*G**5

def IH_poreFormation(t_exp, sigma_exp, h, k, log=False, mu=0.0035, f1=5.0, V_RBC=147.494):
    """
    Model #3: Compute IH with pore formation model based on strain-based morphology.
    Sensible limits:
    0 <= h <= 20
    0 <= k <= 2
    """
    G = sigma_exp / mu  # shear rate

    # Compute integral of pore area formation
    Ap = lambda t: computePoreAreaInterpolated(computeEffShearStrainBased(t, G, f1))
    Apt, _ = quad(Ap, 0, t_exp)

    if log:
        Apt = max(Apt, 1e-10)  # ensure Apt is not zero for log calculation
        return -h - np.log(V_RBC) + k * np.log(G) + np.log(Apt) + np.log(100)
    else:
        return np.exp(-h) * (G ** k) * Apt / V_RBC * 100


def IH_powerLaw_strainBased(t_exp, sigma_exp, A, alpha, beta, f1=5.0, log=False):
    """
        Model #2: Strain-based power law.
        Sensible limits for the parameters:
        0 <= A <= 20
        0.5 <= alpha <= 3
        0.01 <= beta <= 1
    """
    Geff_normalized = lambda t: computeEffShearStrainBased(t, 1, f1)
    G_int, _ = quad(lambda t: Geff_normalized(t) ** (alpha / beta), 0, t_exp)
    if log:
        return -A + alpha * np.log(sigma_exp) + beta * np.log(G_int) + np.log(100)
    else:
        return np.exp(-A) * sigma_exp**alpha * G_int ** beta * 100

def IH_powerLaw_stressBased(t_exp, sigma_exp, A, alpha, beta, log=False):
    """
        Model #1: Stress-based power law.
        Sensible limits for the parameters:
        0 <= A <= 20
        0.5 <= alpha <= 2.5
        0.01 <= beta <= 1
    """
    if log:
        return -A + alpha * np.log(sigma_exp) + beta * np.log(t_exp) + np.log(100)
    else:
        return np.exp(-A) * (sigma_exp ** alpha) * (t_exp ** beta) * 100  # convert to percentage


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
    t_exp = data[:, header.index('exposure_time')]
    sigma_exp = data[:, header.index('shear_stress')]

    return np.array([t_exp, sigma_exp]).T

def get_IH_model(model_name, mu, f1, f2, V_RBC, log=False):
    """
    Returns the appropriate IH model function based on the model name.
    """
    if model_name == 'IH_powerLaw_stressBased':
        def computeIH_powerLaw_stressBased(x, p):
            assert len(p) == 3, "Parameters should contain exactly three values: [A, alpha, beta]"
            assert len(x) == 2, "Input x should contain exactly rows: [t_exp, sigma]"
            return IH_powerLaw_stressBased(*x, *p, log)
        return computeIH_powerLaw_stressBased
    elif model_name == 'IH_powerLaw_strainBased':
        def computeIH_powerLaw_strainBased(x, p):
            assert len(p) == 3, "Parameters should contain exactly three values: [A, alpha, beta]"
            assert len(x) == 2, "Input x should contain exactly rows: [t_exp, sigma]"
            return IH_powerLaw_strainBased(*x, *p, f1, log)
        return computeIH_powerLaw_strainBased
    elif model_name == 'IH_poreFormation':
        def computeIH_poreFormation(x, p):
            assert len(p) == 2, "Parameters should contain exactly two values: [h, k]"
            assert len(x) == 2, "Input x should contain exactly rows: [t_exp, sigma]"
            return IH_poreFormation(*x, *p, log, mu=mu, f1=f1, f2=f2, V_RBC=V_RBC)
        return computeIH_poreFormation
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def evaluate_model(parameters, fname_controlVars='data.csv', model_name='IH_powerLaw_strainBased', log=False,
                   mu=0.0035, f1=5.0, f2=4.2298e-4, V_RBC=147.494):
    """
    Computes IH using the specified model with given parameters.
    Inputs:
    - parameters: list of model parameters [A, alpha, beta]
    - fname_controlVars: path to the CSV file containing control variables
    - model_name: name of the model to use ('IH_powerLaw_strainBased', 'IH_powerLaw_stressBased', 'IH_poreFormation')
    - mu: viscosity (default 0.0035), only used for pore formation model
    - f1: parameter for pore formation model and strain-based power law (default 5.0)
    - f2: parameter only for pore formation model (default 4.2298e-4)
    Returns:
    - log (bool): whether to use logarithmic scaling, leading to log(IH) output
    - A list of computed IH values for each control variable point.
    """

    IH_model = get_IH_model(model_name, mu, f1, f2, V_RBC, log)
    input_data = get_input_data(fname_controlVars)

    return [ IH_model(x, parameters) for x in input_data ]

def main():
    """Example usage of the IH power-law models."""

    # example parameters for the power-law model
    parameters = [1.228e-7, 1.9918, 0.6606]

    # example parameters for the pore formation model
    parameters_pore = [1e-4, 0.8]
    
    fname_data = 'data.csv'
    output_data_stressBased = evaluate_model(parameters, fname_controlVars=fname_data, model_name='IH_powerLaw_stressBased')
    output_data_strainBased = evaluate_model(parameters, fname_controlVars=fname_data, model_name='IH_powerLaw_strainBased', f1=5.0)
    output_data_pore = evaluate_model(parameters_pore, fname_controlVars=fname_data, model_name='IH_poreFormation', mu=0.00424, f1=5.0, f2=4.2298e-4, V_RBC=147.494)
    output_data_pore_log = evaluate_model([np.log(parameters_pore[0]), parameters_pore[1]], fname_controlVars=fname_data, model_name='IH_poreFormation',log=True, mu=0.00424, f1=5.0, f2=4.2298e-4, V_RBC=147.494)

    print("Output data (stress-based):", output_data_stressBased)
    print("Output data (strain-based):", output_data_strainBased)
    print("Output data (pore formation):", output_data_pore)
    print("Output data (pore formation, log):", np.exp(output_data_pore_log))


if __name__ == "__main__":
    main()