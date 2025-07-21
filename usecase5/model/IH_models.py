import numpy as np
from scipy.integrate import quad
import csv

def computeAreaStrain(lamb):
    """
    Compute area strain from principal stretches (lamb).
    Args:
        lamb (array-like): [lambda1, lambda2, lambda3]
    Returns:
        eps (float): area strain
    """
    PI = np.pi

    a = np.sqrt(lamb[0])
    b = np.sqrt(lamb[1])
    c = np.sqrt(lamb[2])
    con = 1.0
    areaold = 0.0
    n = 2

    if np.isclose(a, b) and np.isclose(b, c):
        eps = 0.0
        con = 0.0
    else:
        while abs(con) > 1e-8 and n < 60:
            n += 2
            areatemp = 0.0
            for j in range(1, n // 2 + 1):
                tj = np.cos((2.0 * j - 1.0) * PI / (2.0 * n))
                tau = (1.0 - (c / b) ** 2) * tj ** 2
                temp1 = 1.0 - tau
                temp2 = (1.0 - (c / a) ** 2 - tau)
                temp3 = temp1 / np.sqrt(temp2)
                temp4 = np.sqrt(temp2 / temp1)
                areatemp += temp3 * np.arcsin(temp4)
            areaval = 2.0 * PI * b * c + 4.0 * PI * a * b / n * areatemp
            con = areaval - areaold
            areaold = areaval

        r0 = (a * b * c) ** (1.0 / 3.0) # Compute radius in relaxed state (sphere).
        
        # Compute area strain.
        eps = ((areaval / (4.0 * PI * r0 ** 2)) - 1.0) * 6.0 / 48.4

    return eps

def computePoreArea(eps):
    """
    Compute pore area from area strain using polynomial fit.
    Args:
        eps (float): area strain
    Returns:
        Ap (float): pore area
    """

    # Polynomial coefficients for the area strain to pore area conversion
    p1 = -1.716e4
    p2 = 5.816e2
    p3 = 1.301e2

    # Define the thresholds.
    eps1 = 0.16e-2
    eps2 = 6.0e-2
    Ap1 = 0.0
    Ap2 = 6.1932

    if eps < eps1:
        Ap = Ap1
    elif eps1 <= eps <= eps2:
        Ap = p1 * eps**3 + p2 * eps**2 + p3 * eps
    elif eps > eps2:
        Ap = Ap2

    return Ap

def computeLambda(t, G, f1, f2):
    Geff = G * (1 - np.exp(-f1 * t))
    f1f2 = f1**2 + f2**2 * Geff**2
    lamb2 = (f1**2 / f1f2) ** (1 / 3)
    lamb1 = lamb2 * (f1f2 + Geff*f2 * np.sqrt(f1f2)) / f1**2
    lamb3 = lamb2 * (f1f2 - Geff*f2 * np.sqrt(f1f2)) / f1**2

    return [lamb1, lamb2, lamb3]

def IH_poreFormation(t_exp, sigma, h, k, log=False, mu=0.0035, f1=5.0, f2=4.2298e-4, V_RBC=147.494):
    """
    Model #4: Compute IH with pore formation model based on strain-based morphology.
    """
    G = sigma / mu  # shear rate

    # Compute integral of pore area formation
    int_fun = lambda t: computePoreArea(computeAreaStrain(computeLambda(t, G, f1, f2)))
    Apt, _ = quad(int_fun, 0, t_exp)

    if log:
        # avoid log(0) by adding a small constant
        Apt = max(Apt, 1e-10)  # ensure Apt is not zero for log calculation
        return h - np.log(V_RBC) + k * np.log(G) + np.log(Apt) + np.log(100)
    else:
        return h * (G ** k) * Apt / V_RBC * 100

def IH_powerLaw_strainBased(t, sigma, A, alpha, beta, f1=5.0, log=False):

    sigma_eff = lambda t: sigma * (1.0 - np.exp(-f1 * t))
    sigma_int, _ = quad(lambda t: sigma_eff(t) ** (alpha / beta), 0, t)
    if log:
        return A + beta * np.log(sigma_int) + np.log(100)
    else:
        return A * sigma_int ** beta * 100

def IH_powerLaw_stressBased(t_exp, sigma_exp, A, alpha, beta, log=False):

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