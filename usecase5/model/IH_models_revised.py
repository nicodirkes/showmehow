import numpy as np
from scipy.integrate import solve_ivp, quad

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
    """Compute principal stretches."""
    Geff = G * (1 - np.exp(-f1 * t))
    f1f2 = f1**2 + f2**2 * Geff**2
    lamb2 = (f1**2 / f1f2) ** (1 / 3)
    lamb1 = lamb2 * (f1f2 + Geff*f2 * np.sqrt(f1f2)) / f1**2
    lamb3 = lamb2 * (f1f2 - Geff*f2 * np.sqrt(f1f2)) / f1**2

    return [lamb1, lamb2, lamb3]

def IH_poreFormation(t_exp, sigma, h, k, log=False, mu=0.0035, f1=5.0, f2=4.2298e-4, V_RBC=147.494):
    """
    Compute IH with pore formation model based on strain-based morphology.
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
    """
    Strain-based power law model for IH.
    """
    sigma_eff = lambda t: sigma * (1.0 - np.exp(-f1 * t))
    sigma_int, _ = quad(lambda t: sigma_eff(t) ** (alpha / beta), 0, t)
    
    if log:
        return A + beta * np.log(sigma_int) + np.log(100)
    else:
        return A * sigma_int ** beta * 100

def IH_powerLaw_stressBased(t_exp, sigma_exp, A, alpha, beta, log=False):
    """
    Stress-based power law model for IH.
    """
    if log:
        return A + alpha * np.log(sigma_exp) + beta * np.log(t_exp) + np.log(100)
    else:
        return A * (sigma_exp ** alpha) * (t_exp ** beta) * 100  # convert to percentage  