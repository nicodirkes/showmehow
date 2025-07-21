# server.py
import umbridge
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
import os  

class IHPowerLawModel(umbridge.Model):
    def __init__(self, model_name="IH_powerLaw_strainBased", data_file="reshaped_human_data.csv"):
        super().__init__("forward")
        self.model_name = model_name
        self.data_file = data_file

        print(f"[SERVER INIT] Model: {model_name}, Data file: {data_file}")
        self.input_data = self.get_input_data(self.data_file)

    def get_input_sizes(self, config):
        return [3]  # [A, alpha, beta]

    def get_output_sizes(self, config):
        return [len(self.input_data)]

    def __call__(self, parameters, config):
        try:
            print(f"[CALL] Received parameters: {parameters}")
            IH_values = self.evaluate_model(parameters[0], self.data_file, self.model_name)
            # Convert any numpy types to native Python types
            IH_values = [float(value) for value in IH_values]
            print(f"[CALL] Returning IH values: {IH_values}")
            return [IH_values]
        except Exception as e:
            print(f"[SERVER ERROR] {e}")
            raise e

    def supports_evaluate(self):
        return True

    def computeGeff_shear(self, t, G, f1=5.0, n=100):
        t0, t1 = t[0], t[1]
        ti = np.linspace(t0, t1, n)
        Geff = G * (1 - np.exp(-f1 * ti))
        return ti, Geff

    def dlIHdt_powerLaw(self, t, sigma_all, t_all, A, alpha, beta):
        sigmai = np.interp(t, t_all, sigma_all)
        return (A * sigmai**alpha) ** (1 / beta)

    def computeIH_powerLaw(self, t_all, sigma_all, parameters):
        A, alpha, beta = parameters
        dlIHdt = lambda t, y: self.dlIHdt_powerLaw(t, sigma_all, t_all, A, alpha, beta)
        sol = solve_ivp(dlIHdt, [t_all[0], t_all[-1]], [0], t_eval=t_all)
        return (sol.y[0][-1]) ** beta

    def computeIH_powerLaw_algebraic(self, t_exp, sigma_exp, parameters):
        A, alpha, beta = parameters
        return A * (sigma_exp ** alpha) * (t_exp ** beta)

    def IH_powerLaw_strainBased(self, x, p):
        print(f"[DEBUG] Strain-Based Input: {x}, Parameters: {p}")
        t_exp, sigma = x
        t_arr = np.array([0, t_exp])
        t_strain, sigma_eff = self.computeGeff_shear(t_arr, sigma)
        result = self.computeIH_powerLaw(t_strain, sigma_eff, p) * 100
        print(f"[DEBUG] Strain-Based Result: {result}")
        return result

    def IH_powerLaw_stressBased(self, x, p):
        print(f"[DEBUG] Stress-Based Input: {x}, Parameters: {p}")
        t_exp, sigma = x
        t_arr = np.array([0, t_exp])
        sigmai = np.ones_like(t_arr) * sigma
        result = self.computeIH_powerLaw(t_arr, sigmai, p) * 100
        print(f"[DEBUG] Stress-Based Result: {result}")
        return result

    def IH_powerLaw_algebraic(self, x, p):
        print(f"[DEBUG] Algebraic Input: {x}, Parameters: {p}")
        t_exp, sigma = x
        result = self.computeIH_powerLaw_algebraic(t_exp, sigma, p) * 100
        print(f"[DEBUG] Algebraic Result: {result}")
        return result

    def get_input_data(self, fname):
        try:
            df = pd.read_csv(fname)
            df.columns = df.columns.str.strip().str.lower()
            if 'exposure_time' not in df.columns or 'shear_stress' not in df.columns:
                raise ValueError("CSV must contain 'exposure_time' and 'shear_stress' columns")
            return df[['exposure_time', 'shear_stress']].to_numpy()
        except Exception as e:
            print(f"[CSV LOAD ERROR] {e}")
            raise e

    def get_IH_model(self, model_name):
        if model_name == 'IH_powerLaw_strainBased':
            return self.IH_powerLaw_strainBased
        elif model_name == 'IH_powerLaw_stressBased':
            return self.IH_powerLaw_stressBased
        elif model_name == 'IH_powerLaw_algebraic':
            return self.IH_powerLaw_algebraic
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def evaluate_model(self, parameters, fname_controlVars='reshaped_human_data.csv', model_name='IH_powerLaw_strainBased'):
        IH_model = self.get_IH_model(model_name)
        input_data = self.get_input_data(fname_controlVars)
        return [IH_model(x, parameters) for x in input_data]

def main():
    data_file = os.getenv("DATA_FILE", "reshaped_human_data.csv")
    port = int(os.getenv("UMBRIDGE_PORT", 4242))

    models = [
        IHPowerLawModel("IH_powerLaw_strainBased", data_file),
        IHPowerLawModel("IH_powerLaw_stressBased", data_file),
        IHPowerLawModel("IH_powerLaw_algebraic", data_file)
    ]

    models[0].name = "IH_powerLaw_strainBased"
    models[1].name = "IH_powerLaw_stressBased"
    models[2].name = "IH_powerLaw_algebraic"

    print(f"[SERVER] Starting UMBridge server on port {port}")
    print(f"[SERVER] Using data file: {data_file}")
    print(f"[SERVER] Available models: {[model.name for model in models]}")

    umbridge.serve_models(models, port)

if __name__ == "__main__":
    main()
