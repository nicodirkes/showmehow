import umbridge
import numpy as np
import pandas as pd
import os
import argparse
from IH_models_revised import (
    IH_powerLaw_stressBased,
    IH_powerLaw_strainBased, 
    IH_poreFormation
)

class IHModel(umbridge.Model):
    # Model constants
    MU = 0.0035
    F1 = 5.0
    F2 = 4.2298e-4
    V_RBC = 147.494
    
    # Shared data across all models
    _shared_data = None
    
    def __init__(self, model_name):
        super().__init__("forward")
        self.name = model_name
        
        # Define model configurations
        self.model_configs = {
            'IH_powerLaw_stressBased': {
                'params': 3, 'function': IH_powerLaw_stressBased, 'log': False
            },
            'IH_powerLaw_stressBased_log': {
                'params': 3, 'function': IH_powerLaw_stressBased, 'log': True
            },
            'IH_powerLaw_strainBased': {
                'params': 3, 'function': IH_powerLaw_strainBased, 'log': False
            },
            'IH_powerLaw_strainBased_log': {
                'params': 3, 'function': IH_powerLaw_strainBased, 'log': True
            },
            'IH_poreFormation': {
                'params': 2, 'function': IH_poreFormation, 'log': False
            },
            'IH_poreFormation_log': {
                'params': 2, 'function': IH_poreFormation, 'log': True
            }
        }
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.config = self.model_configs[model_name]

    @classmethod
    def load_shared_data(cls, data_file):
        """Load data once and share across all model instances."""
        if cls._shared_data is None:
            df = pd.read_csv(data_file)
            df.columns = df.columns.str.strip().str.lower()
            
            if 'exposure_time' not in df.columns or 'shear_stress' not in df.columns:
                raise ValueError("CSV must contain 'exposure_time' and 'shear_stress' columns")
            
            cls._shared_data = df[['exposure_time', 'shear_stress']].to_numpy()
            print(f"Loaded data: {len(cls._shared_data)} rows")
        
        return cls._shared_data

    def get_input_sizes(self, config):
        return [self.config['params']]

    def get_output_sizes(self, config):
        return [len(self._shared_data)]

    def __call__(self, parameters, config):
        try:
            # Parameter validation
            if len(parameters[0]) != self.config['params']:
                raise ValueError(f"Model {self.name} expects {self.config['params']} parameters, got {len(parameters[0])}")
            
            IH_values = self.evaluate_model(parameters[0])
            return [[float(value) for value in IH_values]]
        except Exception as e:
            print(f"Error in {self.name}: {e}")
            raise e

    def supports_evaluate(self):
        return True

    def evaluate_model(self, parameters):
        """Evaluate the model with given parameters."""
        model_func = self.config['function']
        use_log = self.config['log']
        
        results = []
        for row in self._shared_data:
            t_exp, sigma = row
            
            if 'stressBased' in self.name:
                result = model_func(t_exp, sigma, *parameters, log=use_log)
            elif 'strainBased' in self.name:
                result = model_func(t_exp, sigma, *parameters, f1=self.F1, log=use_log)
            elif 'poreFormation' in self.name:
                result = model_func(t_exp, sigma, *parameters, log=use_log, 
                                  mu=self.MU, f1=self.F1, f2=self.F2, V_RBC=self.V_RBC)
            
            results.append(result)
        
        return results

def parse_arguments():
    parser = argparse.ArgumentParser(description='IH Model UMBridge Server')
    parser.add_argument('--data', type=str, 
                       default='data_ding_human_processed_linear.csv',
                       help='Data file path')
    parser.add_argument('--port', type=int, default=4242,
                       help='Server port')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Override with environment variables if present
    data_file = os.getenv("DATA_FILE", args.data)
    port = int(os.getenv("UMBRIDGE_PORT", args.port))
    
    print(f"Starting UMBridge server on port {port}")
    print(f"Data file: {data_file}")
    
    # Load shared data once
    IHModel.load_shared_data(data_file)
    
    # Create ALL available model instances
    all_models = [
        'IH_powerLaw_stressBased',
        'IH_powerLaw_stressBased_log', 
        'IH_powerLaw_strainBased',
        'IH_powerLaw_strainBased_log',
        'IH_poreFormation',
        'IH_poreFormation_log'
    ]
    
    models = [IHModel(model_name) for model_name in all_models]
    
    print(f"Available models: {[model.name for model in models]}")
    
    umbridge.serve_models(models, port)

if __name__ == "__main__":
    main()