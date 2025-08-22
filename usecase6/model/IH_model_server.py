import umbridge
import numpy as np
import pandas as pd
import os
import argparse
from src.IH_model import IH_powerLaw_stressBased

class IH_Model(umbridge.Model):
    
    _data = None  # Class variable to store shared data
    
    def __init__(self, name):
        self.name = name
        self.params = 3
        if name == "IH_powerLaw_stressBased":
            self.model_func = IH_powerLaw_stressBased

    @classmethod
    def load_data(cls, data_file):
        """Load data once and share across all model instances."""
        if cls._data is None:
            required_columns = ['exposure_time', 'shear_stress']
            df = pd.read_csv(
                data_file,
                usecols = required_columns
                )

            cls._data = df[required_columns].to_numpy()
            print(f"Loaded {data_file} with {len(cls._data)} rows")
        return cls._data

    def get_input_sizes(self, config):
        return [self.params]

    def get_output_sizes(self, config):
        return [len(self._data)]

    def __call__(self, parameters, config):
        try:
            # Parameter validation
            if len(parameters[0]) != self.params:
                raise ValueError(f"Model expects {self.params} parameters, got {len(parameters[0])}")
            
            IH_values = self.evaluate_model(parameters[0])
            return [IH_values.tolist()]
        except Exception as e:
            print(f"Error in model: {e}")
            raise e

    def supports_evaluate(self):
        return True

    def evaluate_model(self, parameters):
        """Evaluate the model with given parameters."""
        A, alpha, beta = parameters

        t_exp_all = self._data[:, 0]
        sigma_all = self._data[:, 1]
        result_all = self.model_func(t_exp_all, sigma_all, A, alpha, beta)
        return result_all

def parse_arguments():
    parser = argparse.ArgumentParser(description='IH Model UMBridge Server')
    parser.add_argument('--name', type=str, 
                       help='Model name')
    parser.add_argument('--data', type=str, 
                       help='Data file path')
    parser.add_argument('--port', type=int, default=4242,
                       help='Server port')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Override with environment variables if present
    name = args.name
    data_file = args.data
    port = args.port
    
    print(f"Starting UMBridge server on port {port}")
    print(f"Data file: {data_file}")
    print(f"Model: {name}")
    
    # Load shared data once
    IH_Model.load_data(data_file)
    
    # Create the model instance
    model = IH_Model(name)
    print(f"Successfully created model: {model.name}")
    
    # Serve the model
    umbridge.serve_models([model], port)

if __name__ == "__main__":
    main()