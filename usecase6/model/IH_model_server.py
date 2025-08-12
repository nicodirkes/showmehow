import umbridge
import numpy as np
import pandas as pd
import os
import argparse
from IH_model import IH_powerLaw_stressBased

class IHModel(umbridge.Model):
    
    _shared_data = None  # Class variable to store shared data
    
    def __init__(self):
        super().__init__("forward")
        self.name = "IH_powerLaw_stressBased"
        self.params = 3

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
        return [self.params]

    def get_output_sizes(self, config):
        return [len(self._shared_data)]

    def __call__(self, parameters, config):
        try:
            # Parameter validation
            if len(parameters[0]) != self.params:
                raise ValueError(f"Model expects {self.params} parameters, got {len(parameters[0])}")
            
            IH_values = self.evaluate_model(parameters[0])
            return [[float(value) for value in IH_values]]
        except Exception as e:
            print(f"Error in model: {e}")
            raise e

    def supports_evaluate(self):
        return True

    def evaluate_model(self, parameters):
        """Evaluate the model with given parameters."""
        results = []
        for row in self._shared_data:
            t_exp, sigma = row
            result = IH_powerLaw_stressBased(t_exp, sigma, *parameters)
            results.append(result)
        
        return results

def parse_arguments():
    parser = argparse.ArgumentParser(description='IH Model UMBridge Server')
    parser.add_argument('--data', type=str, 
                       default='data.csv',
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
    print(f"Model: IH_powerLaw_stressBased")
    
    # Load shared data once
    IHModel.load_shared_data(data_file)
    
    # Create the model instance
    model = IHModel()
    print(f"Successfully created model: {model.name}")
    
    # Serve the model
    umbridge.serve_models([model], port)

if __name__ == "__main__":
    main()