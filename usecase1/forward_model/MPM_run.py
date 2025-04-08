from psimpy.simulator import MassPointModel
from psimpy.simulator import RunSimulator
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("elevation_filename")
parser.add_argument("lhs_output_filename")
parser.add_argument("forward_model_output_filename")

args = parser.parse_args()      
print(args)
MPM_M1 = MassPointModel()
var_inp_parameter = ['coulomb_friction','turbulent_friction']

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
elevation_path = os.path.join(dir_path, args.elevation_filename)



# initial location
x0 = 200
y0 = 2000
dt = 0.1




run_mass_point_model = RunSimulator(simulator=MPM_M1.run,var_inp_parameter=var_inp_parameter,fix_inp={'elevation' : elevation_path,'x0' : x0, 'y0' : y0, 'dt':dt})

train_samples = np.loadtxt(os.path.join(dir_path, "..", "lhs", args.lhs_output_filename), delimiter=',', skiprows=1)
var_samples = train_samples



run_mass_point_model.serial_run(var_samples=var_samples)
serial_output = run_mass_point_model.outputs
nsamples = 100


U_res = np.ones(nsamples)
x_res = np.ones(nsamples)
for i in range(nsamples):
        U_res[i] = serial_output[i][:,5].max()
        x_res[i] = serial_output[i][:,1].max()


np.savetxt(os.path.join(dir_path, args.forward_model_output_filename), U_res, delimiter=",")