# Usecase 1: Bayesian Parameter Calibration

This usecase shows you how to calibrate the parameters of a Mass Point Model (MPM) using Monte-Carlo Markov Chain method.

## Prerequisites
In order to use containers (apptainer) for certain computational units, we first need to build them. This can be done by running the following command

```
cd  <computational_unit>
apptainer build <computational_unit>.sif <coomputational_unit.def>
cd ..
```

An example for the `lhs` computational unit is shown below. 
``
cd lhs
apptainer build lhs.sif lhs.def
cd ..
```

## Running the Workflow

The usecase can be run by executing the following command:

```
nextflow run SHOWME.how -params-file params.yaml
```
