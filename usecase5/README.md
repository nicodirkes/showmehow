# Usecase 5: Calibration 

This repository aims to recreate the workflow outlined in the following paper:
Blum, C., Steinseifer, U. and Neidlin, M. (2025), Toward Uncertainty-Aware Hemolysis Modeling: A Universal Approach to Address Experimental Variance. Int J Numer Meth Biomed Engng, 41: e70040. https://doi.org/10.1002/cnm.70040


This usecase shows you how to ..

## Running the Workflow
We use `nextflow` to run this usecase which is specified through the SHOWME.how file. 

1. Create and activate the conda environment
```
conda env create -f environment.yml
conda activate showmehow
```

2. Run the workflow

```
nextflow run SHOWME.how -params-file params.yml
```
