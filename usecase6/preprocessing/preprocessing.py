import pandas as pd

# Load your data
df = pd.read_csv('data.csv') 
# Group and assign a count to each row within each group
df['Output_number'] = df.groupby(['shear_stress', 'exposure_time']).cumcount() + 1
# Pivot the table to get outputs in columns
wide_df = df.pivot(index=['shear_stress', 'exposure_time'], columns='Output_number', values='fHb')
# Rename columns to Output1, Output2, Output3, etc.
wide_df.columns = [f'Output{i}' for i in wide_df.columns]
# Reset index to make alpha and beta regular columns
wide_df = wide_df.reset_index()
# Add mean and standard deviation
wide_df['Mean'] = wide_df[['Output1', 'Output2', 'Output3']].mean(axis=1)
wide_df['SD'] = wide_df[['Output1', 'Output2', 'Output3']].std(axis=1)
import numpy as np
df_mod = pd.read_csv("reshaped_data.csv")