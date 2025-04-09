library('coda')
library('reticulate')

np <- import("numpy")

samples <- np$load("mcmc_samples.npy")

# Convert samples to R matrix
samples <- as.matrix(samples)

# Convert to coda object
mcmc_chain <- mcmc(samples)

# Summary of the MCMC chain
summary(mcmc_chain)

plot(mcmc_chain)

chained_samples <- np$load("/content/chained_mcmc_samples.npy")

dim(chained_samples)

chained_mcmc_chain <- mcmc(chained_samples)

# Reshape the array
chain_length <- 5000
num_chains <- 50
num_parameters <- 2

# Reshape the array into a list of chains
mcmc_chain_list <- vector(mode = "list", length = num_parameters)
for (param_index in 1:num_parameters) {
  mcmc_samples_param <- chained_samples[, , param_index]  # Extract samples for a parameter
  mcmc_chain_list[[param_index]] <- lapply(1:num_chains, function(i) as.mcmc(mcmc_samples_param[, i]))
}

gelman_rubin_results <- lapply(mcmc_chain_list, gelman.diag)