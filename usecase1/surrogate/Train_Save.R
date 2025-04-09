library("RobustGaSP")


dir_path <- box::file()
# Read command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Read input data
Input <- read.csv(file.path(dir_path, "..", "lhs", args[1]), header = TRUE)
Input_data <- as.matrix(Input)

# Read output data
Output <- read.csv(file.path(dir_path, "..", "forward_model", args[2]), header = FALSE)
Output_data <- as.matrix(Output)

# Fit the Gaussian process model
ScalarGP_Umax <- rgasp(design = Input_data, response = Output_data, lower_bound = TRUE)


LOO <- leave_one_out_rgasp(ScalarGP_Umax)
print(LOO)

# Save the model using saveRDS instead of save
saveRDS(ScalarGP_Umax, file = file.path(dir_path, args[3]))