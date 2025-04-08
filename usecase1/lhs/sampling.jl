using Pkg
working_directory = dirname(@__FILE__)
using QuasiMonteCarlo, Distributions
using CSV
using DataFrames
using JSON3
using StructTypes


struct LHS_Input
    dimensions::Int
    bounds::Dict{String, Vector{Float64}}
    number_of_samples::Int
end


StructTypes.StructType(::Type{LHS_Input}) = StructTypes.Struct()

json_string = read(joinpath(working_directory, ARGS[1]), String)
input = JSON3.read(json_string, LHS_Input)


d = input.dimensions
lb = [input.bounds["parameter1"][1], input.bounds["parameter2"][1]]
ub = [input.bounds["parameter1"][2], input.bounds["parameter2"][2]]
n = input.number_of_samples  # Number of samples


# # Generate LHS samples
s = QuasiMonteCarlo.sample(n, lb, ub, LatinHypercubeSample())

# # Convert to DataFrame (transpose and convert to regular matrix)
df = DataFrame(Matrix(s'), :auto)  # Specify :auto for automatic column names


# # Save the DataFrame to a CSV file without headers
CSV.write(joinpath(working_directory, ARGS[2]), df)
