using UMBridge
using CSV
using DataFrames
using ArgParse

include("src/IH_model.jl")

function get_IH_model(
    name::String,
    data::DataFrame
    )

    if name=="IH_powerLaw_stressBased"
        model_func = IH_powerLaw_stressBased
        parameters = 3
    elseif name=="IH_powerLaw_strainBased"
        model_func = IH_powerLaw_strainBased_quad
        parameters = 3
    elseif name=="IH_poreFormation_stressBased"
        model_func = IH_poreFormation_stressBased
        parameters = 2
    elseif name=="IH_poreFormation_strainBased"
        model_func = IH_poreFormation_strainBased_quad
        parameters = 2
    end

    # setup umbridge model
    model = UMBridge.Model(
            name = name,
            inputSizes = [parameters],
            outputSizes = [1 for _ in 1:nrow(data)],
            supportsEvaluate = true
        )
    
    function evaluate_model(parameters; data::DataFrame=data)
            t_exp_all = data[:, :exposure_time]
            sigma_all = data[:, :shear_stress]
            return model_func.(t_exp_all, sigma_all, parameters...)
        end


    # evaluation function
    function evaluate(inputs, config)
        IH_values = evaluate_model(inputs[1])
        return [[value] for value in IH_values]
    end

    UMBridge.define_evaluate(model, evaluate)

    return model
end






function parse_arguments()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--name"
            help = "Model name"
            arg_type = String
        "--data"
            help = "Data file path"
            arg_type = String
        "--port"
            help = "Server port"
            arg_type = Int
            default = 4242
    end
    return parse_args(s)
end

function main()
    args = parse_arguments()

    name = args["name"]
    data_file = args["data"]
    port = get(args, "port", 4242)

    println("Starting UMBridge server on port $port")
    println("Data file: $data_file")
    println("Model: $name")

    data = CSV.read(data_file, DataFrame)

    # Create the model instance
    model = get_IH_model(name, data)

    println("Successfully created model: $(model.name)")

    # Serve the model
    UMBridge.serve_models([model], port)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end