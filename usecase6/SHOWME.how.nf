#!/usr/bin/env nextflow
import groovy.yaml.YamlBuilder


workflow {
    def species = params.species
    def model = params.model.name
    def umbridge_port = params.umbridge_port



    PULL_DATA_COSCINE (
        token_file = file("$moduleDir/pull_data/.token.txt"),
        project_name = "showmehow_usecase5",
        resource_name = "Field Data"
    )

    PREPROCESS_DATA(
        script = file("$moduleDir/preprocessing/preprocessing.py"),
        prefix = "data_ding_",
        species = species,
        indir = PULL_DATA_COSCINE.out
    )

    UQ_STATUS (
        PREPROCESS_DATA.out.done
    ) 
    
    SERVE_MODEL(
        script = file("$moduleDir/model/IH_model_server.py"),
        src = file("$moduleDir/model/src"),
        name = model,
        data = PREPROCESS_DATA.out.data, 
        port = umbridge_port,
        UQ_STATUS.out.comm
    )

    // SERVE_MODEL_JL(
    //     script = file("$moduleDir/model_julia/IH_model_server.jl"),
    //     src = file("$moduleDir/model_julia/src"),
    //     name = "IH_powerLaw_stressBased",
    //     data = PREPROCESS_DATA.out.data, 
    //     port = umbridge_port,
    //     UQ_STATUS.out.comm
    // )


    CALIBRATION( 
        script = file("$moduleDir/mcmc/calibrate_emcee.py"),
        config = params,
        data = PREPROCESS_DATA.out.data,
        port = umbridge_port,
        UQ_STATUS.out.comm
    )

    // DIAGNOSTICS(
    //   script = file("$moduleDir/diagnostics/run_diagnostics.py"),
    //   mcmc_results = CALIBRATION.out.mcmc_results,
    //   outdir = "diagnostics_assesment",
    //   species = species,
    //   config = params,
    // )

}

process PULL_DATA_COSCINE {
    conda "$moduleDir/pull_data/environment.yml"
    publishDir "$moduleDir/outputs", mode: 'copy'
    
    input:
      path token_file
      val project_name
      val resource_name

    output:
      path "pulled_data"

    script:
        """
        #!/usr/bin/env python3
        import coscine

        # Read access token and setup client
        with open("$token_file", "rt") as fp:
	        token = fp.read()
        client = coscine.ApiClient(token)

        # Download the data
        resource = client.project("$project_name").resource("$resource_name")
        # resource = client.project("showmehow_usecase5").resource("Field Data")
        for file in resource.files():
            file.download(path="./pulled_data/")

        """

}

process PREPROCESS_DATA{
    conda "$moduleDir/preprocessing/environment.yml"
    publishDir "$moduleDir/outputs", mode: 'copy'

    input:
        file script
        val prefix
        val species
        path input

    output:
        path "${prefix}${species}_processed.csv", emit: data
        val true, emit: done

    script:
    """
    python3 preprocessing.py $species --prefix $prefix --indir $input  
    """
}

process UQ_STATUS {
    input:
    val ready

    script:
    """
    mkfifo status_info
    """

    output:
    path "status_info", emit: comm
}

process SERVE_MODEL {
    conda "$moduleDir/model/environment.yml"
    cache 'lenient'

    input:
    path script
    path src
    val name
    path data
    val port
    path status_comm

    script:
    """
    #!/bin/bash


    # Start the model server in the background
    python ${script} --name ${name} --data ${data} --port ${port} & 
    
    PID=\$!
    echo "Model Server PID: \$PID"
    # Wait for the model server to start
    while ! nc -z localhost ${port}; do
        sleep 1
    done
    echo "model_server_up" > ${status_comm}
    echo "Model server is up and running on port ${port}"

    # Monitor the status 
    cat ${status_comm}  &
    STATUS_PID=\$!
    wait \$STATUS_PID

    # Stop the model server when the signal is received
    kill \$PID

    rm ${status_comm}
    
    """
}

// process SERVE_MODEL_JL {
//     container "file://$moduleDir/model_julia/container.sif"
//     cache 'lenient'

//     input:
//     path script
//     path src
//     val name
//     path data
//     val port
//     path status_comm

//     script:
//     """
//     #!/bin/bash


//     # Start the model server in the background
//     julia ${script} --name ${name} --data ${data} --port ${port} & 
    
//     PID=\$!
//     echo "Model Server PID: \$PID"
//     # Wait for the model server to start
//     while ! curl http://localhost:${port}/Info -X GET ; do
//         sleep 1
//     done
//     echo "model_server_up" > ${status_comm}
//     echo "Model server is up and running on port ${port}"

//     # Monitor the status 
//     cat ${status_comm}  &
//     STATUS_PID=\$!
//     wait \$STATUS_PID

//     # Stop the model server when the signal is received
//     kill \$PID

//     rm ${status_comm}
    
//     """
// }

process CALIBRATION {
    conda "$moduleDir/mcmc/environment.yml"
    cache 'lenient'
    publishDir "$moduleDir/outputs", mode: 'copy'

    
    input:
    path script
    val config
    path data
    val port
    path status_comm


    script:
    def parameters = new YamlBuilder()
    parameters(config)
    """

    echo "${parameters.toString()}" > _params.yml
    

    cat ${status_comm} # blocked until the model server is up

    python ${script}  --config _params.yml --data ${data} --port ${port}
    
    echo "mcmc_done" > ${status_comm} # signal to stop the model server
    """

    output:
    path "*.npz", emit: mcmc_results

}

process DIAGNOSTICS {
    conda "$moduleDir/diagnostics/environment.yml"
    publishDir "$moduleDir/outputs", mode: 'copy'

    input:
    path script
    path mcmc_results
    val outdir
    val species
    val config


    script:
    def parameters = new YamlBuilder()
    parameters(config)
    """
    #!/bin/bash
    echo "${parameters.toString()}" > _params.yml
    python3 ${script} --mcmc_results ${mcmc_results} --outdir "${outdir}_${mcmc_results.simpleName}" --species ${species} --config _params.yml
    """

    output:
    path "${outdir}_${mcmc_results.simpleName}"


}
