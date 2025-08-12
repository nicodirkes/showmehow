#!/usr/bin/env nextflow

workflow {
    STAGE_DATA (
        file("$moduleDir/stage_data/.token.txt")
    )

    PREPROCESS_DATA(
        STAGE_DATA.out
    )

    UQ_STATUS (
        PREPROCESS_DATA.out.done
    ) 
    
    SERVE_IH_MODEL(
        UQ_STATUS.out.pipe,
        file("$moduleDir/model/IH_model_server.py"),
        PREPROCESS_DATA.out.data, 
        port_number = 4242
    )

    CALIBRATION ( 
        UQ_STATUS.out.pipe,
        file("$moduleDir/mcmc/MCMC_revised.py"),
        "IH_powerLaw_strainBased",
        PREPROCESS_DATA.out.data
        ) 
}

process STAGE_DATA {
    conda "$moduleDir/stage_data/environment.yml"
    publishDir "$moduleDir/outputs", mode: 'copy'
    
    input:
      path token

    output:
      path "staged_data"

    script:
        """
        #!/usr/bin/env python3

        import coscine
        with open("$token", "rt") as fp:
	        token = fp.read()
        # You can now use token to intialize the coscine ApiClient!

        client = coscine.ApiClient(token)
        resource = client.project("showmehow_usecase5").resource("Field Data")
        
        for file in resource.files():
            file.download(path="./staged_data/")

        """

}

process PREPROCESS_DATA{
    conda "$moduleDir/preprocessing/environment.yml"
    publishDir "$moduleDir/outputs", mode: 'copy'

    input:
        path raw_data

    output:
        path "*.csv", emit: data
        val true, emit: done

    script:
    """
    #!/usr/bin/env python3
    import os
    import pandas as pd
    # Load your data

    df = pd.read_csv(os.path.join("$raw_data", 'data_ding_human.csv'))
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
    wide_df.to_csv('data_ding_human_processed_linear.csv')

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
    path "status_info", emit: pipe
}


process SERVE_IH_MODEL {
    conda "$moduleDir/model/environment.yml"
    cache 'lenient'

    input:
    path status
    path script
    path control_variables
    val port_number

    script:
    """
    #!/bin/bash


    # Start the model server in the background
    python ${script}  & 
    PID=\$!
    echo \$PID
    # Wait for the model server to start
    while ! nc -z localhost ${port_number}; do
        sleep 1
    done
    echo "model_server_up" > $status
    echo "Model server is up and running on port ${port_number}"

    # Monitor the status 
    cat $status  &
    STATUS_PID=\$!
    wait \$STATUS_PID

    # Stop the model server when the signal is received
    kill \$PID
    
    """
}

process CALIBRATION {
    conda "$moduleDir/mcmc/environment.yml"
    cache 'lenient'
    publishDir "$moduleDir/outputs", mode: 'copy'
    
    input:
    path status
    path script
    val model_name
    path calibration_data


    script:
        """
        cat $status # blocked until the model server is up

        python ${script} ${calibration_data} ${model_name}
        
        echo "mcmc_done" > $status # signal to stop the model server
        """

    output:
    path "mcmc_samples.npy"
    path "mcmc_full_chain.npy"

}

// process DIAGNOSTICS {

// }
