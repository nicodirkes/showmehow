#!/usr/bin/env nextflow
import groovy.yaml.YamlBuilder


workflow bpc_hemolysis {

    take:
    config_file
    use_julia
    output_base_dir

    main:
    // Single source of truth: all per-experiment metadata derived once from the config
    workflow_meta = config_file.map { f ->
        def cfg = new groovy.yaml.YamlSlurper().parseText(f.text) as Map
        def key = cfg.containsKey('experiment_hash') ? cfg.experiment_hash : workflow.sessionId
        def bundle_name = "${cfg.species}_${cfg.model.name}_${key}".toString()
        def params_yaml = new YamlBuilder().tap { it(cfg) }.toString()
        [key, cfg, bundle_name, params_yaml]
    }
    // workflow_meta: [key, cfg, bundle_name, params_yaml]

    PULL_DATA_COSCINE(
        token_file    = file("$moduleDir/pull_data/.token.txt"),
        project_name  = "showmehow_usecase5",
        resource_name = "Field Data"
    )

    PREPROCESS_DATA(
        script  = file("$moduleDir/preprocessing/preprocessing.py"),
        prefix  = "data_ding_",
        species = workflow_meta.map { it[1].species },
        indir   = PULL_DATA_COSCINE.out,
        key     = workflow_meta.map { it[0] }
    )

    SETUP_UM_IPC(
        PREPROCESS_DATA.out.done
    )

    // Join workflow_meta with [key, data] and [key, comm] — guaranteed per-experiment pairing
    model_config = workflow_meta
        .join(PREPROCESS_DATA.out.data)
        .join(SETUP_UM_IPC.out)
    // model_config: [key, cfg, bundle_name, params_yaml, data, comm]

    if (use_julia) {
        SERVE_MODEL_JL(
            script        = file("$moduleDir/model_julia/IH_model_server.jl"),
            src           = file("$moduleDir/model_julia/src"),
            name          = model_config.map { it[1].model.name },
            data          = model_config.map { it[4] },
            umbridge_port = model_config.map { Math.abs(new Random().nextInt() % (31767 - 16384)) + 16384 },
            um_highway    = model_config.map { it[5] }
        )
    }
    else {
        SERVE_MODEL(
            script        = file("$moduleDir/model/IH_model_server.py"),
            src           = file("$moduleDir/model/src"),
            name          = model_config.map { it[1].model.name },
            data          = model_config.map { it[4] },
            umbridge_port = model_config.map { Math.abs(new Random().nextInt() % (31767 - 16384)) + 16384 },
            um_highway    = model_config.map { it[5] }
        )
    }

    MCMC_CALIBRATION(
        script      = file("$moduleDir/mcmc/calibrate_emcee.py"),
        key         = model_config.map { it[0] },
        params_yaml = model_config.map { it[3] },
        data        = model_config.map { it[4] },
        um_highway  = model_config.map { it[5] }
    )

    RUN_DIAGNOSTICS(
        script     = file("$moduleDir/diagnostics/run_diagnostics.py"),
        mcmc_idata = MCMC_CALIBRATION.out.mcmc_idata,
        outdir     = "diagnostics",
    )

    // Join workflow_meta with all keyed outputs — guaranteed per-experiment pairing into BUNDLE_OUTPUTS
    bundle_inputs = workflow_meta
        .join(MCMC_CALIBRATION.out.mcmc_output)
        .join(MCMC_CALIBRATION.out.mcmc_corner_plot)
        .join(MCMC_CALIBRATION.out.mcmc_trace)
        .join(MCMC_CALIBRATION.out.mcmc_idata)
        .join(RUN_DIAGNOSTICS.out)
    // bundle_inputs: [key, cfg, bundle_name, params_yaml, mcmc_output, mcmc_corner_plot, mcmc_trace, mcmc_idata, diagnostics]

    BUNDLE_OUTPUTS(
        output_base_dir  = output_base_dir,
        bundle_name      = bundle_inputs.map { it[2] },
        params_yaml      = bundle_inputs.map { it[3] },
        mcmc_output      = bundle_inputs.map { it[4] },
        mcmc_corner_plot = bundle_inputs.map { it[5] },
        mcmc_trace       = bundle_inputs.map { it[6] },
        mcmc_idata       = bundle_inputs.map { it[7] },
        mcmc_diagnostics = bundle_inputs.map { it[8] }
    )

    GENERATE_REPORT(
        script          = file("$moduleDir/report/generate_report.py"),
        output_base_dir = output_base_dir,
        bundle_dir      = BUNDLE_OUTPUTS.out.bundle_dir
    )

    emit:
    bundle = BUNDLE_OUTPUTS.out.bundle_dir
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

    input:
        file script
        val prefix
        val species
        path input
        val key

    output:
        tuple val(key), path("${prefix}${species}_processed.csv"), emit: data
        tuple val(key), val(true), emit: done

    script:
    """
    python3 preprocessing.py $species --prefix $prefix --indir $input
    """
}

process SETUP_UM_IPC {
    input:
    tuple val(key), val(ready)

    output:
    tuple val(key), path("comm")

    script:
    """
    mkdir comm
    mkdir comm/model_info
    mkdir comm/uq_info
    mkdir comm/umbridge_port
    """
}

process SERVE_MODEL {
    conda "$moduleDir/model/environment.yml"
    cache 'lenient'
    errorStrategy 'retry'
    maxRetries 3

    input:
    path script
    path src
    val name
    path data
    val umbridge_port
    path um_highway

    script:
    """
    #!/bin/bash
    UMBRIDGE_PORT=${umbridge_port+100*(task.attempt-1)}

    # Abort if port already used (success means something is listening)
    if bash -c "echo > /dev/tcp/localhost/\$UMBRIDGE_PORT" 2>/dev/null; then
        echo "Port \$UMBRIDGE_PORT already in use"
        exit 1
    fi

    # Start model server
    python ${script} --name ${name} --data ${data} --port \$UMBRIDGE_PORT &
    SERVER_PID=\$!
    trap 'kill \$SERVER_PID 2>/dev/null || true' EXIT INT TERM
    echo "Model Server PID: \$SERVER_PID (trying port \$UMBRIDGE_PORT)"

    # Wait for model server to start 
    while ! bash -c "echo > /dev/tcp/localhost/\$UMBRIDGE_PORT" 2>/dev/null; do
        sleep 1
    done

    touch $um_highway/model_info/READY 
    echo "Model ${name} ready" > $um_highway/model_info/READY

    touch $um_highway/umbridge_port/\$UMBRIDGE_PORT
    echo "Model server is running on port \$UMBRIDGE_PORT"

    # Monitor the status 
    echo "Waiting for UQ to complete..."
    until [ -e $um_highway/uq_info/DONE ]; do
        sleep 1
    done

    # Stop the model server when the signal is received
    kill \$SERVER_PID 2>/dev/null || true
    echo "Model server on port \$UMBRIDGE_PORT stopped."

    """
}

process SERVE_MODEL_JL {
    container "bpc_hemolysis/model_julia" 
    containerOptions "-p ${umbridge_port+100*(task.attempt-1)}:${umbridge_port+100*(task.attempt-1)}"
    // container "file://$moduleDir/model_julia/container.sif"
    cache 'lenient'
    errorStrategy 'retry'
    maxRetries 3

    input:
    path script
    path src
    val name
    path data
    val umbridge_port
    path um_highway

    script:
    """
    #!/bin/bash
    UMBRIDGE_PORT=${umbridge_port+100*(task.attempt-1)}

    # Abort if port already used (success means something is listening)
    if bash -c "echo > /dev/tcp/localhost/\$UMBRIDGE_PORT" 2>/dev/null; then
        echo "Port \$UMBRIDGE_PORT already in use"
        exit 1
    fi

    # Start model server
    julia ${script} --name ${name} --data ${data} --port \$UMBRIDGE_PORT &
    SERVER_PID=\$!
    trap 'kill \$SERVER_PID 2>/dev/null || true' EXIT INT TERM
    echo "Model Server PID: \$SERVER_PID (trying port \$UMBRIDGE_PORT)"

    # Wait for model server to start
    while ! bash -c "echo > /dev/tcp/localhost/\$UMBRIDGE_PORT" 2>/dev/null; do
        sleep 1
    done

    touch $um_highway/model_info/READY
    echo "Model ${name} ready" > $um_highway/model_info/READY

    touch $um_highway/umbridge_port/\$UMBRIDGE_PORT
    echo "Model server is running on port \$UMBRIDGE_PORT"

    # Monitor the status 
    echo "Waiting for UQ to complete..."
    until [ -e $um_highway/uq_info/DONE ]; do
        sleep 1
    done

    # Stop the model server when the signal is received
    kill \$SERVER_PID 2>/dev/null || true
    echo "Model server on port \$UMBRIDGE_PORT stopped."

    """
}

process MCMC_CALIBRATION {
    conda "$moduleDir/mcmc/environment.yml"
    cache 'lenient'

    input:
    path script
    val key
    val params_yaml
    path data
    path um_highway

    output:
    tuple val(key), path("mcmc_output.npz"), emit: mcmc_output
    tuple val(key), path("corner_plot.png"), emit: mcmc_corner_plot
    tuple val(key), path("trace.npy"),       emit: mcmc_trace
    tuple val(key), path("mcmc_idata.nc"),   emit: mcmc_idata

    script:
    """
    echo "${params_yaml}" > _params.yml

    echo "Waiting for model server to start..."
    until [ -e $um_highway/model_info/READY ]; do
        sleep 1
    done
    cat $um_highway/model_info/READY

    MODEL_PORT=\$(ls $um_highway/umbridge_port/ | head -n 1)
    echo "Model server is running on port \${MODEL_PORT}"

    python ${script}  --config _params.yml --data ${data} --port \${MODEL_PORT}

    touch $um_highway/uq_info/DONE # signal to stop the model server
    """
}


process RUN_DIAGNOSTICS {
    conda "$moduleDir/diagnostics/environment.yml"

    input:
    path script
    tuple val(key), path(mcmc_idata)
    val outdir

    output:
    tuple val(key), path("${outdir}")

    script:
    """
    #!/bin/bash
    python3 ${script} --idata-path ${mcmc_idata} --output-dir "${outdir}"
    """
}

process BUNDLE_OUTPUTS {
    publishDir "${output_base_dir}", mode: 'copy'

    input:
    val output_base_dir
    val bundle_name
    val params_yaml
    path mcmc_output
    path mcmc_corner_plot
    path mcmc_trace
    path mcmc_idata
    path mcmc_diagnostics

    output:
    path "${bundle_name}", emit: bundle_dir

    script:
    """
    #!/bin/bash
    echo "${params_yaml}" > params.yml

    mkdir "${bundle_name}"

    cp params.yml "${bundle_name}"
    cp ${mcmc_output} "${bundle_name}/"
    cp ${mcmc_corner_plot} "${bundle_name}/"
    cp ${mcmc_trace} "${bundle_name}/"
    cp ${mcmc_idata} "${bundle_name}/"
    cp -r ${mcmc_diagnostics} "${bundle_name}/"
    """
}


process GENERATE_REPORT {
    conda "$moduleDir/report/environment.yml"
    publishDir "${output_base_dir}/${bundle_dir.name}", mode: 'copy'

    input:
    path script
    val output_base_dir
    path bundle_dir

    output:
    path "report.pdf"

    script:
    """
    #!/bin/bash
    python3 ${script} \\
        --bundle-dir ${bundle_dir} \\
        --output report.pdf
    """
}

workflow {
    def cfg = new groovy.yaml.YamlSlurper().parseText(file(params.config_file).text) as Map
    bpc_hemolysis(
        Channel.value(file(params.config_file)),
        cfg.model.use_julia,
        "$projectDir/outputs".toString()
    )
}