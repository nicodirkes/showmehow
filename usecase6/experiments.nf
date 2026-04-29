#!/usr/bin/env nextflow
include { bpc_hemolysis } from './main.nf'

workflow {
    config = new groovy.yaml.YamlSlurper().parseText(file(params.config_file).text)

    exp_base_dir  = "$projectDir/outputs/experiments/${workflow.start}_${workflow.sessionId}".toString()
    params_yaml   = new groovy.yaml.YamlBuilder().tap { it(config) }.toString()

    experiments = GENERATE_EXPERIMENTS(
        script      = file("$projectDir/experiments/generate_experiments.py"),
        params_yaml = params_yaml
    ).flatten()

    def use_julia = config.containsKey('groups')
        ? config.groups.any { it.base_params?.model?.use_julia == true }
        : config.base_params.model.use_julia

    bpc_hemolysis(experiments, use_julia, exp_base_dir)
}

process GENERATE_EXPERIMENTS {
    conda "$projectDir/experiments/environment.yml"
    publishDir "$projectDir/outputs/experiments/${workflow.start}_${workflow.sessionId}/", mode: 'copy'

    input:
    path script
    val params_yaml

    output:
    path "experiment_*.yml"

    script:
    """
    echo "${params_yaml}" > params.yml
    python3 ${script} params.yml
    """
}
