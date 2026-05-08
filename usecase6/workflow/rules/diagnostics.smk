rule run_diagnostics:
    input:
        script     = "diagnostics/run_diagnostics.py",
        mcmc_idata = f"{WORK}/{{experiment}}/mcmc_idata.nc",
    output:
        directory(f"{WORK}/{{experiment}}/diagnostics/")
    conda: "../../diagnostics/environment.yml"
    shell:
        """
        python3 {input.script} \
            --idata-path {input.mcmc_idata} \
            --output-dir {output}
        """
