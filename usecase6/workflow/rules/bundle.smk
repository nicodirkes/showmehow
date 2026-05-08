rule bundle_outputs:
    input:
        cfg         = f"{EXP_DIR}/{{experiment}}.yml",
        mcmc_output = f"{WORK}/{{experiment}}/mcmc_output.npz",
        corner_plot = f"{WORK}/{{experiment}}/corner_plot.png",
        trace       = f"{WORK}/{{experiment}}/trace.npy",
        mcmc_idata  = f"{WORK}/{{experiment}}/mcmc_idata.nc",
        diagnostics = f"{WORK}/{{experiment}}/diagnostics/",
    output:
        bundle = directory(f"outputs/{RUN_ID}/{{experiment}}/bundle/")
    params:
        bundle_name = lambda wc: (
            "{species}_{model}_{hash}".format(
                species = get_exp_cfg(wc.experiment)["species"],
                model   = get_exp_cfg(wc.experiment)["model"]["name"],
                hash    = get_exp_cfg(wc.experiment).get(
                              "experiment_hash",
                              wc.experiment.removeprefix("experiment_")
                          ),
            )
        )
    shell:
        """
        mkdir -p {output.bundle}
        cp    {input.cfg}         {output.bundle}/params.yml
        cp    {input.mcmc_output} {output.bundle}/
        cp    {input.corner_plot} {output.bundle}/
        cp    {input.trace}       {output.bundle}/
        cp    {input.mcmc_idata}  {output.bundle}/
        cp -r {input.diagnostics} {output.bundle}/
        """
