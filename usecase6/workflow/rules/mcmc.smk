def _ready_file(wc):
    use_julia = get_exp_cfg(wc.experiment)["model"]["use_julia"]
    suffix = "jl" if use_julia else "py"
    return f"{WORK}/comm/{wc.experiment}/READY_{suffix}"


rule mcmc_calibration:
    input:
        script = "mcmc/calibrate_emcee.py",
        data   = data_for_experiment,
        ready  = _ready_file,
        cfg    = f"{EXP_DIR}/{{experiment}}.yml",
    output:
        mcmc_output = f"{WORK}/{{experiment}}/mcmc_output.npz",
        corner_plot = f"{WORK}/{{experiment}}/corner_plot.png",
        trace       = f"{WORK}/{{experiment}}/trace.npy",
        mcmc_idata  = f"{WORK}/{{experiment}}/mcmc_idata.nc",
    params:
        port = lambda wc: experiment_port(wc.experiment),
    threads: lambda wc: int(get_exp_cfg(wc.experiment)["calibration"].get("n_workers", 1))
    conda: "../../mcmc/environment.yml"
    shell:
        """
        SCRIPT=$(realpath {input.script})
        CONFIG=$(realpath {input.cfg})
        DATA=$(realpath {input.data})
        OUT_NPZ=$(realpath {output.mcmc_output})
        OUT_PNG=$(realpath {output.corner_plot})
        OUT_NPY=$(realpath {output.trace})
        OUT_NC=$(realpath {output.mcmc_idata})
        mkdir -p $(dirname $OUT_NPZ)
        TMPDIR=$(mktemp -d)
        trap "rm -rf $TMPDIR" EXIT
        cd $TMPDIR
        python3 $SCRIPT --config $CONFIG --data $DATA --port {params.port}
        mv mcmc_output.npz $OUT_NPZ
        mv corner_plot.png $OUT_PNG
        mv trace.npy       $OUT_NPY
        mv mcmc_idata.nc   $OUT_NC
        """
