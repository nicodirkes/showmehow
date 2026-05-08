rule preprocess_data:
    input:
        script = "preprocessing/preprocessing.py",
        indir  = f"{WORK}/pulled_data/",
    output:
        f"{WORK}/data/data_ding_{{species}}_processed.csv"
    conda: "../../preprocessing/environment.yml"
    shell:
        """
        python3 {input.script} {wildcards.species} \
            --prefix data_ding_ \
            --indir  {input.indir} \
            --outdir $(dirname {output})
        """
