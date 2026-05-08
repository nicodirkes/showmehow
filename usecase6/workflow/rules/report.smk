rule generate_report:
    input:
        script = "report/generate_report.py",
        bundle = f"outputs/{RUN_ID}/{{experiment}}/bundle/",
    output:
        f"outputs/{RUN_ID}/{{experiment}}/report.pdf"
    conda: "../../report/environment.yml"
    shell:
        """
        python3 {input.script} \
            --bundle-dir {input.bundle} \
            --output     {output}
        """
