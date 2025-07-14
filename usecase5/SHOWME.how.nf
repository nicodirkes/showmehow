#!/usr/bin/env nextflow

workflow {
    // STAGE_DATA (
    //     file("$moduleDir/stage_data/.token.txt")
    // )

    IH_MODEL(
        file("$moduleDir/model/data.csv")
    )
}

process STAGE_DATA {
    conda "$moduleDir/stage_data/environment.yml"
    publishDir "$moduleDir/outputs", mode: 'copy'
    
    input:
      path token

    output:
      file "**/data*.csv"

    script:
        """
        #!/usr/bin/env python3

        import coscine
        with open("$token", "rt") as fp:
	        token = fp.read()
        # You can now use token to intialize the coscine ApiClient!

        print(token)
        client = coscine.ApiClient(token)
        resource = client.project("showmehow_usecase5").resource("Field Data")
        resource.download(path="./")

        """

}

process IH_MODEL  {
    conda "$moduleDir/model/environment.yml"

    input:
    path data

    output:

    script:
    """
    python $moduleDir/model/IH_power_law.py
    """
}