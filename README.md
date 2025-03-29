# The I in FAIR: Solutions for HeterOgeneous Workflows in Model-based Engineering applications – SHOWME.how
This repository contains data used in the contribution to the NFDI4ING Community Meeting - "Innovating Software Solutions in Engineering Research"

## Authors
- Alan Correa (RWTH Aachen, presenting author)
- V Mithlesh Kumar (RWTH Aachen), Anil Yildiz (RWTH Aachen), Julia Kowalski (RWTH Aachen)


## Abstract:

Model-based development, design, decision support, and diagnostics in engineering applications often require complex workflows that involve heterogeneous computational models, programming languages, operating systems, and computing infrastructures. The diversity of workflow components presents significant challenges in interoperability, limiting the usability and subsequent reusability of the generated digital assets, including software and data.

To address these challenges, we present an approach to orchestrate various computational workflow components, enhancing their combined value through synergy. Additionally, it facilitates the seamless integration of state-of-the-art research methodologies into application-driven engineering tasks. We call this method SHOWME.how. It is enabled by technologies like package managers (Conda, Mamba), containerization (Apptainer, Docker), data exchange protocols (HTTP, Filesystem), and workflow managers (Nextflow).

Using SHOWME.how, we develop blueprints to perform high-throughput tasks for applications in heat transfer and free-surface flow, that use different software for computational models (OpenFOAM, FEniCS) and packages in various languages (Python, Julia, R). Our approach enables the reuse of existing research software in its native environment, eliminating the need to build wrappers, develop language-bridging interfaces, or rely on suboptimal implementations. It also streamlines the prototyping of complex computational workflows for engineering studies,  facilitating their creation and reuse with minimal effort.

The demonstrated blueprints can be readily adapted to develop highly interoperable and reusable digital assets and software frameworks, ultimately increasing their longevity and value.

## Keywords
Computational Workflows, Model-based Engineering, High-Throughput Computing, Uncertainty Quantification


## Usage

### Requirements

1. Conda or Mamba
    - Conda is a package manager that allows installing and managing software packages and their dependencies. It is widely used in the scientific computing community.
    - Mamba/Micromamba is a fast, drop-in replacement for Conda that uses parallel downloading and dependency resolution to speed up package installation.
    - To install Conda, follow the instructions on the [Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
    - To install Mamba, follow the instructions on the [Mamba website](https://mamba.readthedocs.io/en/latest/installation.html).
    - To install Micromamba, follow the instructions on the [Micromamba website](https://mamba.readthedocs.io/en/latest/installation.html#micromamba).
2. Nextflow 
    - Nextflow is a workflow management system that allows writing and executing   data-driven workflows. It is designed to be portable and can run on various platforms, including local machines, clusters, and cloud environments. For the official installation instructions, visit the [Nextflow website](https://www.nextflow.io/docs/latest/getstarted.html).
    - As an easier alternative, we can use the following commands:
        Using conda:  
        ```
        conda install conda-forge::curl bioconda::nextflow=24.10.5
        ```
        Using mamba:  
        ```
        mamba install conda-forge::curl bioconda::nextflow=24.10.5
        ```
        Using micromamba:  
        ```
        micromamba install conda-forge::curl bioconda::nextflow=24.10.5
        ```

3. Apptainer
   - Apptainer is a containerization tool that allows us to create and run containers. It is similar to Docker but is designed for high-performance computing (HPC) environments.
   - To install Apptainer, follow the instructions on the [Apptainer website](https://apptainer.org/docs/admin/latest/installation/).
  
4. Docker
   - Docker is a containerization tool that allows us to create and run containers. It is widely used for developing, shipping, and running applications.
   - To install Docker, follow the instructions on the [Docker website](https://docs.docker.com/get-docker/).