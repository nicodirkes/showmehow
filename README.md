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
