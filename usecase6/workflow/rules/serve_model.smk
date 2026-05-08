rule serve_model_py:
    input:
        script = "model/IH_model_server.py",
        src    = "model/src",
        data   = data_for_experiment,
    output:
        ready = service(f"{WORK}/comm/{{experiment}}/READY_py"),
    params:
        port      = lambda wc: experiment_port(wc.experiment),
        name      = lambda wc: get_exp_cfg(wc.experiment)["model"]["name"],
    threads: lambda wc: int(get_exp_cfg(wc.experiment)["model"].get("n_workers", 1))
    conda: "../../model/environment.yml"
    shell:
        """
        python3 {input.script} \
            --name      {params.name} \
            --data      {input.data} \
            --port      {params.port} \
            --n_workers {threads} &
        SERVER_PID=$!
        python3 -c "
import socket, time
while True:
    try:
        socket.create_connection(('localhost', {params.port}), timeout=1).close()
        break
    except OSError:
        time.sleep(1)
"
        mkdir -p $(dirname {output.ready})
        echo "Model {params.name} ready on port {params.port}" > {output.ready}
        wait $SERVER_PID
        """


rule serve_model_jl:
    input:
        script  = "model_julia/IH_model_server.jl",
        src     = "model_julia/src",
        data    = data_for_experiment,
    output:
        ready = service(f"{WORK}/comm/{{experiment}}/READY_jl"),
    params:
        port    = lambda wc: experiment_port(wc.experiment),
        name    = lambda wc: get_exp_cfg(wc.experiment)["model"]["name"],
        basedir = lambda wc: workflow.basedir,
    shell:
        """
        docker run --rm \
            -p {params.port}:{params.port} \
            -v {params.basedir}:/workspace \
            bpc_hemolysis/model_julia \
            julia /workspace/{input.script} \
                --name {params.name} \
                --data /workspace/{input.data} \
                --port {params.port} &
        SERVER_PID=$!
        python3 -c "
import socket, time
while True:
    try:
        socket.create_connection(('localhost', {params.port}), timeout=1).close()
        break
    except OSError:
        time.sleep(1)
"
        mkdir -p $(dirname {output.ready})
        echo "Model {params.name} ready on port {params.port}" > {output.ready}
        wait $SERVER_PID
        """
