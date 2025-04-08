workflow {
    STATUS()
    MODEL( STATUS.out.pipe ) 
    MCMC ( STATUS.out.pipe ) 
}

process STATUS {
    script:
    """
    mkfifo status_info
    """

    output:
    path "status_info", emit: pipe
}

process MODEL {

    input:
    path status

    script:
    """
    # Sleep 100000 represents the model server
    # which is started in the background
    # and its process ID is stored, inorder to kill later
    sleep 1000000 &> /dev/null & 
    PID=\$! && echo "Model Server Started" && echo "model=1" > $status
        
    # Wait for MCMC to finish | Look for "mcmc=0" in status
    tail -f $status | grep -q "mcmc=0" 
    
    # Stop model server (kill process)
    kill \$PID &> /dev/null
    echo "Model Server Stopped"
    rm $status
    """
}

process MCMC {
    publishDir "$moduleDir/outputs", mode: 'copy'
    
    input:
    path status

    script:
    """
    # Wait for MODEL to start | Look for "model=1" in status
    tail -f $status | grep -q "model=1"
    
    echo "MCMC Started"
    # sleep 1 represents the MCMC program
    sleep 1
    echo "MCMC Finished"
    # Store MCMC results
    echo "42" > results.txt
    
    echo "mcmc=0" > $status
    """

    output:
    path 'results.txt'
}