#!/bin/bash

for lead in 10 15 20 25
do 
    export lead
    sbatch run_mk_job.sh
done
