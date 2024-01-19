#!/bin/bash
#SBATCH --job-name=IBMA
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_40C_512G
# Outputs ----------------------------------
#SBATCH --output=log/%x_%A-%a.out
#SBATCH --error=log/%x_%A-%a.err
# ------------------------------------------

# IB_44C_512G, IB_40C_512G, IB_16C_96G, for running workflow
# investor, for testing
pwd; hostname; date

#==============Shell script==============#
# Load evironment
source /home/data/nbc/misc-projects/Peraza_large-scale-ibma/env/activate_env
set -e

PROJECT_DIR="/home/data/nbc/misc-projects/Peraza_large-scale-ibma"

# Setup done, run the command
cmd="python ${PROJECT_DIR}/ibma_workflow.py \
    --project_dir ${PROJECT_DIR} \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --n_cores ${SLURM_CPUS_PER_TASK}"
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo Finished tasks with exit code $exitcode
exit $exitcode

date