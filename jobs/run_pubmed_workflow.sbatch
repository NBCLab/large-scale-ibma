#!/bin/bash
#SBATCH --job-name=pubmed-IBMA-filtered
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --account=iacc_nbc
#SBATCH --qos=pq_nbc
#SBATCH --partition=IB_40C_512G
# Outputs ----------------------------------
#SBATCH --output=log/%x_%A-%a.out
#SBATCH --error=log/%x_%A-%a.err
# ------------------------------------------

# IB_44C_512G, IB_40C_512G, IB_16C_96G, for running workflow
# investor, for testing
# sbatch --array=0-24 run_pubmed_workflow.sbatch
pwd; hostname; date

#==============Shell script==============#
# Load evironment
source /home/data/nbc/misc-projects/Peraza_large-scale-ibma/env/activate_env
set -e

PROJECT_DIR="/home/data/nbc/misc-projects/Peraza_large-scale-ibma"
DSET_TYPE="filtered"
# "raw", "filtered"

# Setup done, run the command
cmd="python ${PROJECT_DIR}/pubmed_workflow.py \
    --project_dir ${PROJECT_DIR} \
    --dset_type ${DSET_TYPE} \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --n_cores ${SLURM_CPUS_PER_TASK}"
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo Finished tasks with exit code $exitcode
exit $exitcode

date