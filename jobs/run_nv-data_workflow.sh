#!/bin/bash
#SBATCH --job-name=nv-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=iacc_nbc
#SBATCH --qos=qos_download
#SBATCH --partition=download
# Outputs ----------------------------------
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
# ------------------------------------------

pwd; hostname; date

#==============Shell script==============#
# Load evironment
source /home/data/nbc/misc-projects/Peraza_large-scale-ibma/env/activate_env
set -e

PROJECT_DIR="/home/data/nbc/misc-projects/Peraza_large-scale-ibma"
DSET_TYPE="raw"

# Setup done, run the command
cmd="python ${PROJECT_DIR}/nv-data_workflow.py --project_dir ${PROJECT_DIR}"
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo Finished tasks with exit code $exitcode
exit $exitcode

date