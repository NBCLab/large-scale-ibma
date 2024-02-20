#!/bin/bash
#SBATCH --job-name=get-nv-collections
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
QUERY_ID="a444c1d1cc79f746a519d97ce9672089"

# Setup done, run the command
cmd="python ${PROJECT_DIR}/get_nv_collections.py \
    --project_dir ${PROJECT_DIR}
    --pg_query_id ${QUERY_ID}"
echo Commandline: $cmd
eval $cmd

# Output results to a table
echo Finished tasks with exit code $exitcode
exit $exitcode

date