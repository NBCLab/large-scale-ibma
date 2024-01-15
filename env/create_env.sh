#!/bin/bash
module add miniconda3-4.5.11-gcc-8.2.0-oqs2mbg
conda create -p /home/data/nbc/misc-projects/Peraza_large-scale-ibma/env/nv-ibma_env pip python=3.10 -y

conda config --append envs_dirs /home/data/nbc/misc-projects/Peraza_large-scale-ibma/env

source activate /home/data/nbc/misc-projects/Peraza_large-scale-ibma/env/nv-ibma_env

pip install -r /home/data/nbc/misc-projects/Peraza_large-scale-ibma/requirements.txt