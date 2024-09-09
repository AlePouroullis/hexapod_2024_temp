#!/bin/bash
#PBS -l select=1:ncpus=24:mpiprocs=24
#PBS -P CSCI1142
#PBS -q smp
#PBS -l walltime=90:00:00
#PBS -o logs/10k.out
#PBS -e logs/10k.err
#PBS -m abe
#PBS -M your_email@example.com
#PBS -N NEAT_MapElites_10k

# Set unlimited stack size
ulimit -s unlimited

module purge
module load chpc/python/anaconda/3
conda init bash
eval "$(conda shell.bash hook)"
conda activate /home/apouroullis/lustre/neat_mapelites_env

# Change to the working directory
cd $PBS_O_WORKDIR

# Run the Python script
python3 neat_map_elites.py 10000 1
# Deactivate the virtual environment
conda deactivate