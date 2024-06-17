#!/bin/bash
#SBATCH --job-name=build_VAEs_esm_PG
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-839%25
#SBATCH --time=2-12:00:00
#SBATCH --gres=gpu:1

CONDA_ACTIVATE=$CONDA_PREFIX/etc/conda/activate.d
PROJECT_DIR=/home/pcq275/disentangling-vae/
CONFIG=${PROJECT_DIR}scripts/slurm_config/vae_types_config.txt

if [[ ! -d ${CONDA_ACTIVATE} ]]; then
	mkdir -p ${CONDA_ACTIVATE} 
fi

## Activate conda environment
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh

conda activate disvae-env

dataset=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG})
loss=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $3}' ${CONFIG})
embedding=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG})
modeltype=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $5}' ${CONFIG})
msa=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $6}' ${CONFIG})
latent_dim=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $7}' ${CONFIG})
rec_dist=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $8}' ${CONFIG})
subset=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $9}' ${CONFIG})

vae_type="${loss}_${dataset}_${embedding}_${latent_dim}_${subset}_${rec_dist}"

cd ${PROJECT_DIR}  # required to be run from here due to relative experiment config paths in repo,

if [ -z "${subset}" ]; then
    if [[ "${msa}" == "True" ]]; then
        vae_type="${vae_type}_MSA"
        echo "python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} --msa_only -m ${modeltype} --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100"
        python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} --msa_only -m ${modeltype} --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100
        exit 0
    else 
        echo "python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} -m ${modeltype} --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100"
        python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} -m ${modeltype} --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100
        exit 0
    fi
else
    if [[ "${msa}" == "True" ]]; then
        vae_type="${vae_type}_MSA"
        echo "python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} --subset ${subset} --msa_only -m ${modeltype} --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100"
        python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} --subset ${subset} --msa_only -m ${modeltype} --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100
        exit 0
    else 
        echo "python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} --subset ${subset} -m ${modeltype} --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100"
        python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} --subset ${subset} -m ${modeltype} --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100
        exit 0
    fi
fi