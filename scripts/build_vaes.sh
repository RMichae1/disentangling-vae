#!/bin/bash
#SBATCH --job-name=build_VAEs_esm2_PG
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-167%10
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
latent_dim=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $5}' ${CONFIG})
aggregate=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $6}' ${CONFIG})
rec_dist=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $7}' ${CONFIG})

vae_type="${loss}_${dataset}_${embedding}_${latent_dim}"

cd ${PROJECT_DIR}  # required to be run from here due to relative experiment config paths in repo,


if [[ "${aggregate}" == "True" ]]; then

    vae_type="${vae_type}_meanpooled"
    echo "python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} -m Seq --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100 --aggregate"
    python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} -m Seq --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100 --aggregate
    exit 0
else 
    echo "python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} -m Seq --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100"
    python ${PROJECT_DIR}main.py ${vae_type} -d ${dataset} -m Seq --embedding ${embedding} -z ${latent_dim} -l ${loss} -r ${rec_dist} --lr 0.001 -b 256 -e 100
    exit 0
fi