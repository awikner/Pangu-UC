#!/bin/bash -l
#SBATCH --time=1-00:00:00
#SBATCH -p gpu
#SBATCH --mem-per-cpu=4G 
#SBATCH --nodes=1
#SBATCH --gpus=a40:4       #gpus=a100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8 #16 
#SBATCH -o /scratch/user/u.pr160292/PanguWeather/v2.0/outs/faster_ddp.out

#echo $SLURM_NTASKS   # WORLD_SIZE
#echo $SLURM_PROCID   # WORLD_RANK
#echo $SLURM_LOCALID  # LOCAL_RANK


export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)


set -x
srun -u --mpi=pmi2 \
    bash -c "
    source /scratch/user/u.pr160292/PanguWeather/v2.0/export_DDP_vars.sh
    python /scratch/user/u.pr160292/PanguWeather/v2.0/objective.py 
    "
