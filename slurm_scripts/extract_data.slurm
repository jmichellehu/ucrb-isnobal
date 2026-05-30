#!/bin/bash

#SBATCH --job-name=extract_data
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest

#SBATCH --time=04:00:00
#SBATCH --ntasks=4
#SBATCH --mem=128G

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=j.michelle.hu@utah.edu

#SBATCH --output=slurm-%j_quicksubmit.out-%N
#SBATCH --error=slurm-%j_quicksubmit.err-%N

# Usage: sbatch quicksubmit.slurm

# Log some items
echo "Running $0 from $SLURM_SUBMIT_DIR"
echo "        with SLURM_JOB_ID:  $SLURM_JOB_ID"
echo "        with SLURM_JOB_PARTITION:  $SLURM_JOB_PARTITION"
echo "        with SLURM_JOB_NODELIST:  $SLURM_JOB_NODELIST"

module use $HOME/MyModules
module load miniconda3/latest
conda activate studio

BASIN=$1
WY=$2
SCRIPT=extract_data.py
for VAR in em snow smrf_2 smrf_energy_balance
do
	echo "time $SCRIPT $BASIN $WY -var $VAR"
	time $SCRIPT $BASIN $WY -var $VAR
done

# standard output directory setup
LOGDIR=/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/slurm_scripts/slurm_logss
if [ ! -e $LOGDIR ] ; then
    mkdir -pv $LOGDIR
fi

# Move log files to log dir
mv slurm-${SLURM_JOB_ID}* $LOGDIR
