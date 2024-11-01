#!/bin/bash

#SBATCH --job-name=SDDbasin
#SBATCH --account=skiles
#SBATCH --partition=notchpeak

#SBATCH --time=2:00:00
#SBATCH --ntasks=2
#SBATCH --mem=128G

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=j.michelle.hu@utah.edu

#SBATCH --output=slurm-%j_calcsdd.out-%N
#SBATCH --error=slurm-%j_calcsdd.err-%N

# Usage: sbatch calc_sdd.slurm <basin> <-t daythresh>

# Log some items
echo "Running $0 [$HOME/git_dirs/ucrb-isnobal/slurm_scripts/calc_sdd.slurm] $1 from $SLURM_SUBMIT_DIR"
echo "        with SLURM_JOB_ID:  $SLURM_JOB_ID"
echo "        with SLURM_JOB_PARTITION:  $SLURM_JOB_PARTITION"
echo "        with SLURM_JOB_NODELIST:  $SLURM_JOB_NODELIST"

module use $HOME/MyModules
module load miniconda3/latest
conda activate studio

basin=$1
daythresh=$2

echo "calc_basin_sdd.py $basin -t $daythresh"
calc_basin_sdd.py $basin -t $daythresh

# standard output directory setup
LOGDIR=/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/slurmlogs/
if [ ! -e $LOGDIR ] ; then
    mkdir -pv $LOGDIR
fi

# Move log files to log dir
mv slurm-${SLURM_JOB_ID}* $LOGDIR
