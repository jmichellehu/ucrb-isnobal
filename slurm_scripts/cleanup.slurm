#!/bin/bash

#SBATCH --job-name=cleanup_data
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest

#SBATCH --time=01:00:00
#SBATCH --ntasks=12
#SBATCH --mem=64G

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
conda activate snobedo

BASIN=$1
WY=$2

### iSnobal compaction
parallel --jobs 4 "smrf_compactor -sd {} --delete-originals ::: /uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/${BASIN}_100m_isnobal/wy${WY}/${BASIN}_basin_100m/run*/"
parallel --jobs 4 "smrf_compactor -sd {} --delete-originals -eb ::: /uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/${BASIN}_100m_isnobal/wy${WY}/${BASIN}_basin_100m/run*/"

### DAILY snow/em compression
compress_outputs.sh /uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/${BASIN}_100m_isnobal/wy${WY}/${BASIN}_basin_100m/ 

# standard output directory setup
LOGDIR=/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/slurm_scripts/slurm_logss
if [ ! -e $LOGDIR ] ; then
    mkdir -pv $LOGDIR
fi

# Move log files to log dir
mv slurm-${SLURM_JOB_ID}* $LOGDIR
