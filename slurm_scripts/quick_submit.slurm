#!/bin/bash

#SBATCH --job-name=quickslurm
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest

#SBATCH --time=01:00:00
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
SCRIPT=$3
VAR=$4
# if four user inputs, then run the script with the variable
if [ $# -eq 4 ]; then
	echo "time $SCRIPT $BASIN $WY -var $VAR"
	time $SCRIPT $BASIN $WY -var $VAR
else
	echo "time $SCRIPT $BASIN $WY"
	time $SCRIPT $BASIN $WY
fi

#conda activate smeshr
#BASIN=$1
#WY=$2
#TOPO=$3
#cd /uufs/chpc.utah.edu/common/home/skiles-group3/SMESHR/
#echo "time prepare_MODIS_HRRR_inputs.sh $BASIN $WY $TOPO 2>&1 | tee prep_${BASIN}_wy${WY}_$(date +%Y%m%d_%H%M).log"
#time prepare_MODIS_HRRR_inputs.sh $BASIN $WY $TOPO 2>&1 | tee prep_${BASIN}_wy${WY}_$(date +%Y%m%d_%H%M).log

#for wy in 2021
#do
	#echo "time plot_eb_terms.py $basin $wy"
	#time plot_eb_terms.py $basin $wy
	#echo "time net_solar_checks.py $basin $wy"
	#time net_solar_checks.py $basin $wy
	#time net_solar_checks.py $basin $wy -st AZ -e 32612
#done

# standard output directory setup
LOGDIR=/uufs/chpc.utah.edu/common/home/u6058223/quickslurm_logs
if [ ! -e $LOGDIR ] ; then
    mkdir -pv $LOGDIR
fi

# Move log files to log dir
mv slurm-${SLURM_JOB_ID}* $LOGDIR
