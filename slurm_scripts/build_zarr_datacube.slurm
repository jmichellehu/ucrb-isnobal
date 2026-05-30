#!/bin/bash
# Build full water-year Zarr data cubes from daily iSnobal model output.
#
# Usage:
#   sbatch build_zarr_datacube.slurm <BASIN> <WY>
#   sbatch build_zarr_datacube.slurm animas 2022
#
# Output Zarr stores are written to the default --outdir in build_zarr_datacube.py.
# Override with: sbatch build_zarr_datacube.slurm animas 2022 /path/to/outdir

#SBATCH --job-name=bJ24
#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest

#SBATCH --time=06:00:00
#SBATCH --ntasks=4
#SBATCH --mem=128G

#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=j.michelle.hu@utah.edu

#SBATCH --output=slurm-%j_build_zarr.out-%N
#SBATCH --error=slurm-%j_build_zarr.err-%N

echo "Running $0 from $SLURM_SUBMIT_DIR"
echo "        SLURM_JOB_ID:        $SLURM_JOB_ID"
echo "        SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "        SLURM_JOB_NODELIST:  $SLURM_JOB_NODELIST"
echo "        Start: $(date)"

module use $HOME/MyModules
module load miniconda3/latest
conda activate studio

BASIN=${1:?'Usage: sbatch build_zarr_datacube.slurm <BASIN> <WY> [OUTDIR]'}
WY=${2:?'Usage: sbatch build_zarr_datacube.slurm <BASIN> <WY> [OUTDIR]'}
OUTDIR=${3:-}   # optional; leave empty to use script default

SCRIPT_DIR=/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/slurm_scripts
SCRIPT="/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/build_zarr_datacube.py"

OUTDIR_ARG=""
if [ -n "$OUTDIR" ]; then
    OUTDIR_ARG="-o $OUTDIR"
fi

# for DATASET in em snow; do
#     echo ""
#     echo "=== Building $DATASET cube: basin=$BASIN  wy=$WY ==="
#     time python "$SCRIPT" "$BASIN" "$WY" -var "$DATASET" $OUTDIR_ARG -v
# done

echo ""
echo "=== Building $DATASET cube: basin=$BASIN  wy=$WY ==="
time python "$SCRIPT" "$BASIN" "$WY" -var em --xy-chunks 300 --overwrite -v $OUTDIR_ARG

echo ""
echo "End: $(date)"

# Move log files to log dir
LOGDIR="$SCRIPT_DIR/slurm_logs"
mkdir -p "$LOGDIR"
mv slurm-${SLURM_JOB_ID}* "$LOGDIR/" 2>/dev/null || true
