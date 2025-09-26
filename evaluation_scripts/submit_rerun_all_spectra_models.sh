#!/bin/bash
#$ -l qu=gtx
#$ -cwd
#$ -o train.out
#$ -e train.err
#$ -q gtx02a,gtx02b,gtx02c,gtx02d,gtx03a,gtx03b,gtx03c,gtx03d,gtx05a,gtx05b,gtx05c,gtx05d,gtx09a,gtx09b,gtx09c,gtx09d,gtx10a,gtx10b,gtx10c,gtx10d

script_path="/home/cschmidt/bin_public/excited_states_networks/nn_scripts/rerun_all_spectra_models.sh"

# Which GPU?
gpu_id=$( echo $QUEUE | awk '/a/ {print 0} /b/ {print 1}  /c/ {print 2}  /d/ {print 3}')

# How many cores are there?
case $HOSTNAME in
    gtx0[1-6]*)
    cores=10;
    ;;
    gtx0[7-8]*)
    cores=12;
    ;;
    gtx09*)
    cores=16;
    ;;
    gtx10*)
    cores=16;
    ;;
    *)
    echo "Error: Unknown compute node $HOSTNAME"
    echo "       This script only works for gtx01 thru 10!"
    echo
    exit -1
    ;;
esac

# Echo important information
echo "# Hostname: " `hostname`
echo "# Job ID: " $JOB_ID
echo "# gpuid: " $gpu_id

# Activate conda environment
source /home/cschmidt/miniconda3/bin/activate excited_states_NN_kgcnn

# Set environment variables
export OMP_NUM_THREADS=$cores
export CUDA_VISIBLE_DEVICES=$gpu_id
ulimit -s unlimited

# Start time
start=$( date "+%s" )

# Execute the script
echo "Starting rerun_all_spectra_models.sh at $(date)"
$script_path

# End time calculation
end=$( date "+%s" )
duration=$(( end - start ))
DAYS=$(( duration / 86400 ))
HOURS=$(( (duration % 86400) / 3600 ))
MINS=$(( ((duration % 86400) % 3600) / 60 ))
SECS=$(( ((duration % 86400) % 3600) % 60 ))
echo "Time taken: $DAYS days, $HOURS hours, $MINS minutes and $SECS seconds."