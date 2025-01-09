#!/bin/bash
#$ -l qu=gtx
#$ -cwd
#$ -o train.out
#$ -e train.err
#$ -q gtx02a,gtx02b,gtx02c,gtx02d,gtx03a,gtx03b,gtx03c,gtx03d,gtx05a,gtx05b,gtx05c,gtx05d,gtx09a,gtx09b,gtx09c,gtx09d,gtx10a,gtx10b,gtx10c,gtx10d

pythonfile=$1
pythonfile=${pythonfile#"/srv/nfs"}

data_file="$2" # mandatory

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

# Echo important information into file
echo "# Hostname: " `hostname`
echo "# Job ID: " $JOB_ID
echo "# gpuid: " $gpu_id

# In case of external API usage I saved some API-keys here
if [ -f ~/.api_keys ]; then
    . ~/.api_keys
fi

# For WandB:
export WANDB_MODE=offline # no internet connection during calculation on nodes

# For data readin in kgcnn
export BABEL_DATADIR="/usr/local/run/openbabel-2.4.1"

# export PATH="/home/lpetersen/anaconda_interpreter/bin:$PATH"

# if [[ "$pythonfile" == *"forces"* ]]; then
#     source /home/lpetersen/anaconda_interpreter/etc/profile.d/conda.sh # from lukas, this is for forces
#     conda activate kgcnn_new
# elif [[ "$pythonfile" == *"spectrum"* ]]; then
#     # source /home/cschmidt/miniconda3/bin/activate excited_states_NN_from_monja # for nn_spectrum, this is for spectrum (based on an environment from monja)
#     # conda activate excited_states_NN_from_monja
#     source /home/lpetersen/anaconda_interpreter/etc/profile.d/conda.sh
#     conda activate kgcnn_new
# else
#     echo "Error: Unknown python file type"
#     exit 1
# fi

source /home/cschmidt/miniconda3/bin/activate excited_states_NN_kgcnn
conda activate excited_states_NN_kgcnn

# Deprecated CUDA setting on server
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/run/cuda/lib

# Even older deprecated CUDA setting on server
#CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# set OpenMP parallel threads variable:
export OMP_NUM_THREADS=$cores

# OpenMP needs this: set stack size to unlimited
ulimit -s unlimited
# Start time of calculation
start=$( date "+%s" )

if [ -z "$data_file" ]
then
    echo "Error: data_file is empty or not set"
    exit 1
else
    time python3 $pythonfile -g $gpu_id -f $data_file
fi

# End time of calculation
end=$( date "+%s" )
# Calculate the calculation time
duration=$(( end - start ))
# Print time
DAYS=$(( duration / 86400 ))
HOURS=$(( (duration % 86400) / 3600 ))
MINS=$(( ((duration % 86400) % 3600) / 60 ))
SECS=$(( ((duration % 86400) % 3600) % 60 ))
echo "Time taken: $DAYS days, $HOURS hours, $MINS minutes and $SECS seconds."
