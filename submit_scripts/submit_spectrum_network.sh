# Submit the python script to the queue and gives it the name of the current folder in the queue
# Use the -s flag to keep this process running until it finishes and sync training data to wandb

python_script="$(dirname "$(dirname "$0")")/nn_scripts/nn_spectrum.py"

queue_script="$(dirname "$0")/qpython_forces_spectrum.sh"

data_file="$1"

config_file="$2"

print_usage() {
  echo "Usage: 'submit_python_file.sh' to run without wandb or 'submit_python_file.sh -s' to sync to wandb"
}

# Ensure at least one argument is provided
if [ "$#" -lt 1 ]; then
  echo "Error: data_file argument is required."
  print_usage
  exit 1
fi

# Check if the provided file exists and is not empty
if [ ! -s "$data_file" ]; then
  echo "ERROR: The provided data file '$data_file' is empty or does not exist."
  exit 1
fi

# Print the name of the data file
echo "Using Input file: $data_file"

# Check if the provided config file exists
if [ ! -z "$config_file" ] && [ ! -s "$config_file" ]; then
  echo "The provided config file '$config_file' is empty or does not exist."
else
  echo "Using Config file: $config_file"
fi

sync=false
while getopts ':s' flag; do
  case $flag in
    s) sync=true ;;
    \?)
      echo "Invalid option: -$OPTARG"
      print_usage
      exit 1;;
    :)
      echo "Option -$OPTARG requires an argument."
      print_usage
      exit 1;;
    *) print_usage
       exit 1 ;;
  esac
done

if [ -z "$python_script" ]
then
  echo "ERROR: Python file not specified."
  print_usage
  exit 1
fi

if [ -f train.err ]
then rm train.err
fi

if [ -f train.out ]
then rm train.out
fi

name=`basename $PWD`

# Submit the job
if [ -z "$config_file" ]; then
    job_id=$(qsub -terse -N $name $queue_script $python_script $data_file)
else
    job_id=$(qsub -terse -N $name $queue_script $python_script $data_file $config_file)
fi

echo "Submitted job $job_id to queue as $name"

echo `date`" $PWD" >> /data/$USER/checklist.txt

# for wandb sync
if $sync
then
    nohup sync_wandb.sh $job_id &
fi
