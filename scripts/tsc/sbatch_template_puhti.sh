#!/bin/bash
#

#SBATCH --account=hy7004
#SBATCH --job-name={{ job_name }}
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1

# Time limit
#SBATCH -t {{ time }}

#SBATCH --chdir={{ job_dir }}
#SBATCH --output={{ job_name }}-%j.out


# Load the Tykky Conda module
source /users/hajaalin/activate_tsc.sh
which python

echo "Starting..."
date

PROG="{{ prog_dir }}/{{ prog }}"
PATHS="{{ paths }}"
OPTIONS="{{ options }} --job_name={{ job_name }} --job_id=$SLURM_JOBID"

cmd="srun python ${PROG} --paths ${PATHS} ${OPTIONS}"
echo ${cmd}
${cmd}

echo "Done."
date
