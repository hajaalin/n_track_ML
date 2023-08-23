#!/bin/bash
#

#SBATCH --job-name={{ job_name }}
#SBATCH -M {{ cluster }}
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000
#SBATCH -p {{ partition }}
#SBATCH --gres=gpu:1

# Time limit
#SBATCH -t {{ time }}

#SBATCH --chdir={{ job_dir }}
#SBATCH --output={{ job_name }}-%j.out


# Load the Conda module
module use /proj/group/lmu/envs_and_modules/slurm_modules/
module --ignore-cache load Miniconda3/4.12.0_py38
source activate tsc
conda env list
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
