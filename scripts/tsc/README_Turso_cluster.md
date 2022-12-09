# Time-series classification on Turso cluster

## Modules
Conda is installed in ```/proj/group/lmu/software/Miniconda3/py38_4.12.0/``` and can be added to $PATH via a SLURM module (see https://github.com/UH-LMU/Ukko2-settings). The module is loaded in sbatch scripts.

```bash
module use /proj/group/lmu/envs_and_modules/slurm_modules/
module --ignore-cache load Miniconda3/4.12.0_py38
```

## Python environments

```bash
# install Mamba
conda install -n base conda-forge::mamba
```


All-in-one env in LMU project folder:
```
mamba create -n tsc
conda activate tsc
CONDA_CUDA_OVERRIDE="11.2" mamba install cudatoolkit==11.2 jinja2 jupyterlab keras matplotlib numpy pandas pip pyyaml scikit-learn scipy==1.4.1 shap==0.41.0 sktime==0.10.1 tensorflow==2.7.0 -c anaconda -c conda-forge
pip install scikeras
pip install click
python -m ipykernel install --user --name tsc --display-name "Python (tsc)"

```

## Workflow

### Create sbatch scripts
```bash
source ~/.bashrc
conda activate tsc

python create_sbatch.py --job_name le_norm2_k20_f1 --loop_epochs 2 50 2 --options "--fset f_mot --kernel_size=20 --repeats=30" --sbatch_dir sbatch/le_norm2_k20_step2b --paths /proj/hajaalin/Projects/n_track_ML/scripts/tsc/paths.yml

```

### Submit sbatch scripts
```bash
# repeat until no python environment is active
# (an active environment will mess up environment variables sent with sbatch)
conda deactivate 

for s in $(ls sbatch/le_norm2_k20_step2b/*.sh); do sbatch $s; done;

```

