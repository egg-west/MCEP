# MCEP
source code for our paper: Mildly Constrained Evaluation Policy (MCEP) for Offline Reinforcement Learning

# Prepare to run
```
sudo apt install libx11-dev libglew-dev patchelf

https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
put it in ~/.mujoco

#download anaconda
https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
# execute
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n jaxrl python=3.8
conda activate jaxrl

#install d4rl by clone
git clone https://github.com/Farama-Foundation/D4RL
cd D4RL
pip install -e .
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# put these into you ~\.bashrc, then execute: source ~\.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/v-linjiexu/.mujoco/mujoco210/bin
export PATH=$PATH:/home/v-linjiexu/anaconda3/bin

pip install wandb tqdm ml_collections optax flax distrax

# to test the environment:
import d4rl
import jax
```

# Get the results
```
# BC
python train_offline.py
# IQL
python train_offline_iql.py
# TD3BC
python train_offline_td3bc.py
# AWAC
python train_offline_awac.py
# TD3BC-MCEP
python train_offline_td3bc_mcep.py
# AWAC-MCEP
python train_offline_awac_mcep.py
```

# Hints to run
To speed up the running, consider reducing the frequency of evaluation.
To control the GPU memory usage, consider `XLA_PYTHON_CLIENT_MEM_FRACTION=.33`

# Thanks
This repository was built based on [ikostrikov/jaxrl2](github.com/ikostrikov/jaxrl2)