## README first

Please Check `/exp_records` for more visualization of overtaking!

![One demo](exp_records/diff/diffusion_overtake_long8_3.mp4)

This repo shows how diffusion policy is used in F1/10 settings. The overtake success rate could be 60% without any opponent-competing expert drivign data, compared with MLP success rate with 30%.

You could check the checkpoints of diffusion policy and our generated data in https://drive.google.com/drive/folders/1NRLhz4Sh3GxZBJBTg8gTesMh8X8I5OL3?usp=sharing


## Acknowledgement

The f1tenth gym wrapper is forked from https://github.com/zzjun725/f1tenth_rl_obs/tree/main. We implement render(mode='rgb_array') into the enviornment for offline visualization, and also support multi-agent racing. We thanks Zhijun for her laying foundation.


The main work lies in `./BC/`, which implements diffusion policy and MLP training. Diffusion Policy networks and training codes are refered from https://github.com/real-stanford/diffusion_policy. Thanks for Chi et. al. great works!!!



## Environment Installation
Currently the environment use python3.8, ray[tune, rllib]==2.8.0, torch==2.0.0, CUDA11.8.

1. Create a conda environment. venv is also fine.

```conda env create -f environment.yml```

2. Install f1tenth_gym

I use my own fork for the f1tenth_gym.

```
git clone -b multibody_test git@github.com:zzjun725/f1tenth_gym.git
cd f1tenth_gym
pip install -e .
```

3. Install ray and pytorch.
```
pip install 'ray[tune, rllib]==2.8.0'
```
Check how to install torch==2.0.0 from the official website. Personally I think it should be fine to use different torch
version, as long as it is compatible with ray.

4. Install f1tenth_rl_obs
This repo is a self-contained package for convenience, so you will need to install it first.
Inside the root directory of this repo(which contains the setup.py), run the following command.
```
pip install -e .
```

5. Possible missing packages(optional, if you get an error)
```
pip install chardet
```
