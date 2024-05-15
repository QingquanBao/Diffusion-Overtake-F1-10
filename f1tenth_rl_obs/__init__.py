import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='f110obs-v0',  # Use the same env name as in the gym.make() call
    entry_point='f1tenth_rl_obs.env_adapters.f110env_adapter:F110EnvObs',
    kwargs={'env_config': None},
)