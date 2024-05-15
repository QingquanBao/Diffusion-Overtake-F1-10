import torch
from torch import nn
import numpy as np

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler


from networks import ConditionalUnet1D

from pathlib import Path


class DiffusionPredictor:
    def __init__(self, config):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.obs_horizon = config.obs_horizon
        self.pred_horizon = config.pred_horizon
        self.action_horizon = config.pred_horizon
        self.num_diffusion_iters = config.num_diffusion_iters
        #self.num_diffusion_iters = 140
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        self.cfg = config
        self._global_step = 0

        self.setup()

    def setup(self):
        # observation and action dimensions corrsponding to
        # the output of Driving

        # create network object
        #self.noise_pred_net = ConditionalUnet1D(
        #    input_dim=self.action_dim,
        #    global_cond_dim=self.obs_dim*self.obs_horizon
        #)

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon,
            diffusion_step_embed_dim=64,
            down_dims=[64,64,64]
            #diffusion_step_embed_dim=128,
            #down_dims=[64,128,128]
        )

        # example inputs
        noised_action = torch.randn((1, self.pred_horizon, self.action_dim))
        obs = torch.zeros((1, self.obs_horizon, self.obs_dim))
        diffusion_iter = torch.zeros((1,))

        # initialize scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )


    def load_checkpoint(self, checkpoint):
        self.noise_pred_net.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.noise_pred_net.eval()

    def evaluate(self, obs_deque, device='cpu'):
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)

        # normalize observation
        nobs = obs_seq / 10.0

        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = naction * np.array([0.192, 10])[None, :]

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        return action

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, done, _, info = env.step(action[i])
            # save observations
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            imgs.append(env.render(mode='rgb_array'))

            # update progress bar
            step_idx += 1