import torch
from torch import nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm


from networks import ConditionalUnet1D
from dataset import get_Diffusion_dataloader
from utils import Logger

import hydra
from pathlib import Path


class Workspace:
    def __init__(self, config):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.obs_horizon = config.obs_horizon
        self.pred_horizon = config.pred_horizon
        self.num_diffusion_iters = config.num_diffusion_iters
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        self.cfg = config
        self._global_step = 0

        self.logger = Logger(config=config, log_dir=self.work_dir)
        self.setup()


    def setup(self):
        # observation and action dimensions corrsponding to
        # the output of Driving

        # create network object
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon
        )

        # create a dataloader
        self.dataloader = get_Diffusion_dataloader(
                            folder=self.cfg.data_folder,
                            obs_horizon=self.obs_horizon,
                            action_horizon=self.pred_horizon,
                            batch_size=self.batch_size,
                            file_num=-1
                            )

        # example inputs
        noised_action = torch.randn((1, self.pred_horizon, self.action_dim))
        obs = torch.zeros((1, self.obs_horizon, self.obs_dim))
        diffusion_iter = torch.zeros((1,))

        # the noise prediction network
        # takes noisy action, diffusion iteration and observation as input
        # predicts the noise added to action
        noise = self.noise_pred_net(
            sample=noised_action,
            timestep=diffusion_iter,
            global_cond=obs.flatten(start_dim=1))

        # illustration of removing noise
        # the actual noise removal is performed by NoiseScheduler
        # and is dependent on the diffusion noise schedule
        denoised_action = noised_action - noise

        # for this demo, we use DDPMScheduler with 100 diffusion iterations
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

        # self.device transfer
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        _ = self.noise_pred_net.to(self.device)


        # Exponential Moving Average
        # accelerates training and improves stability
        # holds a copy of the model weights
        self.ema = EMAModel(
                            parameters=self.noise_pred_net.parameters(),
                            power=0.75)

        # Standard ADAM optimizer
        # Note that EMA parametesr are not optimized
        self.optimizer = torch.optim.AdamW(
                                            params=self.noise_pred_net.parameters(),
                                            lr=1e-4, weight_decay=1e-6)

        # Cosine LR schedule with linear warmup
        self.lr_scheduler = get_scheduler(
                                            name='cosine',
                                            optimizer=self.optimizer,
                                            num_warmup_steps=500,
                                            num_training_steps=len(self.dataloader) * self.num_epochs
                                            )

    def train_step(self, nbatch):
        nobs = nbatch['obs'].to(self.device)
        naction = nbatch['action'].to(self.device)
        B = nobs.shape[0]

        # observation as FiLM conditioning
        # (B, obs_horizon, obs_dim)
        obs_cond = nobs[:,:self.obs_horizon,:]
        # (B, obs_horizon * obs_dim)
        obs_cond = obs_cond.flatten(start_dim=1)

        # sample noise to add to actions
        noise = torch.randn(naction.shape, device=self.device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()

        # add noise to the clean images according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_actions = self.noise_scheduler.add_noise(
            naction, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond)

        # L2 loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        # optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # step lr scheduler every batch
        # this is different from standard pytorch behavior
        self.lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        self.ema.step(self.noise_pred_net.parameters())

        # logging
        loss_cpu = loss.item()
        return loss_cpu

    def train(self ):
        with tqdm(range(self.num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(self.dataloader, desc='Batch', leave=False) as tepoch:
                    for nbatch in tepoch:
                        # data normalized in dataset
                        # self.device transfer
                        loss_cpu = self.train_step(nbatch)
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)

                        if self._global_step % self.cfg.log_interval == 0:
                            self.logger.log('train/loss', loss_cpu, self._global_step)
                        self._global_step += 1

                # save model every k epochs
                if epoch_idx % self.cfg.save_checkpoint_interval == 0:
                    torch.save(self.noise_pred_net.state_dict(), f'noise_pred_net_{epoch_idx}.pth')

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    #from train_dynamics import Workspace as W
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    workspace.train()

if __name__ == '__main__':
    main()