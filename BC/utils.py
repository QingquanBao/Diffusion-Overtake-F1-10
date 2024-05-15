import wandb
import torch


class Logger(object):
    def __init__(self, config, log_dir):
        self.config = config

        exp_name = log_dir.parts[-1]
        self.wandb_writer = wandb.init(project='diffusers', config=dict(config), name=exp_name)

    def log(self, key, value, step):
        if type(value) == torch.Tensor:
            value = value.item()
        self.wandb_writer.log({key: value}, step=step)

    def finish(self):
        self.run.finish()