import time
import numpy as np
import os
import matplotlib.pyplot as plt
import gymnasium as gym

import torch
from MLP import MLPModel, MLPModel2, BMLPModel
import imageio

os.environ['F110GYM_PLOT_SCALE'] = '70'

def save_video_imageio(frames,filename='output.mp4',  framerate=30):
    writer = imageio.get_writer(filename, fps=framerate,format='mp4', codec='libx264')
    
    for frame in frames:
        writer.append_data(frame)  # Append each frame to the video file
    #writer.append_data(frames)
    
    writer.close()

def env_creator(env_config=None):
    # Initialize and return an instance of your environment
    env_config = {
        "seed": 63,
        #"map": "skan",
        #"map": "levine_block",
       "map": "example_map",
        "map_ext": ".png",
        "model": "dynamic_ST",
        "num_agents": 1,
        "drive_control_mode": "vel",
        "steering_control_mode": "angle",
        # Added for customized environment
        "max_episode_steps": 1000,
        "scan_beams": 1081,
        #"scan_beams": 271,
        #"scan_beams": 361,
        #"scan_beams": 55,
        "timestep": 0.01,
        "stack_action_frames": 2,
        # Used by the adapter
        "frenet_s_density": 0.05,
        "track_width": 0.8,
        "progress_window_size": 10,
        "verbose": False,
        "reward": {
            "collision_punish": 5.0,
            "reverse_direction_punish": -1.0,
            "time_reward": 0.04,
            "progress_reward": 1.0,
            "yawrate_punish": 0.05,
            "milestone_distance": 8.0,
            "milestone_speed_threshold": 2.0,  # 1/2 of possible max speed?
            "milestone_reward": 1.0,
        }
    }
    env = gym.make("f1tenth_rl_obs:f110obs-v0", env_config=env_config)
    return env
    # return F110EnvObs(env_config)


def test_env_render():
    env = env_creator()
    obs, info = env.reset()
    done = False
    trunc = False
    epi_reward = 0.0
    renders = []
    while not done and not trunc:
        # action = np.array([0.0, 0.0])
        # action = np.array([1.0, 0.1, 0.05, 8.0])
        action = env.pure_pursuit_action()
        observation, reward, done, trunc, info = env.step(action)
        epi_reward += reward
        render_img = env.render('rgb_array')
        if render_img is not None:
            renders.append(render_img)

    renders = np.array(renders)
    save_video_imageio(renders, 'output.mp4', 60)
    print(f"Episode reward: {epi_reward}")


def collect_data():
    obs_list = []
    action_list = []

    for i in range(50):
        print(f"Episode {i}")
        env = env_creator()
        obs, info = env.reset()
        done = False
        trunc = False

        while not done and not trunc:
            action = env.pure_pursuit_action()
            action[0] += np.random.normal(0, 0.15)
            action[1] += np.random.normal(0, 1.0)

            obs_list.append(obs)
            action_list.append(action)
            obs, reward, done, trunc, info = env.step(action)

    obs_list = np.array(obs_list)
    action_list = np.array(action_list)
    np.save("obs.npy", obs_list)
    np.save("action.npy", action_list)


def test_algo(output_file):
    env = env_creator()
    observation, info = env.reset()
    done = False
    trunc = False
    epi_reward = 0.0

    #model = MLPModel(1081, 256, 128, 2)
    pth = '/Users/mac/Desktop/PENN/f1tenth_rl_obs/BC/pp_1081_mlp_model_50.pth'
    #model.load_state_dict(torch.load(pth))
    #model = MLPModel2(1081, 2)
    model = MLPModel(1081, 256, 128, 2)
    model = torch.load(pth)
    model.eval()

    renders = []

    start_time = time.time()

    while not done and not trunc:
        with torch.no_grad():
            #action = model(torch.tensor(observation)[::20].float() / 10).detach().squeeze().numpy()
            #action = model(torch.tensor(observation)[::8].float() / 10).detach().squeeze().numpy()
            #action = model(torch.tensor(observation)[::4].float() / 10).detach().squeeze().numpy()
            #action = model(torch.tensor(observation)[::3].float() / 10).detach().squeeze().numpy()
            action = model(torch.tensor(observation).float() / 10).detach().squeeze().numpy()
            
            action = action * np.array([0.192 * 1 , 10])

            print(action)

            if env.env_config["num_agents"] == 2:
                action2 = env.pure_pursuit_action(1, v=2.0)
                action = np.stack([action, action2])
            else:
                action = action.reshape(1, 2)

            #action = np.flip(action)

        observation, reward, done, trunc, info = env.step(action)
        epi_reward += reward
        #env.render()
        render_img = env.render('rgb_array')
        if render_img is not None:
            renders.append(render_img)
    print(f"Episode reward: {epi_reward}")
    print(f"Time taken: {time.time() - start_time}")
    print(f'fps: {len(renders) / (time.time() - start_time)}')

    renders = np.array(renders)
    save_video_imageio(renders, output_file, 60)
    
def Btest_algo(output_file):
    env = env_creator()
    observation, info = env.reset()
    done = False
    trunc = False
    epi_reward = 0.0

    #model = MLPModel(1081, 256, 128, 2)
    pth = '/Users/mac/Desktop/PENN/f1tenth_rl_obs/BC/mlp_model1.pth'
    #model = MLPModel2(1081, 2)
    #model = torch.load(pth, map_location=torch.device('cpu'))
    model = BMLPModel(1081, 256, 128, 2*16)
    model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
    model.eval()

    renders = []

    while not done and not trunc:
        with torch.no_grad():
            #action = model(torch.tensor(observation)[::20].float() / 10).detach().squeeze().numpy()
            #action = model(torch.tensor(observation)[::8].float() / 10).detach().squeeze().numpy()
            #action = model(torch.tensor(observation)[::4].float() / 10).detach().squeeze().numpy()
            #action = model(torch.tensor(observation)[::3].float() / 10).detach().squeeze().numpy()
            actions = model(torch.tensor(observation).float() / 10).detach().squeeze().numpy()

            actions = actions.reshape(16, 2)
            
        for i in range(0, 8):
            action = actions[i]
            action = action * np.array([0.192 * 1 , 6])

            print(action)

            if env.env_config["num_agents"] == 2:
                action2 = env.pure_pursuit_action(1, v=2.0)
                action = np.stack([action, action2])

            #action = np.flip(action)

            observation, reward, done, trunc, info = env.step(action)
            epi_reward += reward
            #env.render()
            render_img = env.render('rgb_array')
            if render_img is not None:
                renders.append(render_img)

    print(f"Episode reward: {epi_reward}")

    renders = np.array(renders)
    save_video_imageio(renders, output_file, 60)

def test_diffusion(act_horizon, output_file):
    env = env_creator()
    observation, info = env.reset()
    done = False
    trunc = False
    epi_reward = 0.0

    from evaluate import DiffusionPredictor
    import yaml
    from types import SimpleNamespace
    from collections import deque

    pth_dir = '/Users/mac/Desktop/PENN/f1tenth_rl_obs/BC/diffusion_ckpt'
    #pth_dir = os.path.join(pth_dir, '16-47-12')
    pth_dir = os.path.join(pth_dir, '01-02-02')

    with open(os.path.join(pth_dir, '.hydra/config.yaml'), 'r') as f:
        configs = yaml.safe_load(f)
        cfg = SimpleNamespace(**configs)

    predictor = DiffusionPredictor(cfg)
    predictor.load_checkpoint(os.path.join(pth_dir, 'noise_pred_net_50.pth')) 

    obs_deque = deque(maxlen=cfg.obs_horizon)

    for i in range(cfg.obs_horizon):
        obs_deque.append(observation)

    renders = []
    start_time = time.time()

    while not done and not trunc:

        actions = predictor.evaluate(obs_deque)

        for i in range(0, act_horizon):
            action = actions[i]

            print(action)

            if env.env_config["num_agents"] == 2:
                action2 = env.pure_pursuit_action(1, v=2.0)
                action = np.stack([action, action2])
            else:
                action = action.reshape(1, 2)

            action = action * np.array([1., 0.5])

            observation, reward, done, trunc, info = env.step(action)
            epi_reward += reward
            #env.render()
            render_img = env.render('rgb_array')
            if render_img is not None:
                renders.append(render_img)

            obs_deque.append(observation)

    print(f"Episode reward: {epi_reward}")
    print(f"Time taken: {time.time() - start_time}")
    print(f'fps: {len(renders) / (time.time() - start_time)}')
    renders = np.array(renders)
    save_video_imageio(renders, output_file, 60)


if __name__ == "__main__":
    #test_env_render()
    #test_algo()
    #collect_data()
    #for act_horizon in [4]:
    #    for i in range(1):
    #        output_file = f"diffiter30train_diffusion_overtake_long{act_horizon}_{i}.mp4"
    #        test_diffusion(act_horizon, output_file)
    #        time.sleep(1)

    for i in range(1):
        output_file = f"ood_mlp1081_{i}.mp4"
        test_algo(output_file)
        time.sleep(1)
    #test_diffusion()
