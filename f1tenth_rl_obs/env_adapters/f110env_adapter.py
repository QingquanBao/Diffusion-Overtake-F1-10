from f110_gym.envs import F110Env
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.envs.registration import EnvSpec
from f1tenth_rl_obs.utils.traj_utils import generate_clothoid_traj, ContinuousFrenetSpline
from f1tenth_rl_obs.utils.utils import WORKDIR
from f1tenth_rl_obs.env_adapters.planners.pure_pursuit import AdaptivePurePursuit
import os


class F110EnvObs(gym.Env):
    def __init__(self, env_config=None):
        assert env_config is not None, "Please provide an environment configuration."
        self.env_config = env_config
        self.reward_config = env_config["reward"]
        self.action_space = Box(low=np.array([-0.4, 0.0]), high=np.array([0.4, 6.0]), dtype=np.float32)
        #self.observation_space = Box(low=0.0, high=30.0, shape=(54,),
        #                             dtype=np.float32)  # Example for continuous observation

        # Extended observation space: LiDAR (54) + Velocity (1) + Steering Angle (1)

        if env_config.get("add_state_to_obs", False):
            obs_array_low = np.array( [-np.inf] * 7 + [0.0]*env_config['scan_beams'])
            obs_array_high = np.array([np.inf] * 7 + [30.0]*env_config['scan_beams'] )
        else:
            obs_array_low = np.array([0.0]*env_config['scan_beams'])
            obs_array_high = np.array([30.0]*env_config['scan_beams'])

        self.observation_space = Box(low=obs_array_low, high=obs_array_high, dtype=np.float32)

        # self.spec = EnvSpec(id="63")
        # self.spec.max_episode_steps = env_config.get("max_episode_steps", 3000)
        self.f110 = F110Env(env_config)
        self.centerline = np.genfromtxt(
            os.path.join(WORKDIR, "maps", env_config["map"], env_config["map"] + "_centerline.csv"), delimiter=",",
            skip_header=0)
        # self.frenetSpline = DiscreteFrenetSpline(self.centerline[:, 0], self.centerline[:, 1], env_config["frenet_s_density"])
        self.frenetSpline = ContinuousFrenetSpline(self.centerline[:, 0], self.centerline[:, 1],
                                                   env_config["frenet_s_density"])
        self.track_width = env_config["track_width"]
        self.stack_action_frames = env_config.get("stack_action_frames", 2)
        self.timestep = env_config["timestep"]
        self.verbose = env_config.get("verbose", False)
        # [s, x, y, psi, v, yaw_rate]
        self.curr_odom = None
        self.last_odom = None
        # self.s_window = PushOnlyCircularQueue(env_config["progress_window_size"])
        self.last_action = [0.0, 0.0]
        self.lap_count = 0
        self.step_count = 0
        self.planner = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.lap_count = 0
        self.step_count = 0
        done = True
        # Reset until a valid start position is found
        while done:
            start_idx = np.random.randint(0, self.frenetSpline.wp_num - 1)
            self.frenetSpline.roll_frenet_wp(start_idx - 1)
            # DISCRETE FRENET RESET
            # start_s = self.frenetSpline.s[1]
            # start_psi = self.frenetSpline.psi[1] + np.random.uniform(-0.3 * np.pi, 0.3 * np.pi)
            # CONTINOUSE FRENET RESET

            # Support multi agents
            starts = []
            curr_odom = []
            obs = []
            for i in range(self.env_config["num_agents"]):
                start_s = 0.5 + 7.9 * i # a small value for reset
                start_psi = self.frenetSpline.calc_yaw(start_s) + np.random.uniform(-0.2 * np.pi, 0.2 * np.pi)
                start_d = np.random.uniform(-self.track_width, self.track_width) / 2.0
                start_x, start_y = self.frenetSpline.frenet_to_cartesian(start_s, start_d)
                start_s, start_d = self.frenetSpline.cartesian_to_frenet(start_x, start_y)
                starts.append(np.array([start_x, start_y, start_psi]))
                curr_odom.append(np.array([start_s, start_x, start_y, start_psi, 0.0, 0.0]))

            start = np.array(starts)
            self.curr_odom = np.array(curr_odom)
            self.last_odom = np.array(curr_odom)
            raw_obs, reward, done, info = self.f110.reset(start)
            # print(f"Start_x: {start[0]:.4f}, Start_y: {start[1]:.4f}, Start_theta: {start[2]:.4f}")
        obs = np.array([self.obs_adapter(raw_obs, agent_id=i) for i in range(self.env_config["num_agents"])])
        return obs[0], info

    def step(self, action):
        self.step_count += 1
        raw_obs, reward, done, truncated, info = self.f110.step(action)
        i = 1
        while not done and not truncated and i < self.stack_action_frames:
            raw_obs, reward, done, truncated, info = self.f110.step(action)
            i += 1
        info = raw_obs
        # Update state
        obss = []
        rewards = []
        for agent_id in range(self.env_config["num_agents"]):
            raw_obs['abs_yaw_rate'] = abs(raw_obs['ang_vels_z'][agent_id])
            raw_obs['addition_reward'] = 0.0
            raw_obs['steer'] = action[agent_id]
            x, y, psi, v, yaw_rate = raw_obs['poses_x'][agent_id], raw_obs['poses_y'][agent_id], raw_obs['poses_theta'][agent_id], raw_obs['linear_vels_x'][agent_id], raw_obs['abs_yaw_rate']
            s, _ = self.frenetSpline.cartesian_to_frenet(x, y)
            self.curr_odom[agent_id] = np.array([s, x, y, psi, v, yaw_rate])
            # Check for reverse direction.
            if self.frenetSpline.is_reverse_direction(s, psi):
                if self.verbose:
                    print("Reverse direction detected, episode terminated.")
                raw_obs['addition_reward'] = -self.reward_config["reverse_direction_punish"]
                self.curr_odom[agent_id] = self.last_odom[agent_id]  # To avoid large gap in s
                #done = True
            # Check for finishing one lap.
            if self.frenetSpline.is_cross_start_line(s, self.last_odom[agent_id][0]):
                self.lap_count += 1 if agent_id == 0 else 0
                if self.lap_count == 1:
                    done = True
            # Calculate obs and reward
            obss.append(self.obs_adapter(raw_obs, agent_id=agent_id))
            rewards.append(self.reward_adapter(raw_obs, agent_id=agent_id))
        if self.verbose:
            print(f"[STEP] collision: {raw_obs['collisions'][0]}, reward: {reward}, done: {done}, truncated: {truncated}")

        self.last_odom = self.curr_odom
        return obss[0], rewards[0], done, truncated, info

    def close(self):
        self.f110.close()

    def render(self, mode='human'):
        return self.f110.render(mode=mode)

    def obs_adapter(self, raw_obs, agent_id):
        obs = raw_obs["scans"][agent_id].astype(np.float32)
        # TODO: obs seem to have invalid values(<0 or >30), need to be fixed
        obs = np.clip(obs, 0, 30)
        if self.env_config.get("add_state_to_obs", False):
            obs = np.append(raw_obs['vehicle_state'][agent_id], obs)
            #obs =  np.append(obs, [self.curr_odom[4], self.curr_odom[5]])
        return obs

    def reward_adapter(self, raw_obs, agent_id):
        is_ego_collide = 1 if raw_obs["collisions"][agent_id] else 0
        # Assume an episode is done in one lap
        # prog = self.lap_count * self.lap_dist + self.curr_s
        # SURVICE REWARD:
        # reward = 0.1 - 0.5 * is_ego_collide - self.yawrate_punish_scale * raw_obs['abs_yaw_rate'] + raw_obs['addition_reward']
        # DIFF-PROGRESS + Punishment on yaw rate REWARD:
        s_diff = self.curr_odom[agent_id][0] - self.last_odom[agent_id][0]
        if s_diff > 5.0:
            print(
                f"step: {self.step_count}, last_s: {self.last_odom[0]:.4f}, curr_s: {self.curr_odom[0]:.4f}, max_s: {self.frenetSpline.max_s:.4f}")
        # noinspection PyTypeChecker
        reward = (self.reward_config["time_reward"]
                  + max(s_diff, 0.0) * self.reward_config["progress_reward"]
                  - self.reward_config["collision_punish"] * is_ego_collide
                  - self.reward_config["yawrate_punish"] * 10 * abs(raw_obs['steer'])
                  + raw_obs['addition_reward'])
        return reward


    def pure_pursuit_action(self, agent_id=1, v=4.0):
        if not self.planner:
            self.planner = AdaptivePurePursuit()
        s_diff, d, psi_diff, v = 1.0, 0.0, 0.0, v
        # Assume agent id=1 is the baseline agent
        agent_odom = self.curr_odom[agent_id]

        goal_x, goal_y = self.frenetSpline.frenet_to_cartesian(agent_odom[0] + s_diff, d)
        goal_theta = agent_odom[3] + psi_diff
        traj = generate_clothoid_traj(goal_x, goal_y, goal_theta, v, agent_odom[1], agent_odom[2],
                                      agent_odom[3])

        # use generated traj for pure pursuit controller
        steer, speed = self.planner.plan(agent_odom[1], agent_odom[2], agent_odom[3],
                                               agent_odom[4], traj)
        return [steer, speed]

class F110EnvObs_v2(F110EnvObs):
    def __init__(self, env_config=None):
        super().__init__(env_config)
        # (s_diff, d, psi_diff, v)
        self.action_space = Box(low=np.array([1.0, -1.5, -0.4, 1.0]), high=np.array([3.0, 1.5, 0.4, 8.0]),
                                dtype=np.float32)
        self.observation_space = Box(low=-500.0, high=500.0, shape=(58,),
                                     dtype=np.float32)
        # [x, y, psi, v]
        self.curr_odom = None
        # [s, d, psi, v]
        self.curr_odom_frenet = None
        self.pp_controller = AdaptivePurePursuit()

    def reset(self, seed=None, options=None, DEBUG=False):
        super().reset(seed=seed)
        self.lap_count = 0
        done = True
        # Reset until a valid start position is found
        while done:
            start_idx = np.random.randint(0, self.frenetSpline.wp_num - 1)
            self.frenetSpline.roll_frenet_wp(start_idx - 1)
            start_s = self.frenetSpline.s[1]
            start_psi = self.frenetSpline.psi[1] + np.random.uniform(-0.3 * np.pi, 0.3 * np.pi)
            start_d = np.random.uniform(-self.track_width, self.track_width) / 2.0
            start_x, start_y = self.frenetSpline.frenet_to_cartesian(start_s, start_d)
            start = np.array([start_x, start_y, start_psi])
            self.curr_odom = [start_x, start_y, start_psi, 0.0]
            self.curr_s = start_s
            self.last_s = start_s
            raw_obs, reward, done, info = self.f110.reset(np.array([start]))
            # print(f"Start_x: {start[0]:.4f}, Start_y: {start[1]:.4f}, Start_theta: {start[2]:.4f}")
            self.curr_odom = [start_x, start_y, start_psi, 0.0, 0.0]
            self.curr_odom_frenet = [start_s, start_d, start_psi, 0.0]
        obs = self.obs_adapter(raw_obs)
        if DEBUG:
            return start
        return obs, info

    def step(self, action, DEBUG=False):
        s_diff, d, psi_diff, v = action[0], action[1], action[2], action[3]
        goal_x, goal_y = self.frenetSpline.frenet_to_cartesian(self.curr_s + s_diff, d)
        goal_theta = self.curr_odom[2] + psi_diff
        traj = generate_clothoid_traj(goal_x, goal_y, goal_theta, v, self.curr_odom[0], self.curr_odom[1],
                                      self.curr_odom[2])

        # use generated traj for pure pursuit controller
        steer, speed = self.pp_controller.plan(self.curr_odom[0], self.curr_odom[1], self.curr_odom[2],
                                               self.curr_odom[3], traj)
        raw_obs, reward, done, truncated, info = self.f110.step(np.array([np.array([steer, speed])]))
        abs_yaw_rate = abs(raw_obs['ang_vels_z'][0])
        abs_yaw_rate_count = 1
        action_frame = 1
        while not done and not truncated and action_frame < self.stack_action_frames:
            steer, speed = self.pp_controller.plan(self.curr_odom[0], self.curr_odom[1], self.curr_odom[2],
                                                   self.curr_odom[3], traj)
            raw_obs, reward, done, truncated, info = self.f110.step(np.array([np.array([steer, speed])]))
            abs_yaw_rate += abs(raw_obs['ang_vels_z'][0])
            abs_yaw_rate_count += 1
            action_frame += 1
        info = raw_obs
        # Update state
        raw_obs['abs_yaw_rate'] = abs_yaw_rate / abs_yaw_rate_count
        raw_obs['addition_reward'] = 0.0
        self.curr_odom = [raw_obs['poses_x'][0], raw_obs['poses_y'][0], raw_obs['poses_theta'][0],
                          raw_obs['linear_vels_x'][0], raw_obs['ang_vels_z'][0]]
        curr_s, curr_d = self.frenetSpline.cartesian_to_frenet(self.curr_odom[0], self.curr_odom[1])
        self.curr_s = curr_s
        self.curr_odom_frenet = [curr_s, curr_d, self.curr_odom[2], self.curr_odom[3]]
        # Check for reverse direction.
        if self.frenetSpline.is_reverse_direction(curr_s, self.curr_odom[2]):
            raw_obs['addition_reward'] = -0.5
            done = True
        # Check for finishing one lap.
        if self.frenetSpline.is_cross_start_line(self.curr_s, self.last_s):
            self.lap_count += 1
            raw_obs['addition_reward'] = 0.5
            done = True
        # Calculate obs and reward
        obs = self.obs_adapter(raw_obs)
        reward = self.reward_adapter(raw_obs)
        # print(f"collision: {raw_obs['collisions'][0]}, reward: {reward}, done: {done}, truncated: {truncated}")
        if self.verbose:
            print(f"collision: {raw_obs['collisions'][0]}, reward: {reward}, done: {done}, truncated: {truncated}")
            print(f"last_s: {self.last_odom[0]:.4f}, curr_s: {self.curr_odom[0]:.4f}")
            self.last_s = self.curr_s
            return obs, reward, done, truncated, info
        self.last_s = self.curr_s
        return obs, reward, done, truncated, info

    def obs_adapter(self, raw_obs):
        obs = raw_obs["scans"][0][::2]
        obs = np.hstack((obs, np.array(self.curr_odom_frenet)))
        return obs

# DEBUG
# start_idx = 19
# start[2] += np.random.uniform(-0.3 * np.pi, 0.3 * np.pi)
# DEBUG
# start = np.array([0.53781196, 2.03148632, 1.62655743])

### Deprecated ###
# env_config = {
#     "map": "example_map",
#     "num_agents": 1,
#     "timestep": 0.01,
#     "integrator": "rk4",
#     "control_input": ["speed", "steering_angle"],
#     "model": "st",
#     "observation_config": {"type": "features",
#                            "features": ["scan", "pose_x", "pose_y", "pose_theta", "collision", "ang_vel_z"]},
#     "params": {"mu": 1.0},
#     "reset_config": {"type": "cl_random_static"},
#     "render_mode": "human",
#     # Added for customized environment
#     "max_episode_steps": 500,
#     "scan_beams": 108,
#     # Used by the adapter
#     "frenet_s_density": 0.05,
#     "track_width": 2.0,
#     "progress_window": 10,
# }

# Check for milestone reward
# milestone_cnt = int(self.frenetSpline.max_s / self.reward_config["milestone_distance"])
# self.milestone_s_list = np.linspace(0, self.frenetSpline.max_s, milestone_cnt, endpoint=False)
# self.milestone_steps_list = np.zeros(milestone_cnt)
# self.milestone_pointer = 1
# if self.milestone_s_list[self.milestone_pointer] < s:
#     milestone_steps = self.step_count - self.milestone_steps_list[self.milestone_pointer - 1]
#     average_speed = (s - self.milestone_s_list[self.milestone_pointer - 1]) / (milestone_steps * self.timestep * self.stack_action_frames)
#     self.milestone_steps_list[self.milestone_pointer] = self.step_count
#     self.milestone_pointer += 1
#     raw_obs['addition_reward'] = np.exp(average_speed - self.reward_config["milestone_speed_threshold"]) * self.reward_config["milestone_reward"]
#     print(f"Milestone{self.milestone_pointer - 1} reached with average speed: {average_speed:.4f}, s is {s:.4f}, milestone reward: {raw_obs['addition_reward']:.4f}")
#     if DEBUG:
#         print(f"Milestone{self.milestone_pointer-1} reached with average speed: {average_speed:.4f}, milestone reward: {raw_obs['addition_reward']:.4f}")
# for lap
# milestone_steps = self.step_count - self.milestone_steps_list[self.milestone_pointer - 1]
# average_speed = (s + self.frenetSpline.max_s - self.milestone_s_list[self.milestone_pointer]) / (
#             milestone_steps * self.timestep * self.stack_action_frames)
# raw_obs['addition_reward'] = np.exp(average_speed - self.reward_config["milestone_speed_threshold"]) * \
#                              self.reward_config["milestone_reward"]