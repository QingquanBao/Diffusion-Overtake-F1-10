import numpy as np
from f1tenth_rl_obs.utils.traj_utils import get_actuation_PD, get_wp_xyv_with_interp


class AdaptivePurePursuit:
    def __init__(self):
        self.minL = 0.5
        self.maxL = 2.0
        self.minP = 0.3
        self.maxP = 0.6
        self.Pscale = 8.0
        self.Lscale = 8.0
        self.D = 0.1
        self.interpScale = 10
        self.prev_error = 0.0
        self.wheelbase = 0.31


    def plan(self, pose_x, pose_y, pose_theta, curr_v, waypoints):
        """
        Args:
            pose_x:
            pose_y:
            pose_theta:
            curr_v:
            waypoints: [x, y, v, psi, kappa]

        Returns:

        """
        # get L, P with speed
        L = curr_v * (self.maxL - self.minL) / self.Lscale + self.minL
        P = self.maxP - curr_v * (self.maxP - self.minP) / self.Pscale

        position = np.array([pose_x, pose_y])
        lookahead_point, new_L = get_wp_xyv_with_interp(L, position, pose_theta, waypoints, waypoints.shape[0], self.interpScale)

        speed, steering, error = \
            get_actuation_PD(pose_theta, lookahead_point, position, new_L, self.wheelbase, self.prev_error, P, self.D)
        self.prev_error = error
        return steering, speed



