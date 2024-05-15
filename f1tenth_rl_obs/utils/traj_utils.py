import numpy as np
from pyclothoids import Clothoid
from scipy.interpolate import CubicSpline
from numba import njit
from numba.np.extensions import cross2d
from f1tenth_rl_obs.utils.utils import WORKDIR
import scipy.optimize as so


################### Discrete Frenet Utils #################
def centerline2frenet(input_map, centerline, wp_dist):
    """
    Cubic spline interpolation of the centerline to generate Frenet points and save to a file.
    Args:
        input_map: the name of the map. If none, do not save the Frenet points to a file.
        centerline: [x, y, width_right, width_left]
        wp_dist: the distance between frenet waypoints(used for the discretization after cubic interpolation)
    Returns:
        frenet_points: the Frenet points. [s_m,x_m,y_m,psi_rad,kappa_radpm,w_tr_right_m,w_tr_left_m]
    """
    x = centerline[:, 0]  # (n, )
    y = centerline[:, 1]
    span = centerline[0, 2]  # Assume constant width for the track

    # Ensure the last point is identical to the first point
    x[-1] = x[0]
    y[-1] = y[0]
    # Recompute distances and cumulative distances
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)  # (n-1, )
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)  # (n, )
    cs_x = CubicSpline(cumulative_distances, x, bc_type='periodic')
    cs_y = CubicSpline(cumulative_distances, y, bc_type='periodic')

    wp_num = round((cumulative_distances.max() - 0) / wp_dist) # number of waypoints
    s_dense = np.linspace(0, cumulative_distances.max(), wp_num)  # (wp_num, )
    x_dense = cs_x(s_dense)  # (wp_num, )
    y_dense = cs_y(s_dense)  # (wp_num, )
    psi = np.arctan2(np.gradient(y_dense), np.gradient(x_dense))
    # Unwrap psi to avoid discontinuity because psi belongs to [-pi, pi]
    psi_unwrapped = np.unwrap(psi)
    kappa = np.gradient(psi_unwrapped) / np.gradient(s_dense)
    w_tr_right = np.ones_like(s_dense) * span
    w_tr_left = np.ones_like(s_dense) * span
    # Create an array of the Frenet points
    frenet_points = np.vstack((s_dense, x_dense, y_dense, psi, kappa, w_tr_right, w_tr_left)).T

    # Save the Frenet points to a file. Note the first and last points have the same x, y to ensure continuity.
    if input_map is not None:
        np.savetxt(f'{WORKDIR}/maps/{input_map}_frenet.csv', frenet_points, delimiter=',', fmt='%.8f',
                   header='s_m,x_m,y_m,psi_rad,kappa_radpm,w_tr_right_m,w_tr_left_m')
    return frenet_points

@njit
def find_closest_segment(x, y, traj_x, traj_y, traj_theta):
    """
    Find the closest segment on a trajectory to a given point.
    Assume pass in trajectory has overlapped point(start & end)
    """
    # Cut the overlapped point at the end to avoid edge case
    n = len(traj_x) - 1
    distances = np.sqrt((traj_x[:-1] - x) ** 2 + (traj_y[:-1] - y) ** 2)
    idx_closest = np.argmin(distances)
    head_vector = np.array([np.cos(traj_theta[idx_closest]), np.sin(traj_theta[idx_closest])])
    point_vector = np.array([x - traj_x[idx_closest], y - traj_y[idx_closest]])
    dot_head_point = np.dot(head_vector, point_vector)
    if dot_head_point >= 0:
        idx_last = idx_closest
        idx_next = idx_closest + 1
    else:
        idx_next = idx_closest
        idx_last = n-1 if idx_closest == 0 else idx_closest - 1

    return idx_last, idx_next


@njit
def cartesian_to_frenet(x, y, traj_x, traj_y, traj_s, traj_theta):
    """
    Args:
        pose: [x, y, ...] in Cartesian frame
        traj_x, traj_y, traj_s: Discretized waypoints in Frenet frame
    Returns:
        s, d
    """
    idx_last, idx_next = find_closest_segment(x, y, traj_x, traj_y, traj_theta)
    # Project the point onto the segment for s and d
    segment_vector = np.array([traj_x[idx_next] - traj_x[idx_last], traj_y[idx_next] - traj_y[idx_last]])
    point_vector = np.array([x - traj_x[idx_last], y - traj_y[idx_last]])
    proj_length = np.dot(point_vector, segment_vector) / np.linalg.norm(segment_vector)
    # Use cross product to consider the direction of the point relative to the segment
    # numba use cross2d instead of np.cross()
    # d = np.cross(segment_vector, point_vector) / np.linalg.norm(segment_vector)
    d = cross2d(segment_vector, point_vector) / np.linalg.norm(segment_vector)
    s = traj_s[idx_last] + proj_length
    return s, d

@njit
def frenet_to_cartesian(s, d, traj_x, traj_y, traj_s, traj_theta):
    """
    Convert Frenet coordinates back to Cartesian coordinates.

    Args:
        s: Distance along the path
        d: Lateral offset from the path
        traj_x, traj_y, traj_s: Discretized waypoints in Frenet and Cartesian frames

    Returns:
        x, y: Cartesian coordinates
    """
    # Find the index where `s` should be inserted to keep `traj_s` sorted, assuming `s` goes to the right of duplicates.
    idx = np.searchsorted(traj_s, s, side='right') - 1
    segment_direction = np.array([np.cos(traj_theta[idx]), np.sin(traj_theta[idx])])
    # Calculate projection length
    proj_length = s - traj_s[idx]
    # Calculate the point's position along the segment
    point_on_segment = np.array([traj_x[idx], traj_y[idx]]) + segment_direction * proj_length
    # Calculate the perpendicular vector for lateral offset
    perp_vector = np.array([-segment_direction[1], segment_direction[0]])
    # Calculate final position with lateral offset
    final_position = point_on_segment + perp_vector * d
    return final_position[0], final_position[1]

@njit
def is_reverse_direction(s, theta, traj_s, traj_theta):
    closest_idx = np.argmin(np.abs(traj_s - s))
    desired_heading = traj_theta[closest_idx]
    ego_heading = theta
    heading_dot = np.cos(desired_heading) * np.cos(ego_heading) + np.sin(desired_heading) * np.sin(ego_heading)
    if heading_dot < 0:
        return True
    return False

def roll_frenet_wp(frenet_wp, step, total_length):
    """
    centerline: [s, ...]
    Roll the centerline to start from a specific index. Change the s accordingly.
    The passed in centerline should not have overlapped points(the old start point w/ s=0 is removed)
    """
    s_shift = frenet_wp[step, 0]
    roll_frenet_wp = np.roll(frenet_wp, -step, axis=0)
    adjusted_s = roll_frenet_wp[:, 0] - s_shift
    corrected_s = np.where(adjusted_s < 0, adjusted_s + total_length, adjusted_s)
    roll_frenet_wp = np.concatenate([corrected_s[:, None], roll_frenet_wp[:, 1:]], axis=1)
    # Assign a new end point
    new_end_point = roll_frenet_wp[0].copy()
    new_end_point[0] = total_length
    roll_frenet_wp = np.concatenate([roll_frenet_wp, new_end_point[None, :]], axis=0)
    return roll_frenet_wp


class DiscreteFrenetSpline:
    def __init__(self, x: np.ndarray, y: np.ndarray, s_density=0.05):
        """
        Args:
            x:
            y:
        """

        # Assure Periodicity
        if x[0] != x[-1]:
            x = np.append(x, x[0])
        if y[0] != y[-1]:
            y = np.append(y, y[0])
        centerline_s = self._calc_s(x, y)
        self._sx_spline = CubicSpline(centerline_s, x, bc_type='periodic')
        self._sy_spline = CubicSpline(centerline_s, y, bc_type='periodic')

        # Dense point
        self.max_s = centerline_s[-1]
        self.wp_num = round((centerline_s[-1] - 0) / s_density)  # number of waypoints
        self.s = np.linspace(0, centerline_s[-1], self.wp_num)  # (wp_num, )
        self.x = self._sx_spline(self.s)  # (wp_num, )
        self.y = self._sy_spline(self.s)  # (wp_num, )
        self.psi = np.arctan2(np.diff(self.y), np.diff(self.x))  # (wp_num-1, )
        self.psi = np.append(self.psi, self.psi[0])  # (wp_num, )

    def _calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def _reset_frenet_wp(self, frenet_wp):
        self.s = frenet_wp[:, 0]
        self.x = frenet_wp[:, 1]
        self.y = frenet_wp[:, 2]
        self.psi = frenet_wp[:, 3]

    def cartesian_to_frenet(self, x, y):
        s, d = cartesian_to_frenet(x, y, self.x, self.y, self.s, self.psi)
        return s, d

    def frenet_to_cartesian(self, s, d):
        x, y = frenet_to_cartesian(s, d, self.x, self.y, self.s, self.psi)
        return x, y

    def is_reverse_direction(self, s, phi):
        return is_reverse_direction(s, phi, self.s, self.psi)

    def is_cross_start_line(self, curr_s, last_s):
        return last_s-curr_s > self.max_s/2

    def roll_frenet_wp(self, step):
        original_wp = np.vstack((self.s, self.x, self.y, self.psi)).T
        original_wp = original_wp[1:, :]  # Remove the overlapped points
        rolled_wp = roll_frenet_wp(original_wp, step, self.max_s)
        self._reset_frenet_wp(rolled_wp)


################### Discrete Frenet Utils #################


################### Continuous Frenet Utils #################
class ContinuousFrenetSpline:
    def __init__(self, x: np.ndarray, y: np.ndarray, wp_dist=0.05):
        """
        Args:
            x:
            y:
            wp_dist: float. The distance between frenet waypoints(used for the discretization to apply roll_wp).

        """

        # Assure Periodicity
        if x[0] != x[-1] or y[0] != y[-1]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
        self.wp_dist = wp_dist
        self._reset_Spline_from_xy(x, y, wp_dist)

    def _reset_Spline_from_xy(self, x, y, wp_dist=None):
        self.s = self._calc_s(x, y)
        self.sx = CubicSpline(self.s, x, bc_type='periodic')
        self.sy = CubicSpline(self.s, y, bc_type='periodic')
        self.sx_d1 = self.sx.derivative(1)
        self.sy_d1 = self.sy.derivative(1)
        if wp_dist is not None:
            s_dense, x_dense, y_dense, _ = self.get_dense_frenet_points(wp_dist)
            self.x = x_dense
            self.y = y_dense
            self.s = s_dense
            self.wp_num = len(s_dense)
        else:
            self.x = x
            self.y = y
            self.wp_num = len(x)
        self.max_s = self.s[-1]
        self.psi = np.arctan2(np.diff(self.y), np.diff(self.x))  # (wp_num-1, )
        self.psi = np.append(self.psi, self.psi[0])  # (wp_num, )

    def _calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def _calc_arc_length(self, x, y):
        """

        Args:
            x:
            y:

        Returns: s, abs(d)

        """
        distances = np.sqrt((self.x[:-1] - x) ** 2 + (self.y[:-1] - y) ** 2)
        idx_closest = np.argmin(distances)
        s_guess = self.s[idx_closest]
        def distance_to_spline(s):
            x_eval = self.sx(s)
            y_eval = self.sy(s)
            return np.sqrt((x - x_eval) ** 2 + (y - y_eval) ** 2)
        # output = so.fmin(distance_to_spline, 0.0, full_output=True, disp=False)
        # closest_s = output[0][0]
        # absolute_distance = output[1]
        result = so.minimize_scalar(distance_to_spline, bounds=(s_guess-2*self.wp_dist, s_guess+2*self.wp_dist), method='bounded')
        closest_s = result.x
        absolute_distance = result.fun
        return closest_s, absolute_distance

    def calc_yaw(self, s):
        dx = self.sx_d1(s)
        dy = self.sy_d1(s)
        return np.arctan2(dy, dx)

    def roll_frenet_wp(self, step):
        """
        Args:
            step: roll steps in the frenet waypoints. roll_s = step * wp_dist
        Returns:

        """
        # Only roll x, y here
        roll_x = np.roll(self.x[1:], -step)
        roll_y = np.roll(self.y[1:], -step)
        roll_x = np.append(roll_x, roll_x[0])
        roll_y = np.append(roll_y, roll_y[0])
        self._reset_Spline_from_xy(roll_x, roll_y)

    def get_dense_frenet_points(self, wp_dist):
        s_dense = np.linspace(0, self.s[-1], int(self.s[-1] / wp_dist))
        x_dense = self.sx(s_dense)
        y_dense = self.sy(s_dense)
        psi = np.arctan2(self.sy_d1(s_dense), self.sx_d1(s_dense))
        return s_dense, x_dense, y_dense, psi

    def frenet_to_cartesian(self, s, d):
        x_eval = self.sx(s)
        y_eval = self.sy(s)
        psi = self.calc_yaw(s)

        x = x_eval - d * np.sin(psi)
        y = y_eval + d * np.cos(psi)
        return x, y

    def cartesian_to_frenet(self, x, y):
        s, d = self._calc_arc_length(x, y)
        s = (s+self.max_s) % self.max_s
        x_eval = self.sx(s)
        y_eval = self.sy(s)
        dx_eval = self.sx_d1(s)
        dy_eval = self.sy_d1(s)
        # tangent: np.array([dx_eval, dy_eval])
        normal = np.array([-dy_eval, dx_eval])
        distance_sign = np.sign(np.dot([x-x_eval, y-y_eval], normal))
        d = d * distance_sign
        return s, d

    def is_reverse_direction(self, s, phi):
        desired_psi = self.calc_yaw(s)
        heading_dot = np.cos(desired_psi) * np.cos(phi) + np.sin(desired_psi) * np.sin(phi)
        return heading_dot < 0

    def is_cross_start_line(self, curr_s, last_s):
        return last_s-curr_s > self.max_s/2


################### Pure Pursuit Utils #################
@njit(cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)
    steering_angle = np.arctan(wheelbase / radius)
    return speed, steering_angle


@njit(cache=True)
def get_actuation_PD(pose_theta, lookahead_point, position, lookahead_distance, wheelbase, prev_error, P, D):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    error = 2.0 * waypoint_y / lookahead_distance ** 2
    # radius = 1 / error
    # steering_angle = np.arctan(wheelbase / radius)
    if np.abs(waypoint_y) < 1e-4:
        return speed, 0., error
    steering_angle = P * error + D * (error-prev_error)
    # print(D * (error-prev_error))
    return speed, steering_angle, error


@njit(cache=True)
def simple_norm_axis1(vector):
    return np.sqrt(vector[:, 0]**2 + vector[:, 1]**2)


@njit(cache=True)
def get_wp_xyv_with_interp(L, curr_pos, theta, waypoints, wpNum, interpScale):
    traj_distances = simple_norm_axis1(waypoints[:, :2] - curr_pos)
    traj_distances = np.abs(traj_distances - L)
    sorted_idx = np.argsort(traj_distances)
    # the trajectory has no loop
    segment_begin = min(sorted_idx[0], sorted_idx[1])
    segment_end = max(sorted_idx[0], sorted_idx[1])

    x_array = np.linspace(waypoints[segment_begin, 0], waypoints[segment_end, 0], interpScale)
    y_array = np.linspace(waypoints[segment_begin, 1], waypoints[segment_end, 1], interpScale)
    v_array = np.linspace(waypoints[segment_begin, 2], waypoints[segment_end, 2], interpScale)
    xy_interp = np.vstack((x_array, y_array)).T
    dist_interp = simple_norm_axis1(xy_interp - curr_pos) - L
    i_interp = np.argmin(np.abs(dist_interp))
    target_global = np.array((x_array[i_interp], y_array[i_interp]))
    new_L = np.linalg.norm(curr_pos - target_global)
    return np.array((x_array[i_interp], y_array[i_interp], v_array[i_interp])), new_L


@njit(cache=True)
def get_wp_xyv_with_L(L, curr_pos, theta, waypoints):
    traj_distances = simple_norm_axis1(waypoints[:, :2] - curr_pos)
    traj_distances = np.abs(traj_distances - L)
    i = np.argmin(traj_distances)
    target_global = np.array((waypoints[i, 0], waypoints[i, 1], waypoints[i, 2]))
    return target_global

################### Pure Pursuit Utils #################


def sample_traj(clothoid, npts, v):
    # traj (m, 5)
    traj = np.empty((npts, 5))
    k0 = clothoid.Parameters[3]
    dk = clothoid.Parameters[4]

    for i in range(npts):
        s = i * (clothoid.length / max(npts - 1, 1))
        traj[i, 0] = clothoid.X(s)
        traj[i, 1] = clothoid.Y(s)
        traj[i, 2] = v
        traj[i, 3] = clothoid.Theta(s)
        traj[i, 4] = np.sqrt(clothoid.XDD(s) ** 2 + clothoid.YDD(s) ** 2)
    return traj


def generate_clothoid_traj(goal_x, goal_y, goal_theta, goal_v, pose_x, pose_y, pose_theta):
    """
    Generate a clothoid trajectory from current pose to goal pose
    """
    clothoid = Clothoid.G1Hermite(pose_x, pose_y, pose_theta, goal_x, goal_y, goal_theta)
    traj = sample_traj(clothoid, 20, goal_v)
    return traj