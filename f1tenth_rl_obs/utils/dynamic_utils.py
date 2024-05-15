"""
Reference:
    - TUM Common Road: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models
    - FozaETH racing stack:
    - f1tenth_gym:

"""

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import math

######################### CONTROL INPUT: sv(steer velocity), accl(longitudinal acceleration) #########################

@njit(cache=True)
def steering_constraint(steering_angle, steering_velocity, st_min, st_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            st_min (float): minimum steering angle
            st_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    steering_velocity = max(sv_min, min(sv_max, steering_velocity))
    if (steering_angle <= st_min and steering_velocity <= 0) or (steering_angle >= st_max and steering_velocity >= 0):
        steering_velocity = 0.0

    return steering_velocity


@njit(cache=True)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    if vel > v_switch:
        pos_limit = a_max*v_switch/vel
    else:
        pos_limit = a_max

    accl = max(-a_max, min(pos_limit, accl))
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.0
    return accl

@njit(cache=True)
def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Kinematic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = np.array([x[3]*np.cos(x[4]),
         x[3]*np.sin(x[4]),
         u[0],
         u[1],
         x[3]/lwb*np.tan(x[2])])
    return f


@njit(cache=True)
def vehicle_dynamics_st(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """
    Single Track Dynamic Vehicle Dynamics, with linear tire model

        Args:
            x (numpy.ndarray (7, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: steering angle of front wheels
                x4: velocity in x direction
                x5: yaw angle
                x6: yaw rate
                x7: slip angle at vehicle center
            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])

    # switch to kinematic model for small velocities
    if abs(x[3]) < 1.0:
        # wheelbase
        lwb = lf + lr

        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        0])))

    else:
        # system dynamics
        f = np.array([x[3]*np.cos(x[6] + x[4]),
            x[3]*np.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
                +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
                +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
                -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
                +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

    return f


def kinematic_st_steer(x, u, p):
    """
    Single Track Kinematic Vehicle Dynamics.
        Args:
            x (numpy.ndarray (5, )): vehicle state vector
                x0: x position in global coordinates
                x1: y position in global coordinates
                x2: steering angle of front wheels
                x3: yaw angle
                x4: velocity in x direction

            u (numpy.ndarray (2, )): control input vector
                u0: steering angle
                u1: longitudinal acceleration
        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """
    # clip sv and accl
    u[0] = max(p["sv_min"], min(p["sv_max"], u[0]))
    u[1] = accl_constraints(x[4], u[1], p["v_switch"], p["a_max"], p["v_min"], p["v_max"])

    lf = p["lf"]
    lr = p["lr"]
    lwb = lf + lr
    steer_delay_time = p["steer_delay_time"]
    f = np.array([x[4]*np.cos(x[3]),
         x[4]*np.sin(x[3]),
         (u[0] - x[2]) / steer_delay_time,  # steer_delay_time = 0.2 in MAP
         x[4] / lwb * np.tan(x[2]),
         u[1]])
    return f


def dynamic_st_steer(x, u, p, type):
    """
    Args:
        x (numpy.ndarray (5, )): vehicle state vector
            x0: x-position in a global coordinate system
            x1: y-position in a global coordinate system
            x2: steering angle of front wheels
            x3: yaw angle
            x4: velocity in x-direction
            x5: velocity in y direction
            x6: yaw rate

        u: (numpy.ndarray (2, )) control input vector
            u0: steering angle
            u1: longitudinal acceleration
        p:
        type:

    Returns:

    """
    # clip sv and accl
    u[0] = max(p["sv_min"], min(p["sv_max"], u[0]))
    u[1] = accl_constraints(x[4], u[1], p["v_switch"], p["a_max"], p["v_min"], p["v_max"])

    lf = p["lf"]
    lr = p["lr"]
    h = p["h"]
    m = p["m"]
    I = p["I"]
    steer_delay_time = p["steer_delay_time"]
    lwb = lf + lr

    ## In low speed, switch to kinematic model
    if abs(x[4]) < 1.0:
        f_ks = kinematic_st_steer(x[:5], u, p)
        # u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0]
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[4]/(lwb*np.cos(x[2])**2)*u[0], 0])))
        return f

    # set gravity constant
    g = 9.81  # [m/s^2]

    # create equivalent bicycle parameters
    mu = p["mu"]

    if type == "pacejka":
        B_f = p["B_f"]
        C_f = p["C_f"]
        D_f = p["D_f"]
        E_f = p["E_f"]
        B_r = p["B_r"]
        C_r = p["C_r"]
        D_r = p["D_r"]
        E_r = p["E_r"]
    elif type == "linear":
        C_Sf = p["C_Sf"]  # -p.tire.p_ky1/p.tire.p_dy1
        C_Sr = p["C_Sr"]  # -p.tire.p_ky1/p.tire.p_dy1


    # compute lateral tire slip angles
    alpha_f = -math.atan((x[5] + x[6] * lf) / x[4]) + x[2]
    alpha_r = -math.atan((x[5] - x[6] * lr) / x[4])

    # compute vertical tire forces
    F_zf = m * (-u[1] * h + g * lr) / (lr + lf)
    F_zr = m * (u[1] * h + g * lf) / (lr + lf)
    F_yf = F_yr = 0

    # combined slip lateral forces
    if type == "pacejka":
        F_yf = mu * F_zf * D_f * math.sin(
            C_f * math.atan(B_f * alpha_f - E_f * (B_f * alpha_f - math.atan(B_f * alpha_f))))
        F_yr = mu * F_zr * D_r * math.sin(
            C_r * math.atan(B_r * alpha_r - E_r * (B_r * alpha_r - math.atan(B_r * alpha_r))))
    elif type == "linear":
        F_yf = mu * F_zf * C_Sf * alpha_f
        F_yr = mu * F_zr * C_Sr * alpha_r

    f = [x[4] * math.cos(x[3]) - x[5] * math.sin(x[3]),
         x[4] * math.sin(x[3]) + x[5] * math.cos(x[3]),
         (u[0] - x[2]) / steer_delay_time,  # steer_delay_time = 0.2 in MAP
         x[6],
         u[1],
         1 / m * (F_yr + F_yf) - x[4] * x[6],
         1 / I * (-lr * F_yr + lf * F_yf)]
    return f


class VehicleDynamicModel:
    def __init__(self, model: str, params: dict):
        self.model = model
        self.params = params
        self.tire_model = self.params["tire_model"]

    def dynamics(self, t, s, u):
        # This wrapper adapts the vehicle dynamics to the solve_ivp format
        # We assume 'u' is constant over each interval `dt`
        if self.model == "kinematic_st_steer":
            return kinematic_st_steer(s, u, self.params)
        if self.model == "dynamic_st_steer":
            return dynamic_st_steer(s, u, self.params, self.tire_model)

    def forward_trajectory(self, s0: np.ndarray, u_list: np.ndarray, dt):
        # NOTE: self-implementation Explicit Euler or RK4 are not stable for small timestep.
        s_list = [s0]
        t_span = [0, dt]
        for u in u_list:
            # Using solve_ivp with method 'RK45'
            sol = solve_ivp(self.dynamics, t_span, s_list[-1], args=(u,), method='RK45', rtol=1e-6, atol=1e-9)
            if sol.success:
                s_list.append(sol.y[:, -1])
            else:
                raise RuntimeError("Integration failed")
        return s_list


# DEPRECATED
# return vehicle_dynamics_st(
#     s, u, self.params['mu'], self.params['C_Sf'], self.params['C_Sr'],
#     self.params['lf'], self.params['lr'], self.params['h'], self.params['m'],
#     self.params['I'], self.params['s_min'], self.params['s_max'],
#     self.params['sv_min'], self.params['sv_max'], self.params['v_switch'],
#     self.params['a_max'], self.params['v_min'], self.params['v_max'])