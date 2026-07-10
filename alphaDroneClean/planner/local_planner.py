import numpy as np
from planner.spline import CubicSpline
from typing import List


class LinearLocalPlanner:
    def __init__(self, spline_list: List[CubicSpline], velocity):
        self.closest_u = 0.0
        self.current_curve_i = 0
        self.current_position = None
        self.spline_list = spline_list
        self.velocity = velocity

        self.last_u = 0.0
        self.consecutive_iterations = 0

        self.a_max = 0.05
        self.k_dev = 0.3
        self.k_dir = 1.0
        self.lookahead = 0.01

    def transition(self):
        if self.current_curve_i + 1 < len(self.spline_list):
            self.current_curve_i += 1
        else:
            print("we have reached the max")

    def adjust_u_if_stuck(self, u, epsilon=1e-5, n=10, adjustment_value=0.7):
        u_new = u
        if abs(u - self.last_u) < epsilon:
            self.consecutive_iterations += 1
            if self.consecutive_iterations >= n:
                self.current_curve_i += 1
                u_new = u + adjustment_value
        self.last_u = u_new
        return u

    def update_position(self, pos):
        self.current_position = pos
        max_iter = 100
        conv_threshold = 0.0001
        dist_closest = np.linalg.norm(
            self.spline_list[self.current_curve_i].get_position(self.closest_u)
            - np.float64(self.current_position)
        ) ** 2
        current_curve = self.spline_list[self.current_curve_i]
        dist_zero = np.linalg.norm(current_curve.p0 - self.current_position) ** 1

        u = self.closest_u if dist_closest < dist_zero else 0.0
        self.current_u = self.closest_u if dist_closest < dist_zero else 0.0

        for _ in range(max_iter):
            curve = self.spline_list[self.current_curve_i]
            p = curve.get_position(u) - self.current_position
            p_prime = curve.get_velocity(u)
            p_2prime = curve.get_second_derivative(u)
            grad = np.dot(p.T, p_prime) / (np.dot(p_prime.T, p_prime) + np.dot(p.T, p_2prime))

            if np.isnan(grad).any():
                break

            u = u - grad.item()
            if (np.abs(grad) < conv_threshold).any():
                break

        if u < 0:
            u = 0.0
        if u > self.current_u:
            self.current_u = u
        else:
            u = self.current_u
        if self.current_u > 1:
            u = 0.0
            self.transition()

        self.closest_u = self.current_u

    def get_current_curve(self):
        return self.spline_list[self.current_curve_i]

    def get_position_dev(self, pos):
        curve = self.spline_list[self.current_curve_i]
        return (curve.get_position(self.closest_u) - pos).squeeze()

    def get_current_dir(self):
        return self.spline_list[self.current_curve_i].get_velocity(self.closest_u).squeeze()

    def get_current_curvature(self):
        return self.spline_list[self.current_curve_i].get_curvature(self.closest_u)

    def get_closest_point(self):
        return self.get_current_curve().get_position(self.closest_u)

    def calculate_r_from_curvature(self, curv):
        return 1e6 if abs(curv) <= 0.0001 else 1 / abs(curv)

    def direction_target_function(self, dir, curvature):
        speed_compensation = 1.0 / (1 + abs(curvature))
        return dir * np.sqrt(self.a_max * speed_compensation)

    def get_control_target(self, curr_pos):
        dir_norm = self.get_current_dir() / np.linalg.norm(self.get_current_dir(), axis=0, keepdims=True)
        directionTarget_ = self.k_dev * self.get_position_dev(curr_pos) + dir_norm * self.k_dir
        dir = directionTarget_ / np.linalg.norm(directionTarget_)
        curvature = self.get_current_curvature()
        controller_target = np.array(self.direction_target_function(dir, curvature) * self.velocity)

        curve = self.spline_list[self.current_curve_i]
        _temp_u = np.clip(self.closest_u, 0.0, 0.95)
        pos_on_curve = curve.get_position(_temp_u + self.lookahead)

        return pos_on_curve, controller_target
