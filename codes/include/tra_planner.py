import numpy as np
from tra_spline import CubicSpline
from typing import List

controller_target = {
    "vel": 0.0,
    "heading": 0.0
}


class LinearLocalPlanner:
    def __init__(self, spline_list: List[CubicSpline], velocity):
        self.closest_u = 0.0
        self.current_curve_i = 0
        self.current_position = None
        self.spline_list = spline_list
        self.velocity = velocity

        # for fixing u "stuckness"
        self.last_u = 0.0
        self.consecutive_iterations = 0

        # params
        self.a_max = 0.05

        self.k_dev = 0.1
        self.k_dir = 1.0

    def transition(self):
        if self.current_curve_i + 1 < len(self.spline_list):
            # Update the current curve and other necessary operations
            self.current_curve_i += 1
        else:
            # You've reached the end of the spline_list, so you can't transition anymore
            print("we have reached the max")


    def adjust_u_if_stuck(self, u, epsilon=1e-5, n=10, adjustment_value=0.7):
        u_new = u
        if abs(u - self.last_u) < epsilon:
            self.consecutive_iterations += 1
            if self.consecutive_iterations >= n:
                # u += adjustment_value
                self.current_curve_i += 1
                u_new = u + adjustment_value
                #self.consecutive_iterations = 0
        self.last_u = u_new
        return u

    def update_position(self, pos):
        self.current_position = pos
        max_iter = 100
        conv_threshold = 0.0001
        dist_closest = np.linalg.norm(
            self.spline_list[self.current_curve_i].get_position(self.closest_u) - np.float64(self.current_position)) ** 2
        current_curve = self.spline_list[self.current_curve_i]

        dist_zero = np.linalg.norm(current_curve.p0 - self.current_position) ** 1
        u = self.closest_u if dist_closest < dist_zero else 0.0
        self.current_u = self.closest_u if dist_closest < dist_zero else 0.0
        #u = self.adjust_u_if_stuck(u)
        # print(self.current_position)
        for _ in range(max_iter):
            curve = self.spline_list[self.current_curve_i]
            p = curve.get_position(u) - self.current_position
            p_prime = curve.get_velocity(u)
            p_2prime = curve.get_second_derivative(u)

            grad = np.dot(p.T, p_prime) / (np.dot(p_prime.T, p_prime) + np.dot(p.T, p_2prime))
            #print(grad.item())
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
        #print("this is the closest point ", self.closest_u)


    def get_current_curve(self):
        return self.spline_list[self.current_curve_i]

    def get_position_dev(self, pos):
        curve = self.spline_list[self.current_curve_i]
        pos_on_curve = curve.get_position(self.closest_u)
        dist = pos_on_curve - pos
        return dist.squeeze()

    def get_current_dir(self):
        curve = self.spline_list[self.current_curve_i]
        dir = curve.get_velocity(self.closest_u) # todo mazbe normailze this
        return dir.squeeze()

    def get_current_dir_new(self):
        curve = self.spline_list[self.current_curve_i]
        direction = curve.get_velocity(self.closest_u)
        return direction

    def get_current_curvature(self):
        curve = self.spline_list[self.current_curve_i]
        curvature = curve.get_curvature(self.closest_u)
        #print("this is the current curvature ", curvature)
        return curvature

    def get_closest_point(self):
        return self.get_current_curve().get_position(self.closest_u)

    def calculate_r_from_curvature(self, curv):
        if abs(curv) <= 0.0001:
            # Handle the case of a small curvature by using a large radius (no curvature)
            radius = 1e6  # Use a very large value or handle it as needed
        else:
            radius = 1 / abs(curv)  # Compute the radius for non-zero curvature
        return radius

    def direction_target_function(self, dir, curvature):
        # You can adjust the speed compensation factor (e.g., 1.0 for no change) based on curvature.
        speed_compensation = 1.0 / (1 + abs(curvature))  # Adjust this factor as needed
        return dir * np.sqrt(self.a_max * speed_compensation)

    def get_control_target(self, curr_pos):

        # Heading
        # directionTarget_ =  self.k_heading * self.get_position_dev(curr_pos) + self.get_current_curvature()
        dir_norm = self.get_current_dir() / np.linalg.norm(self.get_current_dir(), axis=0, keepdims=True)
        directionTarget_ = self.k_dev * self.get_position_dev(curr_pos) + dir_norm * self.k_dir
        dir = directionTarget_ / np.linalg.norm(directionTarget_)
        curvature = self.get_current_curvature()
        controller_target = self.direction_target_function(dir,curvature)
        controller_target = np.array(controller_target)

        return controller_target


if __name__ == "__main__":
    # Create a list of Bézier curves (you should replace this with your actual data)
    point_array = np.array([[0, 0], [0.25, 0.5], [0.5, 1.0], [1.0, 1.0]], dtype=np.float64)
    bez_curve = CubicSpline(p0=point_array[0][:], p1=point_array[1][:], p2=point_array[2][:], p3=point_array[3][:])
    spline_list = [bez_curve]

    # Initialize the LinearLocalPlanner
    planner = LinearLocalPlanner(spline_list, velocity=1.0)

    # Update the position (provide the actual position as an argument)
    planner.update_position(np.array([0.2, 0.3], dtype=np.float64))
