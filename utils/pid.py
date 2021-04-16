import numpy as np

class LongPID(object):
    def __init__(self, dt, min_a, max_a):
        self.dt = dt
        self.min_a, self.max_a = min_a, max_a
        self.last_error = 0
        self.sum_error = 0

    def run_step(self, v_current, v_target):
        Kp, Ki, Kd = 1.00, 0.01, 0.05

        error = v_target - v_current

        acceleration = Kp * error
        acceleration += Ki * self.sum_error * self.dt
        acceleration += Kd * (error - self.last_error) / self.dt

        self.last_error = error
        self.sum_error += error
        '''eliminate drift'''
        if abs(self.sum_error) > 10:
            self.sum_error = 0.0

        throttle = np.clip(acceleration, 0, self.max_a)
        brake = -np.clip(acceleration, self.min_a, 0)
        return throttle, brake
