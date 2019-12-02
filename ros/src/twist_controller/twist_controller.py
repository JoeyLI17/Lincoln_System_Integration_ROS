from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
        # YawControl
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        kp = 0.3
        kd = 0.
        ki = 0.1
        min_gas = 0.
        max_gas = 0.2
        self.throttle_controller = PID(kp, ki, kd, min_gas, max_gas)

        tau = 0.5
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau, ts)

        self.decel_limit    = decel_limit
        self.accel_limit    = accel_limit
        self.wheel_radius   = wheel_radius
        self.vehicle_mass   = vehicle_mass
        self.fuel_capacity  = fuel_capacity
        self.brake_deadband = brake_deadband
        
        
        self.last_time      = rospy.get_time()


    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        return 1., 0., 0.
