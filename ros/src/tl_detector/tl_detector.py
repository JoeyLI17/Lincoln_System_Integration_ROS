#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import os

import calendar
from time import gmtime, strftime

from scipy import spatial

STATE_COUNT_THRESHOLD = 2

class TLDetector(object):
    def __init__(self):
        
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''

        rospy.init_node('tl_detector')

        self.waypoint_tree = None
        self.has_image   = False
        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights =  []
        
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        
        self.debug = False
        self.frame_count = 0

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # not sure this usage is correct - so leave this out for now
#         rospy.wait_for_message('/base_waypoints', Lane)
#         rospy.wait_for_message('/current_pose', PoseStamped)
#         rospy.wait_for_message('/vehicle/traffic_lights', TrafficLightArray)
#         rospy.wait_for_message('/image_color', Image)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # add subscribers last
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1, buff_size=2**25)

        # everything depends on waypoints - so wait here until they arrive
        # this may break stuff! 
        rospy.wait_for_message('/base_waypoints', Lane)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        
        # rospy.logerr("waypoints_cb") ## don't show
        
        if self.waypoint_tree:
            rospy.logerr("waypoints already assigned - return")
        else:
            self.waypoints = waypoints
            waypoints_2d = [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = spatial.KDTree(waypoints_2d)  

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        
        #rospy.logerr("image_cb")
        
        if self.waypoints == None or self.waypoint_tree == None:
            self.has_image = False
            self.camera_image = None
            return
        
        self.has_image = True
        self.camera_image = msg

        if self.frame_count % 5 == 0:
    
            light_wp, state = self.process_traffic_lights()
            
    #         rospy.logerr("state: %s",state)
    #         rospy.logerr("wp: %s",light_wp)

            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                light_wp = light_wp if state == TrafficLight.RED else -1
                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1

        self.frame_count+=1    

    def get_closest_waypoint(self,x,y):
        
        """Identifies the closest path waypoint to the given position

        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
    
        if self.waypoint_tree is None:
            rospy.logerr("error: get_closest_waypoint_idx - waypoint_tree (kdtree) not assigned")
        else:
            closest_idx = self.waypoint_tree.query([x,y],1)[1]
            return closest_idx

    def get_light_state(self, light):
        
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
        if self.debug:
            return light.state
        
        else:
            
            if self.has_image is False or self.camera_image is None:
                self.prev_light_loc = None
                return TrafficLight.UNKNOWN
            
            if self.light_classifier is None:
                return TrafficLight.UNKNOWN

            # use computer vision to detect traffic light and it's state
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        
        if (self.waypoints is None) or (self.waypoint_tree is None):
            rospy.logerr("process_traffic_lights: no waypointa")
            return -1, TrafficLight.UNKNOWN
        
        #light = None
        closest_light = None
        line_wp_idx = None
        
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        site_sim = self.config['is_site']

        if(self.pose):
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y
            car_wp_idx = self.get_closest_waypoint(x,y)

            diff = len(self.waypoints.waypoints)
            if(self.lights):
                for i, light in enumerate(self.lights):

                    line = stop_line_positions[i]
                    x = line[0]
                    y = line[1]

                    temp_wp_idx = self.get_closest_waypoint(x,y)   

                    d = temp_wp_idx - car_wp_idx

                    if d >= 0 and d < diff:
                        diff = d
                        closest_light = light
                        state = light.state
                        
                        # self.__create_training_data(state)
                        
                        line_wp_idx = temp_wp_idx
            
        elif(site_sim == 1): # run site test bag
            state = self.get_light_state(None)

        if closest_light != None:
            #rospy.logerr("closest light:")
            state = self.get_light_state(closest_light)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN


    def __light_label(self, state):
        if state == TrafficLight.RED:
            return "RED"
        elif state == TrafficLight.YELLOW:
            return "YELLOW"
        elif state == TrafficLight.GREEN:
            return "GREEN"
        return "UNKNOWN"
    
    def __create_training_data(self, state):
        
        f_name = "sim_tl_{}_{}.jpg".format(calendar.timegm(gmtime()), self.__light_label(state))
        dir = './data/train/sim'

        if not os.path.exists(dir):
            os.makedirs(dir)

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image)
        cv_image = cv_image[:, :, ::-1]
        cv2.imwrite('{}/{}'.format(dir, f_name), cv_image)
        

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
