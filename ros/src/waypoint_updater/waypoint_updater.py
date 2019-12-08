#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy import spatial
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5
LOOP_RATE = 5

class WaypointUpdater(object):
    def __init__(self):
        
        #rospy.init_node('waypoint_updater',log_level=rospy.DEBUG)
        rospy.init_node('waypoint_updater')

        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.waypoints_2d   = None
        self.waypoint_tree  = None
        self.pose           = None
        
        self.stopline_idx = -1
        self.obstacle_idx = -1
        self.stop_buffer  = 2

        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)   
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=10)
        
        # these recieve only a waypoint index that refers back to the self.base_waypoints list
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb, queue_size=10)
        
        # optional: 
        #rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        self.loop()
    
    def loop(self):
        
        rate = rospy.Rate(LOOP_RATE)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoint_tree:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                #rospy.logerr("cwp %s",closest_waypoint_idx)
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()    

    def get_closest_waypoint_idx(self):

        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        if self.waypoint_tree == None:
            rospy.logerr("error: get_closest_waypoint_idx - waypoint_tree (kdtree) not assigned")
            return
        
        closest_idx = self.waypoint_tree.query([x,y],1)[1]
        
        # ahead or behind?
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        
        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        
        return closest_idx
    
    def publish_waypoints(self, closest_idx):
        
        #rospy.logerr("--- closest: %s: ",closest_idx)
        
        lane = Lane()
        lane.header = self.base_waypoints.header
        
        if self.base_waypoints.waypoints == None:
            rospy.logerr("error: self.base_waypoints.waypoints== none!")
            return

        size = len(self.base_waypoints.waypoints)
        sl_idx = self.stopline_idx

        lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx+LOOKAHEAD_WPS]
          
        if sl_idx >= 0 and sl_idx < size:
            #rospy.logerr("decelerate waypoint")
            lane.waypoints = self.decelerate_waypoints(closest_idx,lane.waypoints)    
   
        self.final_waypoints_pub.publish(lane)
        
    def pose_cb(self, msg):        
        self.pose = msg
        
    def waypoints_cb(self, waypoints):
            
        #rospy.logerr("wp updater: waypoints_cp")    
        size = len(waypoints.waypoints)
     
        # make a copy as they are only sent once
        self.base_waypoints = waypoints
        
        #use scipi KDTree to get closes waypoint
        if not self.waypoints_2d:
             self.waypoints_2d = [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
             self.waypoint_tree = spatial.KDTree(self.waypoints_2d)   
        else:
            rospy.logerr("self.waypoints_2d already assigned?: %s",self.waypoints_2d)
        
    def traffic_cb(self, msg):
        self.stopline_idx = msg.data
        #rospy.logerr("waypoint_updater: way point index %s ",self.stopline_idx)
        
    # def obstacle_cb(self, msg):
    #     self.obstacle_idx = msg
       
    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def decelerate_waypoints(self,closest_idx,wp_in):
        
        wp_out = []

        for i, wp in enumerate(wp_in):
            
            p = Waypoint()
            p.pose = wp.pose
            
            stop_idx = max(self.stopline_idx - closest_idx - self.stop_buffer,0)
            
            dist = self.distance(self.base_waypoints.waypoints, i, stop_idx)
            vel = math.sqrt(2*MAX_DECEL*dist)
            if vel < 1.0:
                vel = 0
                
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            wp_out.append(p)
            
        return wp_out


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
