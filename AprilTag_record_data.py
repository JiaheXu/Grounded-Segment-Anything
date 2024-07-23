#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pupil_apriltags import Detector
import tf.transformations as tr
from geometry_msgs.msg import Pose
from geometry_msgs.msg import *
import tf2_ros
import tf
from std_srvs.srv import SetBool
from std_msgs.msg import Bool, Int32, UInt8,UInt32MultiArray
from visualization_msgs.msg import Marker
import numpy as np

from sensor_msgs.msg import Joy

from std_msgs.msg import Bool, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose,Quaternion, Vector3, Point
from tf.transformations import *
import time

import cv2
import numpy as np

# distortion_model: "rational_polynomial"
# D: [0.4385905861854553, -2.6185202598571777, -0.00028256000950932503, -0.00051872682524845, 1.5916898250579834, 0.3232973515987396, -2.449460506439209, 1.5187499523162842]

# K: [607.1500244140625, 0.0, 641.7113647460938,
#     0.0, 607.0665893554688, 365.9603576660156,
#     0.0, 0.0, 1.0]

# R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

# P: [607.1500244140625, 0.0, 641.7113647460938,
#     0.0, 607.0665893554688, 365.9603576660156,
#     0.0, 0.0, 1.0]
aprilTag_R = np.array([
    [1.,0.,0.],
    [0.,-1.,0.],
    [0.,0.,-1.],
    ])
class AutoAutoCal:
    def __init__(self):
    
        self.bridge = CvBridge()
        self.relative_pose= Odometry()          
        # Create a detector object    
        self.detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
        )     

        self.left_joystick_x = 0
        self.left_joystick_y = 1
        self.left_trigger = 2
        self.right_joystick_x = 3
        self.right_joystick_y = 4
        self.right_trigger = 5
        self.dpad_left_right = 6
        self.dpad_up_down = 7

        self.max_idx = max(
            self.left_trigger,
            self.left_joystick_x,
            self.left_joystick_y,
            self.dpad_up_down,
            self.right_joystick_x,
            self.right_joystick_y,
            self.right_trigger,
        )

        # Buttons for control
        self.a_button = 0
        self.b_button = 1
        self.x_button = 2
        self.y_button = 3
        self.lb_button = 4
        self.rb_button = 5
        self.back_button = 6
        self.start_button = 7
        self.xbox_button = 8
        self.left_joystick_button = 9
        self.right_joystick_button = 10
        
        self.max_button = max(
            self.b_button,
            self.y_button,
            self.x_button,
            self.a_button,
            self.rb_button,
            self.back_button,
            self.start_button,
            self.left_joystick_button,
            self.right_joystick_button,
            self.lb_button
        )

        # states
        self.recording = False
        # data
        self.current_stack = []
        self.success_stop_pressed_last = False
        self.failure_stop_pressed_last = False

        self.odom=Odometry()
        self.odom_pub = rospy.Publisher( "AprilTagOdom", Odometry, queue_size=1)
        
        self.cam_odom=Odometry()
        self.global_cam_pub = rospy.Publisher("CamOdom", Odometry, queue_size=1)

        self.joystick_sub = rospy.Subscriber("joy", Joy, self.joyCallback)
        
        # Camera Instrinsics used for Undistorting the Fisheye Images
        self.DIM=(1080, 1920)
        self.K=np.array([[738.52671777, 0., 959.40116984], [0. ,739.11251938,  575.51338683], [0.0, 0.0, 1.0]])

        self.D=np.array([0.0, 0.0, 0.0, 0.0])
            
    def subscribeRobotImage(self):
        self.robo_img_subscriber=rospy.Subscriber("/cam1/zed_node_A/left/image_rect_color", Image, self.image_callback) # 

    def image_callback(self, msg):
        # Process the received image data here
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # print()
        result = self.detector.detect(gray_image, True, camera_params=(738.52671777, 739.11251938, 959.40116984, 575.51338683),tag_size = 0.154) 
        # Camera Intrinsics after undistorting fisheye images, Tag Size is the length of the side of an aprilTag
        
        if result: 
            # print("*****************************************************************************************")
            # print(result)
            for tag in result: 
                if(tag.tag_id):  
                
                    original_estimated_rot = tag.pose_R 
                    original_estimated_trans = tag.pose_t
                    original_estimated_rot =   tag.pose_R @ aprilTag_R
                    # print("trans: ", original_estimated_trans)
                    # print("rot: ", original_estimated_rot)
                    # print("original_estimated_rot", type(original_estimated_rot) )

                    roll, pitch, yaw = euler_from_matrix(original_estimated_rot)
      
                    odom_quat = quaternion_from_euler(roll, pitch, yaw)  
                    # self.odom.pose.pose.position = Point(original_estimated_trans[2], -original_estimated_trans[0], -original_estimated_trans[1])
                    self.odom.pose.pose.position = Point(original_estimated_trans[0], original_estimated_trans[1], original_estimated_trans[2])

                    x = original_estimated_trans[0]
                    y = original_estimated_trans[1]
                    z = original_estimated_trans[2]

                    self.odom.pose.pose.orientation.x=odom_quat[0]
                    self.odom.pose.pose.orientation.y=odom_quat[1]
                    self.odom.pose.pose.orientation.z=odom_quat[2]
                    self.odom.pose.pose.orientation.w=odom_quat[3]
                    
                    self.odom.header.stamp=rospy.Time.now()
                    self.odom.header.frame_id="cam1"
                    self.odom_pub.publish(self.odom)
                    self.tag_pose = [ x, y, z, odom_quat[0], odom_quat[1], odom_quat[2], odom_quat[3] ]
                    self.current_stack.append(self.tag_pose)

                    global_cam_rot = original_estimated_rot.transpose()
                    global_cam_trans = -1*global_cam_rot@original_estimated_trans

                    print("trans: ", global_cam_trans)
                    print("rot (x y z w): \n", self.odom.pose.pose.orientation)

                    roll, pitch, yaw = euler_from_matrix(global_cam_rot)
                    odom_quat = quaternion_from_euler(roll, pitch, yaw)
                    self.cam_odom.pose.pose.orientation.x=odom_quat[0]
                    self.cam_odom.pose.pose.orientation.y=odom_quat[1]
                    self.cam_odom.pose.pose.orientation.z=odom_quat[2]
                    self.cam_odom.pose.pose.orientation.w=odom_quat[3]
                    self.cam_odom.pose.pose.position = Point(global_cam_trans[0], global_cam_trans[1], global_cam_trans[2])
                    self.cam_odom.header.stamp=rospy.Time.now()
                    self.cam_odom.header.frame_id="map"
                    self.global_cam_pub.publish(self.cam_odom)

    def save_data(self):
        now = time.time()
        print("collected ", len(self.current_stack), " pairs of data")
        print("collected ", len(self.current_stack), " pairs of data")
        print("collected ", len(self.current_stack), " pairs of data")
        np.save( str(now), self.current_stack)
    
    def clean_data(self):
        self.current_stack.clear()

    def episode_end(self, success_flag):
        if( success_flag == True):
            self.save_data()
        self.clean_data()

    def joyCallback(self, msg):
        start_recording_pressed = msg.buttons[self.triangle_button]
        success_stop_pressed = msg.buttons[self.o_button]
        failure_stop_pressed = msg.buttons[self.x_button]


        if( (start_recording_pressed == True) and (self.start_recording_pressed_last == False) ):
            if( self.recording == False ):
                self.get_logger().info('start recording!!!')
            else:
                self.recording = True
                self.episode_end(False)
                self.get_logger().info('start recording!!!')
                # self.get_logger().info('start recording!!!')                

        if( (success_stop_pressed == True) and (self.success_stop_pressed_last == False) ):
            if( self.recording == True ):
                self.recording = False
                self.episode_end(True)
                self.get_logger().info('episode succeed!!!')

        if( (failure_stop_pressed == True) and (self.failure_stop_pressed_last == False) ):
            if( self.recording == True ):
                self.recording = False
                self.episode_end(False)
                self.get_logger().info('episode failed!!!')

        self.start_recording_pressed_last = start_recording_pressed
        self.success_stop_pressed_last = success_stop_pressed           
        self.failure_stop_pressed_last = failure_stop_pressed

if __name__ == '__main__':
    rospy.init_node('RobotAutoCalibrationNode', anonymous=True)   
    autocal = AutoAutoCal() 
    autocal.subscribeRobotImage()
    print(" Running Auto Auto Calib Node ")  
    rospy.spin()
    
