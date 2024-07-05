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
        
        self.odom=Odometry()
        self.odom_pub = rospy.Publisher( "AprilTagOdom", Odometry, queue_size=1)
        
        self.cam_odom=Odometry()
        self.global_cam_pub = rospy.Publisher("CamOdom", Odometry, queue_size=1)

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
        result = self.detector.detect(gray_image, True, camera_params=(738.52671777, 739.11251938, 959.40116984, 575.51338683),tag_size = 0.155) 
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

                    
                    self.odom.pose.pose.orientation.x=odom_quat[0]
                    self.odom.pose.pose.orientation.y=odom_quat[1]
                    self.odom.pose.pose.orientation.z=odom_quat[2]
                    self.odom.pose.pose.orientation.w=odom_quat[3]
                    
                    self.odom.header.stamp=rospy.Time.now()
                    self.odom.header.frame_id="map"
                    self.odom_pub.publish(self.odom)

                    
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

if __name__ == '__main__':
    rospy.init_node('RobotAutoCalibrationNode', anonymous=True)   
    autocal = AutoAutoCal() 
    autocal.subscribeRobotImage()
    print(" Running Auto Auto Calib Node ")  
    rospy.spin()
    
