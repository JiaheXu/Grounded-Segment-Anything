import argparse
import os
import sys

import numpy as np
import json
import torch
import PIL.Image as PIL_Image

# sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
# sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# # Grounding DINO
# import GroundingDINO.groundingdino.datasets.transforms as T
# from GroundingDINO.groundingdino.models import build_model
# from GroundingDINO.groundingdino.util.slconfig import SLConfig
# from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# import std_msgs

# # segment anything
# from segment_anything import (
#     sam_model_registry,
#     sam_hq_model_registry,
#     SamPredictor
# )
# from igev_stereo import IGEVStereo
# from utils.utils import InputPadder

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time



# ros things
import message_filters
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
bridge = CvBridge()

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
from sensor_msgs import point_cloud2
from lib_cloud_conversion_between_Open3D_and_ROS import convertCloudFromRosToOpen3d
# import ros_numpy  # apt install ros-noetic-ros-numpy numpy==1.23.0
# import ros_numpy

###################################### IGEV stuff

sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import torch
from tqdm import tqdm
from pathlib import Path
import copy

import ctypes
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import DBSCAN
import open3d as o3d
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int32, UInt8,UInt32MultiArray
from geometry_msgs.msg import Pose,Quaternion, Vector3, Point

class clustering_and_filtering():
    def __init__(self):
        self.xyz = None
        self.rgb = None
        self.labels = None

        self.bound_box = np.array( [ [0.0, 1.0], [ -1.0 , 0.0], [ 0.0 , 0.4] ] )

        self.mug_blue_color = np.array( [134.42705382, 154.07507082, 134.78753541] )
        self.mug1_color = np.array([129.5595624,  130.43354943, 117.42058347])
        self.mug2_color = np.array( [136.10209601, 156.15821501, 135.78904665])
        # self.model = DBSCAN(eps=0.003, min_samples=50)
        self.point_cloud_sub = rospy.Subscriber("/object_point_cloud2", PointCloud2, self.callback)

        self.mug_blue_odom=Odometry()
        self.mug_blue = rospy.Publisher("mug_blue", Odometry, queue_size=1)

        self.mug1_odom=Odometry()
        self.mug1 = rospy.Publisher("mug1", Odometry, queue_size=1)

        self.mug2_odom=Odometry()
        self.mug2 = rospy.Publisher("mug2", Odometry, queue_size=1)

        self.mug_number = rospy.Publisher("mug_number", Int32, queue_size=1)

    def update_position(self, pcd, labels):
        max_label = labels.max() + 1
        if(max_label < 1):
            return
        npy_xyz = np.asarray( pcd.points )
        npy_rgb = np.asarray( pcd.colors ) * 255.0

        for idx in range(max_label):
            pcd_idx = np.where(labels == idx)
            pose = np.mean( npy_xyz[pcd_idx], axis = 0 )
            rgb = np.mean( npy_rgb[pcd_idx], axis = 0 )
            # print("pose: ", pose)
            # print("rgb: ", rgb)
            # print("idx: ", idx)
            diff1 = np.sum( np.abs( rgb - self.mug_blue_color) )
            diff2 = np.sum( np.abs( rgb - self.mug1_color) )
            diff3 = np.sum( np.abs( rgb - self.mug2_color) )
            min_diff = np.min( np.array( [diff1, diff2, diff3]))
            # print(" ")
            if( np.abs(diff1 - min_diff) <1e-4):
                self.mug_blue_odom.pose.pose.position = Point(pose[0], pose[1], pose[2])
                self.mug_blue_odom.header.stamp=rospy.Time.now()
                self.mug_blue_odom.header.frame_id="map"
                

            if( np.abs(diff2 - min_diff) <1e-4):
                self.mug1_odom.pose.pose.position = Point(pose[0], pose[1], pose[2])
                self.mug1_odom.header.stamp=rospy.Time.now()
                self.mug1_odom.header.frame_id="map"
                

            if( np.abs(diff3 - min_diff) <1e-4):
                self.mug2_odom.pose.pose.position = Point(pose[0], pose[1], pose[2])
                self.mug2_odom.header.stamp=rospy.Time.now()
                self.mug2_odom.header.frame_id="map"

        self.mug_blue.publish(self.mug_blue_odom)
        self.mug1.publish(self.mug1_odom)
        self.mug2.publish(self.mug2_odom)
        number = Int32()
        number.data = max_label
        self.mug_number.publish(max_label)
        # 
        # print("round end")

    def callback(self, pointclouds_msg):
        print("in call back")
        
        pcd = o3d.geometry.PointCloud()
        pcd = convertCloudFromRosToOpen3d( pointclouds_msg )

        uniform_down_pcd = pcd.uniform_down_sample(every_k_points=10)
        # o3d.visualization.draw_geometries([uniform_down_pcd])
        
        # print(uniform_down_pcd)
        
        labels = np.array(uniform_down_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))

        self.update_position(uniform_down_pcd, labels)


        # pcd = copy.deepcopy(uniform_down_pcd)
        # max_label = labels.max()
        # print(f"point cloud has {max_label + 1} clusters")
        # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        # colors[labels < 0] = 0
        # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # o3d.visualization.draw_geometries([pcd])
    def run(self):
        rospy.spin()  


def demo():
    rospy.init_node("clustering_and_filtering_node")
    clustering_and_filtering_node = clustering_and_filtering()
    clustering_and_filtering_node.run()

if __name__ == "__main__":

    demo()
