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
import ros_numpy  # apt install ros-noetic-ros-numpy numpy==1.23.0
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

import ctypes
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from open3d_ros_helper import open3d_ros_helper as orh

class clustering_and_filtering():
    def __init__(self):
        self.model = DBSCAN(eps=0.003, min_samples=50)
        self.point_cloud_sub = rospy.Subscriber("/points_concatenated", PointCloud2)
        self.microwave_door_handel_pub = rospy.Publisher("/microwave_door_handel", PointCloud2)
        self.microwave_door = rospy.Publisher("/microwave_door", PointCloud2)
        self.blue_mug = rospy.Publisher("/blue_mug", PointCloud2)
        self.gray_mug = rospy.Publisher("/gray_mug", PointCloud2)
        self.white_mug = rospy.Publisher("/white_mug", PointCloud2)

    def callback(self, pointclouds_msg):

        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        gen = point_cloud2.read_points(pointclouds_msg, skip_nans=True)
        int_data = list(gen)
        for x in int_data:
            test = x[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
            # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
            rgb = np.append(rgb,[[r,g,b]], axis = 0)

    def run(self):
        rospy.spin()  


def demo():
    rospy.init_node("clustering_and_filtering_node")
    clustering_and_filtering_node = clustering_and_filtering()
    clustering_and_filtering_node.run()

if __name__ == "__main__":

    demo()
