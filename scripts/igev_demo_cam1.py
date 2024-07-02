import sys
sys.path.append('core')
DEVICE = 'cuda'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

#sys.path.append(os.path.join(os.getcwd(), "IGEV/IGEV-Stereo"))

from igev_stereo import IGEVStereo
from utils.utils import InputPadder
# from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
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

class igev_stereo():
    def __init__(self, args):

        self.args = args
        self.model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
        self.model.load_state_dict(torch.load(args.restore_ckpt))

        self.model = self.model.module
        self.model.to(DEVICE)
        self.model.eval()
        
        self.cameraMatrix1 = []
        self.cameraMatrix2 = []
        
        self.distCoeffs1 = []
        self.distCoeffs2 = []

        self.unified_matrix = np.array([
            [739.1656494140625, 0., 951.3283081054688],
            [0. ,739.1656494140625,  566.4527587890625],
            [0., 0., 1.0]
        ]) 
        # 1080p setting, need to modify into yaml file in the future 1080*1920
        self.cameraMatrix1.append( self.unified_matrix )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        self.cameraMatrix2.append( self.unified_matrix )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        # 720p setting, need to modify into yaml file in the future 720*1280
        self.cameraMatrix1.append( self.unified_matrix * 2.0 / 3.0)

        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        self.cameraMatrix2.append( self.unified_matrix * 2.0 / 3.0 )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        self.imageSize = []

        self.imageSize.append( (1080, 1920) )
        self.imageSize.append( (720, 1280) )
        self.R = np.array([ 
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],            
        ])
        self.T = np.array([ 
            [-0.12],
            [0.],
            [0.]            
        ])

        self.Q = []
        for i in range ( len( self.imageSize ) ):
            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(self.cameraMatrix1[i], self.distCoeffs1[i], self.cameraMatrix2[i], self.distCoeffs2[i], self.imageSize[i], self.R, self.T)
            self.Q.append(Q)
 
        #Todo: feed a startup all zero image to the network
        self.cam1_sub = message_filters.Subscriber(args.left_topic, Image)
        self.cam2_sub = message_filters.Subscriber(args.right_topic, Image)
        self.depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        self.conf_map_sub = message_filters.Subscriber(args.conf_map_topic, Image)

        # self.disparity_pub = rospy.Publisher("zedx/disparity", PointCloud2, queue_size=1)

        self.point_cloud_pub = rospy.Publisher("cam1/igev/point_cloud2", PointCloud2, queue_size=1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_sub, self.cam2_sub, self.depth_sub, self.conf_map_sub], 10, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def load_image(self, img):
        img = img.astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

    def disparity_to_depth(self, disparity):
        #720
        focal_length = self.unified_matrix[0][0] * 2.0 /3.0
        
        if(disparity.shape[0] == 1080):
            focal_length = self.unified_matrix[0][0]

        depth = (0.12 * focal_length) / disparity
        # depth_valid =  np.logical_and( np.logical_not(np.isnan(image_depth_np)), np.logical_not(np.isinf(image_depth_np)) )
        return depth

    def depth_to_disparity(self, depth):
        #720
        focal_length = self.unified_matrix[0][0] * 2.0 /3.0
        
        if(depth.shape[0] == 1080):
            focal_length = self.unified_matrix[0][0]

        disparity = (0.12 * focal_length) / depth
        return disparity



    def compare_depth(self, depth1, depth2):
        if(depth1.shape != depth2.shape):
            print("different img dims!!!!! ", depth1.shape, " ", depth2.shape)
            return None
        np_nan = np.isnan(depth1)
        # print("np.isnan(depth1): ", np.logical_not(np_nan) )
        depth1_valid =  np.logical_and( np.logical_not(np.isnan(depth1)), np.logical_not(np.isinf(depth1)) )
        depth2_valid =  np.logical_and( np.logical_not(np.isnan(depth2)), np.logical_not(np.isinf(depth2)) )
        common_ele = np.logical_and( depth1_valid, depth2_valid )

        print("mean: ", np.mean( np.abs(depth1[common_ele] - depth2[common_ele])) )
        diff = np.zeros( depth2.shape )
        diff[common_ele] = np.abs(depth1[common_ele] - depth2[common_ele])
        return diff


    def merge(self, default_depth, conf_map, depth_valid, igev_depth, conf_threshold):
        merge_depth = igev_depth.copy()
        valid_conf_depth = np.logical_and( conf_map < conf_threshold, depth_valid==True )
        merge_depth[valid_conf_depth] = default_depth[valid_conf_depth]
        return merge_depth

    def callback(self, cam1_msg, cam2_msg, depth_msg, conf_map_msg):

        print("callback")
        with torch.no_grad():
            image1 = bridge.imgmsg_to_cv2(cam1_msg)
            image1_np = np.array(image1[:,:,0:3])
            image2 = bridge.imgmsg_to_cv2(cam2_msg)
            image2_np = np.array(image2[:,:,0:3])

            image_depth = bridge.imgmsg_to_cv2(depth_msg)
            image_depth_np = np.array(image_depth)
            conf_map = bridge.imgmsg_to_cv2(conf_map_msg)
            conf_map_np = np.array(conf_map)

            # downsampel to 720p to speed up
            if(self.args.downsampling):
                image1_np = cv2.resize(image1_np,(1280,720))
                image2_np = cv2.resize(image2_np,(1280,720))
                image_depth_np = cv2.resize(image_depth_np,(1280,720))
                conf_map_np = cv2.resize(conf_map_np,(1280,720))

            depth_valid =  np.logical_and( np.logical_not(np.isnan(image_depth_np)), np.logical_not(np.isinf(image_depth_np)) )
            # image_depth_np[depth_valid == False] = np.max( image_depth_np[depth_valid == True] )

            print("image1.shape: ", image1_np.shape)
            print("image2.shape: ", image2_np.shape)
            cv2.imwrite('img1.png', image1_np)
            cv2.imwrite('img2.png', image2_np)



            # preprocess FOR igev
            image1= self.load_image(image1_np)
            image2= self.load_image(image2_np)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            start = time.time()
            igev_disp = self.model(image1, image2, iters=args.valid_iters, test_mode=True)
            end = time.time()
            print("torch inference time: ", end - start)
            igev_disp = igev_disp.cpu().numpy()
            igev_disp = padder.unpad(igev_disp)
            igev_disp = igev_disp.squeeze()
            print("igev_disp: ", igev_disp.shape)

            default_disp = self.depth_to_disparity( image_depth_np )
            default_disp = np.float32( default_disp )
            # default_disp[image_depth_np]

            print("default_disp: ", default_disp.shape)
            igev_disp = np.float32( igev_disp )

            # minn = min( np.min(default_disp[depth_valid]), np.min(igev_disp) )
            # maxx = max( np.max(default_disp[depth_valid]), np.max(igev_disp) )
            minn = 0.0
            maxx = 200
            # print("minn: ", minn)
            # print("maxx: ", maxx)

            std_igev_disp = (igev_disp - minn ) / (maxx - minn)
            std_default_disp = ( default_disp - minn ) / (maxx - minn)
            
            std_default_disp[ depth_valid == False] = 0

            plt.imsave("IGEV_disp.png", std_igev_disp, cmap='jet')
            plt.imsave("default_disp.png", std_default_disp, cmap='jet')

            # conf_map_np = conf_map_np / 100 * 255
            print("confidence min: ", np.min(conf_map_np))
            print("confidence max: ", np.max(conf_map_np))
            plt.imsave("default_confidence.png", conf_map_np, cmap='gray')

            igev_depth = self.disparity_to_depth(igev_disp)
            # igev_depth[ igev_depth > 1.] = 1.
            print("igev_depth default_disp", igev_depth.shape, " ", default_disp.shape )
            if(igev_depth.shape[0] != default_disp.shape[0]):
                igev_depth = cv2.resize(igev_depth, default_disp.shape)
            # image_depth_np[ depth_valid == False] = 0.0

            print("IGEV:", np.max(igev_depth), " ", np.min(igev_depth) )

            # image_depth_np[image_depth_np>1.] = 1.
            print("stereo:", np.max(image_depth_np[depth_valid]), " ", np.min(image_depth_np[depth_valid]) )

            # diff  = self.compare_depth(igev_depth, image_depth_np)
            diff = np.zeros( igev_depth.shape )
            diff[depth_valid] = np.abs( image_depth_np[depth_valid] - igev_depth[depth_valid] )

            diff_grayscale = ( diff - np.min(diff) ) / ( np.max(diff) - np.min(diff)) * 255
            print("mean: ", np.mean(diff[depth_valid]))
            plt.imsave("diff.png", diff_grayscale, cmap='gray')

            merge_depth = self.merge(image_depth_np, conf_map_np, depth_valid, igev_depth, conf_threshold = 2)
            merge_disp = self.depth_to_disparity(merge_depth)
            std_merge_disp = ( merge_disp - minn ) / (maxx - minn)
            plt.imsave("merge_disp.png", std_merge_disp, cmap='jet')         


            # rgb point cloud, reference : https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
            disp = igev_disp
            image_3d = cv2.reprojectImageTo3D(disp, self.Q[0])
            if(self.args.downsampling == True):
                image_3d = cv2.reprojectImageTo3D(disp, self.Q[1])

            points = []
            lim = 8
            for i in range( image_3d.shape[0] ):
                for j in range(image_3d.shape[1]):
                        x = image_3d[i][j][0]
                        y = image_3d[i][j][1]
                        z = image_3d[i][j][2]
                        b = image1_np[i][j][0]
                        g = image1_np[i][j][1]
                        r = image1_np[i][j][2]
                        a = 255
                        # print r, g, b, a
                        rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                        # print hex(rgb)
                        pt = [x, y, z, rgb]
                        points.append(pt)
            print("finished")   
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgba', 12, PointField.UINT32, 1),
                    ]

            header = Header()
            header.frame_id = "cam1"
            
            pc2 = point_cloud2.create_cloud(header, fields, points)
            pc2.header.stamp = rospy.Time.now()
            self.point_cloud_pub.publish(pc2)
    def run(self):
        rospy.spin()  



def demo(args):
    rospy.init_node("cam1_igev_node")
    igev_stereo_node = igev_stereo(args)
    igev_stereo_node.run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/sceneflow/sceneflow.pth')
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/middlebury/middlebury.pth')
    # parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained_models/eth3d/eth3d.pth')

    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.png")

    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output/")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")


    parser.add_argument('--left_topic', type=str, default="/cam1/zed_node_A/left/image_rect_color", help="left cam topic")
    parser.add_argument('--right_topic', type=str, default="/cam1/zed_node_A/right/image_rect_color", help="right cam topic")
    parser.add_argument('--depth_topic', type=str, default="/cam1/zed_node_A/depth/depth_registered", help="depth cam topic")
    parser.add_argument('--conf_map_topic', type=str, default="/cam1/zed_node_A/confidence/confidence_map", help="depth confidence map topic")

    # parser.add_argument('--downsampling', type=bool, default=False, help="downsampling image dimension")
    parser.add_argument('--downsampling', type=bool, default=False, help="downsampling image dimension")
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)
