import argparse
import os
import sys

import numpy as np
import json
import torch
import PIL.Image as PIL_Image
from scipy.spatial.transform import Rotation
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import std_msgs

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
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
# import ros_numpy

class grounded_sam():
    def __init__(self, args):

        self.args = args
        # cfg
        
        self.config_file = args.config  # change the path of the model config file
        self.grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
        self.sam_version = args.sam_version
        self.sam_checkpoint = args.sam_checkpoint
        self.sam_hq_checkpoint = args.sam_hq_checkpoint
        self.use_sam_hq = args.use_sam_hq
        #image_path = args.input_image
        self.text_prompt = args.text_prompt
        self.output_dir = args.output_dir
        self.box_threshold = args.box_threshold
        self.text_threshold = args.text_threshold
        self.device = args.device

        # make dir
        os.makedirs(self.output_dir, exist_ok=True)
        # load image
        # image_pil, image = self.load_image(image_path)
        
        # load model DINO
        self.model = self.load_model(self.config_file, self.grounded_checkpoint, device=self.device)

        # initialize SAM
        self.predictor = None
        if self.use_sam_hq:
            self.predictor = SamPredictor(sam_hq_model_registry[self.sam_version](checkpoint=self.sam_hq_checkpoint).to(self.device))
        else:
            self.predictor = SamPredictor(sam_model_registry[self.sam_version](checkpoint=self.sam_checkpoint).to(self.device))


        self.cameraMatrix1 = []
        self.cameraMatrix2 = []
        
        self.distCoeffs1 = []
        self.distCoeffs2 = []
        self.cam1_intrinsic = np.array([
            [738.52671777, 0., 959.40116984],
            [0. ,739.11251938,  575.51338683],
            [0., 0., 1.0]
        ])
        self.cam2_intrinsic = np.array([
            [734.44273774, 0., 988.65633632],
            [0. ,735.25578376,  532.02040634],
            [0., 0., 1.0]
        ])
        self.cam3_intrinsic = np.array([
            [729.76729744, 0., 971.88673103],
            [0. ,730.62357717,  554.08913202],
            [0., 0., 1.0]
        ])
        # 1080p setting, need to modify into yaml file in the future 1080*1920
        self.cameraMatrix1.append( self.cam1_intrinsic )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        
        self.cameraMatrix2.append( self.cam1_intrinsic )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        self.cameraMatrix1.append( self.cam2_intrinsic )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        
        self.cameraMatrix2.append( self.cam2_intrinsic )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        self.cameraMatrix1.append( self.cam3_intrinsic )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        
        self.cameraMatrix2.append( self.cam3_intrinsic )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        self.imageSize = []

        self.imageSize.append( (1080, 1920) )
        self.imageSize.append( (1080, 1920) )
        self.imageSize.append( (1080, 1920) )

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

        self.trans = []
        self.trans.append( self.get_transform( [-0.336, 0.060, 0.455], [0.653, -0.616, 0.305, -0.317]) ) #rosrun tf tf_echo  map cam1
        self.trans.append( self.get_transform( [0.090, 0.582, 0.449], [-0.037, 0.895, -0.443, 0.031]) )
        self.trans.append( self.get_transform( [0.015, -0.524, 0.448], [0.887, 0.013, 0.001, -0.461]) )

        self.points = []
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgba', 12, PointField.UINT32, 1),
                    ]
        
        #Todo: feed a startup all zero image to the network
        self.cam1_left_sub = message_filters.Subscriber(args.left_topic1, Image)
        # self.cam2_sub = message_filters.Subscriber(args.right_topic, Image)
        self.cam1_depth_sub = message_filters.Subscriber(args.depth_topic1, Image)

        self.cam2_left_sub = message_filters.Subscriber(args.left_topic2, Image)
        # self.cam2_sub = message_filters.Subscriber(args.right_topic, Image)
        self.cam2_depth_sub = message_filters.Subscriber(args.depth_topic2, Image)

        self.cam3_left_sub = message_filters.Subscriber(args.left_topic3, Image)
        # self.cam2_sub = message_filters.Subscriber(args.right_topic, Image)
        self.cam3_depth_sub = message_filters.Subscriber(args.depth_topic3, Image)

        self.object_point_cloud_pub = rospy.Publisher("/object_point_cloud2", PointCloud2, queue_size=1)
        # self.point_cloud_pub1 = rospy.Publisher("cam1/gsa_point_cloud2", PointCloud2, queue_size=1)
        # self.point_cloud_pub2 = rospy.Publisher("cam2/gsa_point_cloud2", PointCloud2, queue_size=1)
        # self.point_cloud_pub3 = rospy.Publisher("cam3/gsa_point_cloud2", PointCloud2, queue_size=1)
        # self.object_depth_pub = rospy.Publisher("zedx/gsa_objects_depth", Image, queue_size=1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_left_sub, self.cam1_depth_sub, self.cam2_left_sub, self.cam2_depth_sub, self.cam3_left_sub, self.cam3_depth_sub], 1000, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def get_transform(self, trans, quat):
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
        t[:3, 3] = trans
        return t

    def transform(self, point, trans):
        # print(trans.shape)
        point = np.array(point)
        point = trans @ point.reshape(-1,1)
        return point[0], point[1], point[2]

    def single_callback(self, cam_id, cam1_msg, depth_msg):
        with torch.no_grad():


            image1 = bridge.imgmsg_to_cv2(cam1_msg) #bgra
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGRA2RGB)
            image1_np = np.array(image1)

            image_depth = bridge.imgmsg_to_cv2(depth_msg)
            image_depth_np = np.array(image_depth)
            
            image_pil, image = self.process_image(image1_np)
            # run grounding dino model
            start = time.time()
            boxes_filt, pred_phrases = self.get_grounding_output(
                self.model, image, self.text_prompt, self.box_threshold, self.text_threshold, device=self.device
            )
            end = time.time()
            print("run grounding dino model time: ", end - start, " s")


            image = image1 # RGB img

            start = time.time()
            self.predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

            masks, _, _ = self.predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(self.device),
                multimask_output = False,
            )
            end = time.time()
            print("SAM time: ", end - start, " s")
            # self.save_mask_data(self.output_dir, masks, boxes_filt, pred_phrases)

            value = 0  # 0 for background
            mask_img = torch.zeros(masks.shape[-2:])
            for idx, mask in enumerate(masks):
                mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
            image_depth_np[ mask_img == 0] = np.nan
            
            # # rgb point cloud, reference : https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
            disp = self.depth_to_disparity(cam_id, image_depth_np)

            image_3d = cv2.reprojectImageTo3D(disp, self.Q[cam_id-1])
            # if(self.args.downsampling == True):
            #     image_3d = cv2.reprojectImageTo3D(disp, self.Q[1])
            # print("mask: ", np.where(mask_img>0)[0].shape  )
            xs , ys = np.where(mask_img>0) 
            
            lim = 8
            for i in range( xs.shape[0] ):
                x = image_3d[ xs[i] ][ ys[i] ][0]
                y = image_3d[ xs[i] ][ ys[i] ][1]
                z = image_3d[ xs[i] ][ ys[i] ][2]

                x, y, z = self.transform([x,y,z,1], self.trans[cam_id-1])

                r = image1_np[ xs[i] ][ ys[i] ][0]
                g = image1_np[ xs[i] ][ ys[i] ][1]
                b = image1_np[ xs[i] ][ ys[i] ][2]
                a = 255
                # print r, g, b, a
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                # print hex(rgb)
                pt = [x, y, z, rgb]
                self.points.append(pt)

            print("finished")

    def callback(self, cam1_msg, depth1_msg, cam2_msg, depth2_msg, cam3_msg, depth3_msg):
        print("callback")
        start = time.time()
            
        # pointcloud1 = self.single_callback(1, cam1_msg, depth1_msg)
        # pointcloud2 = self.single_callback(2, cam2_msg, depth2_msg) 
        # pointcloud3 = self.single_callback(3, cam3_msg, depth3_msg)
        self.single_callback(1, cam1_msg, depth1_msg)
        self.single_callback(2, cam2_msg, depth2_msg)
        self.single_callback(3, cam3_msg, depth3_msg)
        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pc2 = point_cloud2.create_cloud(header, self.fields, self.points)
        self.object_point_cloud_pub.publish(pc2)
        self.points = []
        # header = std_msgs.msg.Header
        # header.stamp = rospy.Time.now()
        
        # pointcloud1.header = header
        # pointcloud2.header = header
        # pointcloud3.header = header
        # print(pointcloud1.header.stamp)
        # print(pointcloud2.header.stamp)
        # print(pointcloud3.header.stamp)
        # self.point_cloud_pub1.publish(pointcloud1)
        # self.point_cloud_pub2.publish(pointcloud2)
        # self.point_cloud_pub3.publish(pointcloud3)   
        end = time.time()
        print("three pointscloud time: ", end - start, " s")

    def run(self):
        rospy.spin()  


    def disparity_to_depth(self, cam_id, disparity):
        focal_length = self.cameraMatrix1[cam_id-1][0][0] * 2.0 /3.0
        
        if(disparity.shape[0] == 1080):
            focal_length = self.cameraMatrix1[cam_id-1][0][0]

        depth = (0.12 * focal_length) / disparity
        # depth_valid =  np.logical_and( np.logical_not(np.isnan(image_depth_np)), np.logical_not(np.isinf(image_depth_np)) )
        return depth

    def depth_to_disparity(self, cam_id, depth):
        focal_length = self.cameraMatrix1[cam_id-1][0][0] * 2.0 /3.0
        
        if(depth.shape[0] == 1080):
            focal_length = self.cameraMatrix1[cam_id-1][0][0]

        disparity = (0.12 * focal_length) / depth
        
        return disparity

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        ax.text(x0, y0, label)

    def save_mask_data(self, output_dir, mask_list, box_list, label_list):
        value = 0  # 0 for background

        mask_img = torch.zeros(mask_list.shape[-2:])
        for idx, mask in enumerate(mask_list):
            mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
            # print( mask.cpu().numpy().shape)
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy())
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        json_data = [{
            'value': value,
            'label': 'background'
        }]
        for label, box in zip(label_list, box_list):
            value += 1
            name, logit = label.split('(')
            logit = logit[:-1] # the last is ')'
            json_data.append({
                'value': value,
                'label': name,
                'logit': float(logit),
                'box': box.numpy().tolist(),
            })
        with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
            json.dump(json_data, f)

    def load_image(self, image_path):
        # load image
        image_pil = PIL_Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def process_image(self, rgb_img):

        image_pil = PIL_Image.fromarray(rgb_img.astype('uint8'), 'RGB')
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image


    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases



def demo(args):
    rospy.init_node("zedx_grounded_sam_node")
    grounded_sam_node = grounded_sam(args)
    grounded_sam_node.run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, default = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", required=False, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, default = "groundingdino_swint_ogc.pth", required=False, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default = "sam_vit_h_4b8939.pth", required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    # parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=False, default="mug", help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="output", help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on GPU, default=False")

    ##############################################
    parser.add_argument('--left_topic1', type=str, default="/cam1/zed_node_A/left/image_rect_color", help="left cam1 topic")
    # parser.add_argument('--right_topic1', type=str, default="/cam1/zed_node_A/right/image_rect_color", help="right cam1 topic")
    parser.add_argument('--depth_topic1', type=str, default="/cam1/zed_node_A/depth/depth_registered", help="depth cam1 topic")

    parser.add_argument('--left_topic2', type=str, default="/cam2/zed_node_B/left/image_rect_color", help="left cam2 topic")
    # parser.add_argument('--right_topic2', type=str, default="/cam2/zed_node_A/right/image_rect_color", help="right cam2 topic")
    parser.add_argument('--depth_topic2', type=str, default="/cam2/zed_node_B/depth/depth_registered", help="depth cam2 topic")

    parser.add_argument('--left_topic3', type=str, default="/cam3/zed_node_C/left/image_rect_color", help="left cam3 topic")
    # parser.add_argument('--right_topic', type=str, default="/cam1/zed_node_A/right/image_rect_color", help="right cam topic")
    parser.add_argument('--depth_topic3', type=str, default="/cam3/zed_node_C/depth/depth_registered", help="depth cam3 topic")

    # parser.add_argument('--conf_map_topic', type=str, default="/cam1/zed_node_A/confidence/confidence_map", help="depth confidence map topic")
    ##############################################


    args = parser.parse_args()

    demo(args)