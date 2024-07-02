import argparse
import os
import sys

import numpy as np
import json
import torch
import PIL.Image as PIL_Image

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

###################################### IGEV stuff

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

from igev_stereo import IGEVStereo
from utils.utils import InputPadder


class igev_grounded_sam():
    def __init__(self, args):

        self.args = args
            
        self.igev_model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
        self.igev_model.load_state_dict(torch.load(args.restore_ckpt))

        self.igev_model = self.igev_model.module
        self.igev_model.to(DEVICE)
        self.igev_model.eval()

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


        #Todo: feed a startup all zero image to the network
        self.cam1_left_sub = message_filters.Subscriber(args.left_topic1, Image)
        self.cam1_right_sub = message_filters.Subscriber(args.right_topic1, Image)
        self.cam1_depth_sub = message_filters.Subscriber(args.depth_topic1, Image)

        self.cam2_left_sub = message_filters.Subscriber(args.left_topic2, Image)
        self.cam2_right_sub = message_filters.Subscriber(args.right_topic2, Image)
        self.cam2_depth_sub = message_filters.Subscriber(args.depth_topic2, Image)

        self.cam3_left_sub = message_filters.Subscriber(args.left_topic3, Image)
        self.cam3_right_sub = message_filters.Subscriber(args.right_topic3, Image)
        self.cam3_depth_sub = message_filters.Subscriber(args.depth_topic3, Image)

        self.point_cloud_pub1 = rospy.Publisher("cam1/gsa_point_cloud2", PointCloud2, queue_size=1)
        self.point_cloud_pub2 = rospy.Publisher("cam2/gsa_point_cloud2", PointCloud2, queue_size=1)
        self.point_cloud_pub3 = rospy.Publisher("cam3/gsa_point_cloud2", PointCloud2, queue_size=1)
        # self.object_depth_pub = rospy.Publisher("zedx/gsa_objects_depth", Image, queue_size=1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_left_sub, self.cam1_right_sub, self.cam1_depth_sub, self.cam2_left_sub, self.cam2_right_sub, self.cam2_depth_sub, self.cam3_left_sub, self.cam3_right_sub, self.cam3_depth_sub], 1000, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def igev_callback(self, cam_id, image1, image2, image_depth):
        with torch.no_grad():

            image1_np = np.array(image1[:,:,0:3])
            image2_np = np.array(image2[:,:,0:3])
            image_depth_np = np.array(image_depth)

            depth_valid =  np.logical_and( np.logical_not(np.isnan(image_depth_np)), np.logical_not(np.isinf(image_depth_np)) )
            # print("image1.shape: ", image1_np.shape)
            # print("image2.shape: ", image2_np.shape)
            # cv2.imwrite('img1.png', image1_np)
            # cv2.imwrite('img2.png', image2_np)
            # preprocess FOR igev
            image1= self.igev_load_image(image1_np)
            image2= self.igev_load_image(image2_np)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            start = time.time()
            igev_disp = self.igev_model(image1, image2, iters=args.valid_iters, test_mode=True)
            end = time.time()
            print("torch inference time: ", end - start)
            igev_disp = igev_disp.cpu().numpy()
            igev_disp = padder.unpad(igev_disp)
            igev_disp = igev_disp.squeeze()
            # print("igev_disp: ", igev_disp.shape)

            # default_disp = self.depth_to_disparity(cam_id, image_depth_np )
            # default_disp = np.float32( default_disp )
            # default_disp[image_depth_np]
            # print("default_disp: ", default_disp.shape)
            igev_disp = np.float32( igev_disp )
            igev_depth = self.disparity_to_depth(cam_id, igev_disp)
            return igev_depth


    def single_callback(self, cam_id, image1, image_depth):
        with torch.no_grad():
            image1_np = np.array(image1)
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
            points = []
            lim = 8
            for i in range( xs.shape[0] ):
                x = image_3d[ xs[i] ][ ys[i] ][0]
                y = image_3d[ xs[i] ][ ys[i] ][1]
                z = image_3d[ xs[i] ][ ys[i] ][2]
                r = image1_np[ xs[i] ][ ys[i] ][0]
                g = image1_np[ xs[i] ][ ys[i] ][1]
                b = image1_np[ xs[i] ][ ys[i] ][2]
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
            header.stamp = rospy.Time.now()
            # header.frame_id = "zedx"
            header.frame_id = "cam" + str(cam_id)
            pc2 = point_cloud2.create_cloud(header, fields, points)
            return pc2

    def callback(self, left1_msg, right1_msg, depth1_msg, left2_msg, right2_msg, depth2_msg, left3_msg, right3_msg, depth3_msg):
        print("callback")
        start = time.time()

        left1 = bridge.imgmsg_to_cv2(left1_msg) #bgra
        left1 = cv2.cvtColor(left1, cv2.COLOR_BGRA2RGB)
        right1 = bridge.imgmsg_to_cv2(right1_msg) #bgra
        right1 = cv2.cvtColor(right1, cv2.COLOR_BGRA2RGB)
        depth1 = bridge.imgmsg_to_cv2(depth1_msg)

        left2 = bridge.imgmsg_to_cv2(left2_msg) #bgra
        left2 = cv2.cvtColor(left2, cv2.COLOR_BGRA2RGB)
        right2 = bridge.imgmsg_to_cv2(right2_msg) #bgra
        right2 = cv2.cvtColor(right2, cv2.COLOR_BGRA2RGB)
        depth2 = bridge.imgmsg_to_cv2(depth2_msg)

        left3 = bridge.imgmsg_to_cv2(left3_msg) #bgra
        left3 = cv2.cvtColor(left3, cv2.COLOR_BGRA2RGB)
        right3 = bridge.imgmsg_to_cv2(right3_msg) #bgra
        right3 = cv2.cvtColor(right3, cv2.COLOR_BGRA2RGB)
        depth3 = bridge.imgmsg_to_cv2(depth3_msg)
        
        igev_depth1 = self.igev_callback(1, left1, right1, depth1)
        igev_depth2 = self.igev_callback(2, left2, right2, depth2)
        igev_depth3 = self.igev_callback(3, left3, right3, depth3)
        print("IGEV_finshed")
        print("depth1: ", depth1.shape)
        print("igev_depth1: ", igev_depth1.shape)
        pointcloud1 = self.single_callback(1, left1, igev_depth1)
        pointcloud2 = self.single_callback(2, left2, igev_depth2) 
        pointcloud3 = self.single_callback(3, left3, igev_depth3)
        
        # header = std_msgs.msg.Header
        # header.stamp = rospy.Time.now()
        
        # pointcloud1.header = header
        # pointcloud2.header = header
        # pointcloud3.header = header
        # print(pointcloud1.header.stamp)
        # print(pointcloud2.header.stamp)
        # print(pointcloud3.header.stamp)
        self.point_cloud_pub1.publish(pointcloud1)
        self.point_cloud_pub2.publish(pointcloud2)
        self.point_cloud_pub3.publish(pointcloud3)   
        end = time.time()
        print("three pointscloud time: ", end - start, " s")

    def run(self):
        rospy.spin()  

    def igev_load_image(self, img):
        img = img.astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)

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
    rospy.init_node("zedx_igev_grounded_sam_node")
    igev_grounded_sam_node = igev_grounded_sam(args)
    igev_grounded_sam_node.run()

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
    parser.add_argument("--text_prompt", type=str, required=False, default="mug. microwave", help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="output", help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on GPU, default=False")

    ##############################################
    parser.add_argument('--left_topic1', type=str, default="/cam1/zed_node_A/left/image_rect_color", help="left cam1 topic")
    parser.add_argument('--right_topic1', type=str, default="/cam1/zed_node_A/right/image_rect_color", help="right cam1 topic")
    parser.add_argument('--depth_topic1', type=str, default="/cam1/zed_node_A/depth/depth_registered", help="depth cam1 topic")

    parser.add_argument('--left_topic2', type=str, default="/cam2/zed_node_B/left/image_rect_color", help="left cam2 topic")
    parser.add_argument('--right_topic2', type=str, default="/cam2/zed_node_B/right/image_rect_color", help="right cam2 topic")
    parser.add_argument('--depth_topic2', type=str, default="/cam2/zed_node_B/depth/depth_registered", help="depth cam2 topic")

    parser.add_argument('--left_topic3', type=str, default="/cam3/zed_node_C/left/image_rect_color", help="left cam3 topic")
    parser.add_argument('--right_topic3', type=str, default="/cam3/zed_node_C/right/image_rect_color", help="right cam3 topic")
    parser.add_argument('--depth_topic3', type=str, default="/cam3/zed_node_C/depth/depth_registered", help="depth cam3 topic")


    ############################################## IGEV STUFF 
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


    # parser.add_argument('--left_topic', type=str, default="/cam1/zSed_node_A/left/image_rect_color", help="left cam topic")
    # parser.add_argument('--right_topic', type=str, default="/cam1/zed_node_A/right/image_rect_color", help="right cam topic")
    # parser.add_argument('--depth_topic', type=str, default="/cam1/zed_node_A/depth/depth_registered", help="depth cam topic")
    # parser.add_argument('--conf_map_topic', type=str, default="/cam1/zed_node_A/confidence/confidence_map", help="depth confidence map topic")

    # parser.add_argument('--downsampling', type=bool, default=False, help="downsampling image dimension")
    # parser.add_argument('--downsampling', type=bool, default=False, help="downsampling image dimension")

    args = parser.parse_args()

    demo(args)
