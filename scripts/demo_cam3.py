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
        self.cam_intrinsic = np.array([
            [731.402099609375, 0., 985.076416015625],
            [0. ,731.402099609375,  558.0054931640625],
            [0., 0., 1.0]
        ])

        # 1080p setting, need to modify into yaml file in the future 1080*1920
        self.cameraMatrix1.append( self.cam_intrinsic )
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )
        
        self.cameraMatrix2.append( self.cam_intrinsic )
        self.distCoeffs2.append( np.array( [0., 0., 0., 0., 0.] ) )

        # 720p setting, need to modify into yaml file in the future 720*1280
        self.cameraMatrix1.append( self.cam_intrinsic * 2.0 / 3.0)
        self.distCoeffs1.append( np.array( [0., 0., 0., 0., 0.] ) )

        self.cameraMatrix2.append( self.cam_intrinsic * 2.0 / 3.0)
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
        # self.cam2_sub = message_filters.Subscriber(args.right_topic, Image)
        self.depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        # self.conf_map_sub = message_filters.Subscriber(args.conf_map_topic, Image)

        self.point_cloud_pub = rospy.Publisher("cam3/gsa_point_cloud2", PointCloud2, queue_size=1)
        # self.object_depth_pub = rospy.Publisher("zedx/gsa_objects_depth", Image, queue_size=1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.cam1_sub, self.depth_sub], 10, 1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def callback(self, cam1_msg, depth_msg):
        print("callback")
        with torch.no_grad():
            # rgb = bgr[...,::-1].copy()
            # bgr = rgb[...,::-1].copy()
            # gbr = rgb[...,[2,0,1]].copy()

            image1 = bridge.imgmsg_to_cv2(cam1_msg) #bgra
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGRA2RGB)
            image1_np = np.array(image1)
        
            # image2 = bridge.imgmsg_to_cv2(cam2_msg) #bgra
            # image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2RGB)
            # image2_np = np.array(image2)

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

            # draw output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                self.show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box, label in zip(boxes_filt, pred_phrases):
                self.show_box(box.numpy(), plt.gca(), label)

            plt.axis('off')
            plt.savefig(
                os.path.join(self.output_dir, "grounded_sam_output.jpg"),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )
            self.save_mask_data(self.output_dir, masks, boxes_filt, pred_phrases)

            value = 0  # 0 for background
            mask_img = torch.zeros(masks.shape[-2:])
            for idx, mask in enumerate(masks):
                mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
            image_depth_np[ mask_img == 0] = np.nan

            # object_depth_msg = ros_numpy.msgify(Image, image_depth_np, encoding='32FC1')
            # object_depth_msg = bridge.cv2_to_imgmsg(image_depth_np, encoding="32FC1")
            # self.object_depth_pub.publish(object_depth_msg)
            
            
            
            # # rgb point cloud, reference : https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
            disp = self.depth_to_disparity(image_depth_np)

            image_3d = cv2.reprojectImageTo3D(disp, self.Q[0])
            # if(self.args.downsampling == True):
            #     image_3d = cv2.reprojectImageTo3D(disp, self.Q[1])
            print("mask: ", np.where(mask_img>0)[0].shape  )
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
            # header.frame_id = "zedx"
            header.frame_id = "cam3"
            pc2 = point_cloud2.create_cloud(header, fields, points)
            pc2.header.stamp = rospy.Time.now()
            self.point_cloud_pub.publish(pc2)

    def run(self):
        rospy.spin()  


    def disparity_to_depth(self, disparity):
        focal_length = self.cam_intrinsic[0][0] * 2.0 /3.0
        
        if(disparity.shape[0] == 1080):
            focal_length = self.cam_intrinsic[0][0]

        depth = (0.12 * focal_length) / disparity
        # depth_valid =  np.logical_and( np.logical_not(np.isnan(image_depth_np)), np.logical_not(np.isinf(image_depth_np)) )
        return depth

    def depth_to_disparity(self, depth):
        focal_length = self.cam_intrinsic[0][0] * 2.0 /3.0
        
        if(depth.shape[0] == 1080):
            focal_length = self.cam_intrinsic[0][0]

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
    rospy.init_node("zedx_grounded_sam_node2")
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
    parser.add_argument("--text_prompt", type=str, required=False, default="mug. microwave door", help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="output", help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cuda", help="running on GPU, default=False")

    ##############################################
    parser.add_argument('--left_topic', type=str, default="/cam3/zed_node_C/left/image_rect_color", help="left cam topic")
    parser.add_argument('--right_topic', type=str, default="/cam3/zed_node_C/right/image_rect_color", help="right cam topic")
    parser.add_argument('--depth_topic', type=str, default="/cam3/zed_node_C/depth/depth_registered", help="depth cam topic")
    parser.add_argument('--conf_map_topic', type=str, default="/cam3/zed_node_C/confidence/confidence_map", help="depth confidence map topic")
    ##############################################


    args = parser.parse_args()

    demo(args)
