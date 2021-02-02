import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco.coco as coco

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "onmrcnn_v1.h5")

# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # NUM_CLASSES = 2
    NUM_CLASSES = 2


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


#
# # Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#
# # Run detection
# results = model.detect([image], verbose=1)
#
# # Visualize results
# r = results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
#
#
#


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


fn_vid = "C:/_koray/test_data/highway/highway_1600.mp4"
# fn_vid = "C:/_koray/test_data/vid_short.mp4"
# fn_vid = "C:/_koray/test_data/aerial/mexico.mp4"

import cv2

# Video capture
vcapture = cv2.VideoCapture(fn_vid)

success = True
colors = visualize.random_colors(len(class_names))
while success:
    # Read next image
    success, image = vcapture.read()
    if success:
        # OpenCV returns images as BGR, convert to RGB
        image = image[..., ::-1]
        # Detect objects
        r = model.detect([image], verbose=0)[0]

        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             class_names, r['scores'])

        H, W = image.shape[:2]
        boxes, masks, scores, class_ids = r['rois'], r['masks'], r['scores'], r['class_ids']
        masked_image = image.astype(np.uint32).copy()
        for i in range(len(boxes)):
            box = boxes[i]
            # mask = masks[i]
            score = scores[i]
            class_id = class_ids[i]
            color = colors[class_id]

            mask = masks[:, :, i]
            masked_image = visualize.apply_mask(masked_image, mask, color)

            # (startX, startY, endX, endY) = box.astype("int")
            # boxW = endX - startX
            # boxH = endY - startY
            #
            # mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
            # mask = (mask > 100)
            #
            # roi = image.copy()[startY:endY, startX:endX]
            # visMask = (mask * 255).astype("uint8")
            # instance = cv2.bitwise_and(roi, roi, mask=visMask)

            # cv2.imshow("ROI", roi)
            # cv2.imshow("Segmented", instance)

        masked_image = masked_image.astype(np.uint8)
        masked_image = masked_image[..., ::-1]
        cv2.imshow("Mask", masked_image)
        cv2.waitKey(1)

        # # Color splash
        # splash = color_splash(image, r['masks'])
        # # RGB -> BGR to save image to video
        # splash = splash[..., ::-1]
        # # Add image to video writer
        # cv2.imshow("aaa", splash)
        # cv2.waitKey(1)
        #
        # count += 1
