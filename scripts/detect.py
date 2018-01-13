#!/usr/bin/env python

PKG = 'object_detection'
import numpy as np
import tensorflow as tf
import label_map_util
import visualization_utils as vis_util
import os
import rospkg
import roslib

from matplotlib import pyplot as plt
from PIL import Image

if tf.__version__ != '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

roslib.load_manifest(PKG)

# Find the directory of this script
rp = rospkg.RosPack()
SCRIPT_PATH = os.path.join(rp.get_path("object_detection"), "scripts")

# What model to use.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_PATH = os.path.join(SCRIPT_PATH, 'models', MODEL_NAME)

# Path to frozen detection graph.
# This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(MODEL_PATH, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(SCRIPT_PATH, 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect(image):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                'detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name(
                'detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name(
                'detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # represent detected objects by display name
            labels = []
            for cls in classes[0]:
                for item in label_map.item:
                    if cls == item.id:
                        labels.append(str(item.display_name))

            return boxes, scores, classes, labels, num


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'dataset/test_images'
TEST_IMAGE_PATHS = [os.path.join(SCRIPT_PATH, PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

if __name__ == '__main__':
    for image_path in TEST_IMAGE_PATHS:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        boxes, scores, classes, labels, num = detect(image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                           np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                           category_index, use_normalized_coordinates=True,
                                                           line_thickness=4)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.show()
