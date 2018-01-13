#!/usr/bin/env python

#PKG = 'object_detection'
#import roslib; roslib.load_manifest(PKG)
import rospy
import detect
import numpy as np
import Image
import cv2

from rospy.numpy_msg import numpy_msg
from object_detection.msg import DetectedObjects
# Ros Messages
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32


class ImageReceiver:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.callback, queue_size=1)
        self.image = Image.new("RGB", (512, 512), "white")

    def callback(self, ros_data):
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.image = image_np
        # print 'received image of type: "%s"' % ros_data.format


def talker():
    objects_pub = rospy.Publisher('object_detection/objects', numpy_msg(DetectedObjects), queue_size=10)
    rospy.init_node('object_detection_talker', anonymous=True)
    r = rospy.Rate(1)  # 1hz

    ir = ImageReceiver()

    while not rospy.is_shutdown():
        if objects_pub.get_num_connections() > 0:
            boxes, scores, classes, labels, num = detect.detect(ir.image)
            message = DetectedObjects()
            message.minx = boxes[0]
            message.miny = boxes[1]
            message.width = boxes[2]
            message.height = boxes[3]
            message.scores = scores
            message.labels = labels
            objects_pub.publish(message)
        r.sleep()


if __name__ == '__main__':
    talker()
