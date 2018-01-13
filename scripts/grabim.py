#!/usr/bin/env python

#PKG = 'object_detection'
#import roslib; roslib.load_manifest(PKG)
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageReceiver:
    def __init__(self):
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image = None
        self.bridge = CvBridge()

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        #np_arr = np.fromstring(ros_data.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.image = cv_image
        # print 'received image of type: "%s"' % ros_data.format


def talker():
    #objects_pub = rospy.Publisher('object_detection/objects', numpy_msg(DetectedObjects), queue_size=10)
    rospy.init_node('object_detection_talker', anonymous=True)
    r = rospy.Rate(1)  # 1hz

    ir = ImageReceiver()
    image_number = 0
    while not rospy.is_shutdown():
        img = ir.image
        if img is not None:
            cv2.imshow('image', img)
        k = cv2.waitKey(20)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            #cv2.imwrite('/home/mjazadix01/catkin_ws/src/object_detection/scripts/dataset/images/_image{}.jpg'.format(image_number), img)
            image_number += 1
        r.sleep()


if __name__ == '__main__':
    talker()
