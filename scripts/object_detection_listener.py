#!/usr/bin/env python

import roslib
import rospy

from rospy.numpy_msg import numpy_msg
from object_detection.msg import Strings
from std_msgs.msg import Bool, Int32

PKG = 'object_detection'
roslib.load_manifest(PKG)


def objects_callback(data):
    for key in data.keys():
        print rospy.get_name(), "I heard: objects I'm looking at -> {}".format(str(data[key]))


def listener():
    rospy.init_node('object_detection_listener')
    rospy.Subscriber("object_detection/objects", numpy_msg(Strings), objects_callback)
    rospy.spin()


if __name__ == '__main__':
    listener()
