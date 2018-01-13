rosparam set usb_cam/pixel_format yuyv
rosrun usb_cam usb_cam_node _video_device:=/dev/video0
rosrun image_view image_view image:=/usb_cam/image_raw
rosrun object_detection object_detection_talker.py
rosrun object_detection object_detection_listener.py
