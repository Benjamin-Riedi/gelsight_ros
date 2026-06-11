import rospy
import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

target_size = (320, 240)
pub = rospy.Publisher('/gelsight/resize',Image)
bridge = CvBridge()

def callback(msg):
    image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
    image = cv2.resize(image, target_size)

    pub.publish(bridge.cv2_to_imgmsg(image, encoding='rgb8'))


if __name__ == '__main__':
    rospy.init_node('image_resize', anonymous=True)

    rospy.Subscriber('/gelsight/image_raw', Image, callback)
    rospy.spin()