#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros

from gelsight_ros.msg import Angles2dStamped
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from python_utils.gen import quaternion_to_normal_vector, angles_from_normal_vector

pub_topic = rospy.get_param('/topics/vicon/angles', '/vicon/angles')

pub = rospy.Publisher(pub_topic, Angles2dStamped, queue_size=1)

def callback(image):

    angles = Angles2dStamped()
    try:
        transform = tf_buffer.lookup_transform('vicon/world', 'vicon/benjamin_v2/Root',  rospy.Time(0))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logerr("Failed to lookup transform")

    rotation = transform.transform.rotation
    quat = np.array([[rotation.x, rotation.y, rotation.z, rotation.w]])
    normal_vector = quaternion_to_normal_vector(quat)
    angles.angleY, angles.angleX = np.rad2deg(angles_from_normal_vector(-normal_vector))
    angles.header.stamp = rospy.Time.now()

    # print(f"Received angles from TF: {angles.angleX}, {angles.angleY}")
    
    pub.publish(angles)


if __name__ == "__main__":

    rospy.init_node('vicon_angle_publisher', anonymous=True)

    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    rospy.Subscriber('/gelsight/image_raw', Image, callback)

    rospy.spin()
