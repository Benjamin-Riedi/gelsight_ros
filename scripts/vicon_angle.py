#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image
from control_utils.msg import ScalarStamped
from geometry_msgs.msg import TransformStamped
from python_utils.gen import quaternion_to_normal_vector, angles_from_normal_vector

class ViconAnglePublisher:
    def __init__(self):
        rospy.init_node('vicon_angle_publisher', anonymous=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.top_angle_topic = rospy.get_param('/top/vicon')
        self.bottom_angle_topic = rospy.get_param('/bottom/vicon')

        self.top_pub = rospy.Publisher(self.top_angle_topic, ScalarStamped, queue_size=1)
        self.bottom_pub = rospy.Publisher(self.bottom_angle_topic, ScalarStamped, queue_size=1)

    def callback(self, image):
        self.top_angle = ScalarStamped()
        self.bottom_angle = ScalarStamped()

        try:
            transform = self.tf_buffer.lookup_transform('vicon/world', 'vicon/benjamin_v2/Root',  rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("Failed to lookup transform")

        rotation = transform.transform.rotation
        quat = np.array([[rotation.x, rotation.y, rotation.z, rotation.w]])
        normal_vector = quaternion_to_normal_vector(quat)
        self.top_angle.scalar, self.bottom_angle.scalar = np.rad2deg(angles_from_normal_vector(-normal_vector))
        self.top_angle.header.stamp = rospy.Time.now()
        self.bottom_angle.header.stamp = rospy.Time.now()

        # print(f"Received angles from TF: {angles.angleX}, {angles.angleY}")
        
        self.top_pub.publish(self.top_angle)
        self.bottom_pub.publish(self.bottom_angle)

    def run(self):
        rospy.Subscriber('/gelsight/rgb', Image, self.callback)
        rospy.spin()


if __name__ == "__main__":
    node = ViconAnglePublisher()
    node.run()
