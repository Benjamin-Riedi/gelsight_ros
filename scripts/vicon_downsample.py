import rospy
import numpy as np
import tf2_ros

from threading import Thread
from control_utils.msg import ScalarStamped
from geometry_msgs.msg import TransformStamped
from python_utils.gen import quaternion_to_normal_vector, angles_from_normal_vector


class ViconAnglePublisher:
    def __init__(self,):
        rospy.init_node('vicon_angle_publisher', anonymous=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        self.top_angle_topic = rospy.get_param('/top/vicon')
        self.bottom_angle_topic = rospy.get_param('/bottom/vicon')

        self.top_pub = rospy.Publisher(self.top_angle_topic, ScalarStamped, queue_size=1)
        self.bottom_pub = rospy.Publisher(self.bottom_angle_topic, ScalarStamped, queue_size=1)
        self.sub_topic = rospy.get_param('/vicon/pendulum')

        self.top_angle = ScalarStamped()
        self.bottom_angle = ScalarStamped()

        self.downsample_factor = rospy.get_param('/vicon/downsample_factor', 10)
        self.hold = 0
        self.r = rospy.Rate(100)

        rospy.Subscriber(self.sub_topic, TransformStamped, self.callback)
    
    def init_variables(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.angle_x = 0.0
        self.angle_y = 0.0

        self.valid = True
        self.pos_threshold = 0.01
        self.angle_threshold = 0.01

    def callback(self, tf_msg):
        # if self.hold <= self.downsample_factor:
        #     self.hold += 1
        #     return
        # else:
        #     self.hold = 0


        try:
            transform = self.tf_buffer.lookup_transform('vicon/world', 'vicon/benjamin_v2/Root',  rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("Failed to lookup transform")

        rotation = transform.transform.rotation
        quat = np.array([[rotation.x, rotation.y, rotation.z, rotation.w]])
        normal_vector = quaternion_to_normal_vector(quat)
        # self.angle_y, self.angle_x = angles_from_normal_vector(-normal_vector)
        self.top_angle.scalar, self.bottom_angle.scalar = angles_from_normal_vector(-normal_vector)
        # self.bottom_angle.scalar, self.top_angle.scalar = angles_from_normal_vector(-normal_vector)
        self.bottom_angle.scalar = -self.bottom_angle.scalar  # flip sign for top angle to match the convention used in the model

        self.top_angle.header.stamp = rospy.Time.now()
        self.bottom_angle.header.stamp = rospy.Time.now()

    def publisher(self):
        while not rospy.is_shutdown():
            self.top_pub.publish(self.top_angle)
            self.bottom_pub.publish(self.bottom_angle)
            self.r.sleep()

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = ViconAnglePublisher()
    Thread(target=node.publisher, daemon=True).start()
    node.run()