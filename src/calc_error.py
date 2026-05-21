#!/usr/bin/env python3
import rospy
import numpy as np
import message_filters

from gelsight_ros.msg import Angles2d
from std_msgs.msg import Float32

pubx = rospy.Publisher('/angles/errorx', Float32, queue_size=1)
puby = rospy.Publisher('/angles/errory', Float32, queue_size=1)

def callback(gelsight_angles, vicon_angles):
    errorx = Float32()
    errory = Float32()

    errorx.data = gelsight_angles.angleX - vicon_angles.angleX
    errory.data = gelsight_angles.angleY - vicon_angles.angleY

    pubx.publish(errorx)
    puby.publish(errory)

if __name__ == "__main__":
    rospy.init_node('angle_error_publisher', anonymous=True)

    gs_sub = message_filters.Subscriber('/gelsight/angles', Angles2d)
    vicon_sub = message_filters.Subscriber('/vicon/angles', Angles2d)
    ts = message_filters.ApproximateTimeSynchronizer([gs_sub, vicon_sub], queue_size=1, slop=0.01, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.spin()