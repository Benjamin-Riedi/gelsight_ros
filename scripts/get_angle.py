#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import rospkg
import os

from torchvision import transforms
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from gelsight_ros.msg import Angles2dStamped
from control_utils.msg import ScalarStamped
from gelsight_ros.utils.model import AnglePredictor, SimplerModel, SimpleModel

# how do i access the model

class GetAngle():
    def __init__(self):
        rospy.init_node('gelsight_angle_publisher', anonymous=True)

        self.init_topics()
        self.bridge = CvBridge()
        self.top_pub = rospy.Publisher(self.top_pub, ScalarStamped, queue_size=1)
        self.bottom_pub = rospy.Publisher(self.bottom_pub, ScalarStamped, queue_size=1)
        self.top_angle = ScalarStamped()
        self.bottom_angle = ScalarStamped()

        self.model = SimpleModel()

        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path('gelsight_ros')

        self.model_path = os.path.join(self.package_path, "src/gelsight_ros/utils/best_model_simple.pth")
        # model.load_state_dict(torch.load("/home/msrl/data_benjamin/20260501_session/data/exp1/Best Models/best_model_AnglePredictor_Original.pth"))
        # self.model.load_state_dict(torch.load("/home/benjamin/data_benjamin/20260501_session/data/best_model_simple.pth"))
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        self.base_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
        
    def init_topics(self):
        self.top_pub = rospy.get_param('/top/gelsight')
        self.bottom_pub = rospy.get_param('/bottom/gelsight')
        self.sub_topic = rospy.get_param('/gelsight/rgb')

    def callback(self, data):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        # print("Received image with shape:", cv_image.shape)

        with torch.no_grad():
            input_tensor = self.base_transform(cv_image).to(self.device)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            # print("Input tensor shape:", input_tensor.shape)

            # Get angle predictions from the model
            output = self.model(input_tensor)
            # print("Model output:", output)
            self.top_angle.header.stamp = rospy.Time.now()
            self.bottom_angle.header.stamp = rospy.Time.now()
            self.bottom_angle.scalar = output[0, 0].item() # i think angle x is bottom, towards the motor
            self.top_angle.scalar = -output[0, 1].item() # and angle y is top, away from motor. (so flip sign because the model is trained with the opposite convention)

        # somehow get the model in here to caluclate the angle

        # self.top_angle.scalar = 1.0
        # self.bottom_angle.scalar = 2.0


        self.top_pub.publish(self.top_angle)
        self.bottom_pub.publish(self.bottom_angle)

    def run(self):
        rospy.Subscriber(self.sub_topic, Image, self.callback)
        rospy.spin()

if __name__ == '__main__':
    node = GetAngle()
    node.run()

