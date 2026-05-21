#!/usr/bin/env python3
import rospy
import torch
import numpy as np

from torchvision import transforms
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from gelsight_ros.msg import Angles2d
from utils.model import AnglePredictor, SimplerModel, SimpleModel

# how do i access the model

bridge = CvBridge()
pub = rospy.Publisher('/gelsight/angles', Angles2d, queue_size=1)

model = SimpleModel()

# model.load_state_dict(torch.load("/home/msrl/data_benjamin/20260501_session/data/exp1/Best Models/best_model_AnglePredictor_Original.pth"))
model.load_state_dict(torch.load("/home/msrl/data_benjamin/20260501_session/data/best_model_simple.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

def callback(data):
    # Convert ROS Image message to OpenCV image
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
    # print("Received image with shape:", cv_image.shape)

    angles = Angles2d()
    with torch.no_grad():
        input_tensor = base_transform(cv_image).to(device)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        # print("Input tensor shape:", input_tensor.shape)

        # Get angle predictions from the model
        output = model(input_tensor)
        # print("Model output:", output)
        angles.angleX = np.rad2deg(output[0, 0].item())
        angles.angleY = np.rad2deg(output[0, 1].item())

    # somehow get the model in here to caluclate the angle

    # angles.angleX = 1.0
    # angles.angleY = 2.0


    pub.publish(angles)

def main():
    rospy.init_node('angle_publisher', anonymous=True) #why anonymous=True?

    rospy.Subscriber('/gelsight/image_raw', Image, callback)

    rospy.spin()

if __name__ == '__main__':
    main()

