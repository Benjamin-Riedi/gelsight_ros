#!/usr/bin/env python3
import argparse
import json
import os
import time

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from utils import GelSightMiniRGBCompat


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Publish GelSight RGB frames to ROS1")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(package_root, "utils", "rgb_ros1_noetic_config.json"),
        help="Path to JSON config file",
    )
    args = parser.parse_args(rospy.myargv()[1:])

    cfg = load_config(args.config)

    rospy.init_node(cfg.get("node_name", "gelsight_rgb_publisher"), anonymous=False)

    image_topic = cfg.get("image_topic", "/gelsight/image_raw")
    frame_id = cfg.get("frame_id", "gelsight_rgb_optical_frame")
    publish_rate_hz = float(cfg.get("publish_rate_hz", 25.0))

    cam = GelSightMiniRGBCompat(
        target_width=int(cfg.get("target_width", 640)),
        target_height=int(cfg.get("target_height", 480)),
        border_fraction=float(cfg.get("border_fraction", 0.15)),
        prefer_v4l2=bool(cfg.get("prefer_v4l2", True)),
    )

    device = cfg.get("device", None)
    if isinstance(device, str):
        try:
            device = int(device)
        except ValueError:
            pass

    cam.open(device=device)

    pub = rospy.Publisher(image_topic, Image, queue_size=int(cfg.get("queue_size", 10)))
    bridge = CvBridge()
    rate = rospy.Rate(publish_rate_hz)

    rospy.loginfo("Publishing GelSight RGB frames on %s", image_topic)

    try:
        while not rospy.is_shutdown():
            t_loop_start = time.perf_counter()

            t0 = time.perf_counter()
            frame_rgb = cam.read_rgb()
            t1 = time.perf_counter()

            msg = bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
            t2 = time.perf_counter()

            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = frame_id
            t3 = time.perf_counter()

            pub.publish(msg)
            t4 = time.perf_counter()

            rate.sleep()
            t5 = time.perf_counter()

            rospy.logdebug(
                "loop timings (ms) — read_rgb: %.2f  cv2_to_imgmsg: %.2f  "
                "stamp+frame_id: %.2f  publish: %.2f  rate.sleep: %.2f  "
                "total: %.2f",
                (t1 - t0) * 1e3,
                (t2 - t1) * 1e3,
                (t3 - t2) * 1e3,
                (t4 - t3) * 1e3,
                (t5 - t4) * 1e3,
                (t5 - t_loop_start) * 1e3,
            )
    finally:
        cam.release()


if __name__ == "__main__":
    main()
