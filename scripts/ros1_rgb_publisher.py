#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from gelsight_ros.utils.gelsight_rgb_compat import GelSightMiniRGBCompat

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Publish GelSight RGB frames to ROS1")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(package_root, "config", "rgb_ros1_noetic_config.json"),
        help="Path to JSON config file",
    )
    args = parser.parse_args(rospy.myargv()[1:])

    cfg = load_config(args.config)

    rospy.init_node(cfg.get("node_name", "gelsight_rgb_publisher"), anonymous=False)

    image_topic = rospy.get_param('/topics/gelsight/rgb', '/gelsight/rgb')
    frame_id = cfg.get("frame_id", "gelsight_rgb_optical_frame")
    publish_rate_hz = float(cfg.get("publish_rate_hz", 25.0))

    cam = GelSightMiniRGBCompat(
        target_width=int(cfg.get("target_width", 640)),
        target_height=int(cfg.get("target_height", 480)),
        border_fraction=float(cfg.get("border_fraction", 0.15)),
        prefer_v4l2=bool(cfg.get("prefer_v4l2", True)),
        backend=cfg.get("backend", None),
        fps=cfg.get("fps", 25.0),
        log_capture_properties=bool(cfg.get("log_capture_properties", True)),
    )

    device = cfg.get("device", None)
    if isinstance(device, str):
        try:
            device = int(device)
        except ValueError:
            pass

    cam.open(device=device)

    pub = rospy.Publisher(image_topic, Image, queue_size=int(cfg.get("queue_size", 1)))
    bridge = CvBridge()

    rospy.loginfo("Publishing GelSight RGB frames on %s", image_topic)

    # rospy.loginfo("Starting calibration. Don't touch the camera during this time...")
    # rospy.sleep(2.0)
    # cal_frames = []
    # while len(cal_frames) < 50 and not rospy.is_shutdown():
    #     cal_frame = cam.read_rgb()
    #     if cal_frame is None:
    #         rospy.logwarn("Received empty frame from camera during calibration. No calibration will be performed.")
    #     else:
    #         cal_frames.append(cal_frame)

    # if not cal_frames:
    #     rospy.logwarn("No valid calibration frames received. No calibration will be performed.")
    # else:
    #     # average the calibration frames to reduce noise
    #     cal_frame = np.mean(np.stack(cal_frames, axis=0), axis=0)
    #     cal_frame = np.clip(cal_frame, 0, 255).astype(np.uint8)

    try:
        while not rospy.is_shutdown():
            t_loop_start = time.perf_counter()

            t0 = time.perf_counter()
            frame_rgb = cam.read_rgb()
            # frame_rgb = frame_rgb - cal_frame if cal_frame is not None else frame_rgb
            t1 = time.perf_counter()
            if frame_rgb is None:
                rospy.logwarn("Received empty frame from camera, skipping publish.")
                continue

            msg = bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
            t2 = time.perf_counter()

            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = frame_id
            t3 = time.perf_counter()

            pub.publish(msg)
            t4 = time.perf_counter()

            rospy.logdebug(
                "loop timings (ms) — read_rgb: %.2f  cv2_to_imgmsg: %.2f  "
                "stamp+frame_id: %.2f  publish: %.2f  rate.sleep: not tracked  "
                "total: %.2f",
                (t1 - t0) * 1e3,
                (t2 - t1) * 1e3,
                (t3 - t2) * 1e3,
                (t4 - t3) * 1e3,
                (t4 - t_loop_start) * 1e3,
            )
    finally:
        cam.release()


if __name__ == "__main__":
    main()
