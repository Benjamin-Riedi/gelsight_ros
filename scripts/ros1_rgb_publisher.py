#!/usr/bin/env python3
import argparse
import json
import os
import time
import numpy as np
import cv2

import rospy
from threading import Event, Thread
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse

from gelsight_ros.utils.gelsight_rgb_compat import GelSightMiniRGBCompat, detect_circle, crop_to_circle

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

class GelSightRGBPublisher:
    def __init__(self):
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

        self.cam = GelSightMiniRGBCompat(
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

        self.cam.open(device=device)

        rospy.init_node(cfg.get("node_name", "gelsight_rgb_publisher"), anonymous=False)

        self.calib_srv = rospy.Service(
            '/sensor/start',
            Trigger,
            self.calibrate_cb
        )

        image_topic = rospy.get_param('/topics/gelsight/rgb', '/gelsight/rgb')
        self.frame_id = cfg.get("frame_id", "gelsight_rgb_optical_frame")
        publish_rate_hz = float(cfg.get("publish_rate_hz", 25.0))
        self.pub = rospy.Publisher(image_topic, Image, queue_size=int(cfg.get("queue_size", 1)))
        self.bridge = CvBridge()

        self.calibrated = Event()

        rospy.loginfo("Publishing GelSight RGB frames on %s", image_topic)
    
    def calibrate(self):
        rospy.loginfo("Starting calibration. Place the pendulum on the sensor...")
        rospy.sleep(2.0)
        frame_rgb = self.cam.read_rgb()
        cv2.imwrite("/home/benjamin/calibration_frame.png", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow("Calibration Frame", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.detected_circle = detect_circle(frame_rgb)

    def calibrate_cb(self, req):
        self.calibrate()
        self.calibrated.set()
        return TriggerResponse(success=True, message="Calibration complete.")

    def run(self):
        self.calibrated.wait()
        try:
            while not rospy.is_shutdown():
                t_loop_start = time.perf_counter()

                t0 = time.perf_counter()
                frame_rgb = self.cam.read_rgb()
                # frame_rgb = frame_rgb - cal_frame if cal_frame is not None else frame_rgb
                t1 = time.perf_counter()
                if frame_rgb is None:
                    rospy.logwarn("Received empty frame from camera, skipping publish.")
                    continue

                # crop the frame to the detected circle
                crop = crop_to_circle(frame_rgb, self.detected_circle)
                msg = self.bridge.cv2_to_imgmsg(crop, encoding="rgb8")

                # msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
                t2 = time.perf_counter()

                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = self.frame_id
                t3 = time.perf_counter()

                self.pub.publish(msg)
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
            self.cam.release()


if __name__ == "__main__":
    node = GelSightRGBPublisher()
    Thread(target=node.run, daemon=True).start()
    rospy.spin()
