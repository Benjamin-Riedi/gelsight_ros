import cv2
import numpy as np
from gelsight_rgb_compat import detect_circle, crop_to_circle

img = cv2.imread("/home/benjamin/calibration_frame.png")  # BGR image (OpenCV default)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

detected_circle = detect_circle(img)
crop = crop_to_circle(img, detected_circle)

cv2.imshow("Detected Circle", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
