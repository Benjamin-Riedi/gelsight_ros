import os
from utilities.gelsightmini import GelSightMini
from config import default_config

# gs_config = GSConfig(args.gs_config).config
cam_stream = GelSightMini(
            target_width=default_config.camera_width,
            target_height=default_config.camera_height,
            border_fraction=default_config.border_fraction,
)

cam_stream.select_device()
cam_stream.start()
frame = cam_stream.update()