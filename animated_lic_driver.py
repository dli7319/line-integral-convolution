from lic import *
import numpy as np

import os
import os.path as path
import subprocess
import cv2
import matplotlib.pyplot as plt

output_folder = "animated_lic"
os.makedirs(path.join(output_folder, "images"), exist_ok=True)

height, width = 280, 300
motion_field = np.zeros((height, width, 2))
for x in range(width):
    for y in range(height):
        delta_vec = np.array([x, y]) - np.array([int(height / 2), int(width / 2)])
        dist = np.linalg.norm(delta_vec)
        if dist < 140:
            speed = np.sin(np.pi * (dist / 140))
            angle = np.arctan2(delta_vec[0], delta_vec[1])
            tangent = [-np.sin(angle), np.cos(angle)]
            motion_field[y, x, 0] = speed * tangent[1]
            motion_field[y, x, 1] = speed * tangent[0]

noise = get_noise(motion_field)
for t in range(1000):
    lic_image = lic_flow(motion_field, t=(t/10), len_pix=10, noise=noise)
    cv2.imwrite(path.join(output_folder, "images", "%d.png" % t), lic_image * 255)

subprocess.run(["ffmpeg",
                "-i", path.join(output_folder, "images", "%d.png"),
                "-r", "30",
                "-y", path.join(output_folder, "animated_lic.mp4")])
