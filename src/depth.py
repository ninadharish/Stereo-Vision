import cv2
import numpy as np
import matplotlib.pyplot as plt


def computeDepth(disp_map, cam0, baseline):

    f = cam0[0][0]

    depth_map = np.multiply(1/disp_map, (f*baseline))
    depth_map = np.uint8(depth_map)

    max_depth = np.amax(depth_map)
    min_depth = np.amin(depth_map)
    scale = (255/(max_depth - min_depth))

    depth_img = np.multiply(depth_map, scale)
    depth_img = np.subtract(depth_img, ((255*min_depth)/(max_depth-min_depth)))
    depth_img = np.uint8(depth_img)
    # depth_img = 255 - depth_img

    depth_img_heat = cv2.applyColorMap(depth_img, cv2.COLORMAP_TURBO)
    # cv2.imwrite('img9.jpg', depth_img)
    # cv2.imwrite('img10.jpg', depth_img_heat)

    return depth_map, depth_img