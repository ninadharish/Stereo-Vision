import cv2
import numpy as np
import matplotlib.pyplot as plt


def SSD(kernel1, kernel2):

    SSD = np.sum(np.square(np.subtract(kernel1, kernel2)))

    return SSD


def matchingWindows(img1, img2):

    h, w = img1.shape

    disp_map = np.zeros((h, w))
    depth = np.zeros((h, w))

    win_size = 30

    for i in range(0, h, win_size):
        for j in range(0, w, win_size):

            kernel1 = img1[i:i+win_size, j:j+win_size]

            SSD_min_val = np.inf
            index = j+0.001

            for k in range((max(0, j-100)), j, 5):

                kernel2 = img2[i:i+win_size, k:k+win_size]

                SSD_val = SSD(kernel1, kernel2)

                if SSD_val <= SSD_min_val:
                    index = k
                    SSD_min_val = SSD_val

            disp = (np.abs(index - j))

            disp_map[i:i+win_size, j:j+win_size] = disp  

    max_SSD = np.amax(disp_map)
    min_SSD = np.amin(disp_map)
    scale = (255/(max_SSD - min_SSD))

    disp_img = np.multiply(disp_map, scale)
    disp_img = np.subtract(disp_img, ((255*min_SSD)/(max_SSD-min_SSD)))
    disp_img = np.uint8(disp_img)

    disp_img_heat = cv2.applyColorMap(disp_img, cv2.COLORMAP_TURBO)
    # cv2.imwrite('img7.jpg', disp_img)
    # cv2.imwrite('img8.jpg', disp_img_heat)

    return disp_map, disp_img