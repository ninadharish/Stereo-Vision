import cv2
import numpy as np
import matplotlib.pyplot as plt
from calibrate import *
from rectify import *
from correspond import *
from depth import *


def stereo(folder):

    img1 = cv2.imread('./data/'+folder+'/im0.png')
    img2 = cv2.imread('./data/'+folder+'/im1.png')

    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    if folder == 'curule':
        cam0 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
        cam1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
        doffs = 0
        baseline = 88.39
        width = 1920
        height = 1080
        ndisp = 220
        vmin = 55
        vmax = 195

    elif folder == 'octagon':
        cam0 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
        cam1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
        doffs = 0
        baseline = 221.76
        width = 1920
        height = 1080
        ndisp = 100
        vmin = 29
        vmax = 61

    elif folder == 'pendulum':
        cam0 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
        cam1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
        doffs = 0
        baseline = 537.75
        width = 1920
        height = 1080
        ndisp = 180
        vmin = 25
        vmax = 150

    src_pts, dst_pts = features_match(img1, img2)

    F = computeFundamentalMatrix(src_pts, dst_pts)
    # F, src_pts, dst_pts = RANSACforFundamentalMatrix(src_pts, dst_pts)

    E = computeEssentialMatrix(F, cam0, cam1)

    print("Fundamental Matrix: F = ")
    print(F)
    print()
    print("Essential Matrix: E = ")
    print(E)
    print()

    decomposeEssentialMatrix(E, cam0, cam1, src_pts[0], dst_pts[0])

    img1_rect, img2_rect, F, src_pts_new, dst_pts_new = rectifyImages(img1, img2, F, src_pts, dst_pts)

    drawEpipolarLines(img1_rect, img2_rect, F, src_pts_new, dst_pts_new)

    disp_map, disp_img = matchingWindows(img1_rect, img2_rect)
    disp_fig = plt.figure()
    disp_fig.add_subplot(1,2,1)
    plt.imshow(disp_img, cmap='gray', interpolation='nearest')
    disp_fig.add_subplot(1,2,2)
    plt.imshow(disp_img, cmap='inferno')
    plt.show()

    depth_map, depth_img = computeDepth(disp_map, cam0, baseline)
    depth_fig = plt.figure()
    depth_fig.add_subplot(1,2,1)
    plt.imshow(depth_img, cmap='gray', interpolation='nearest')
    depth_fig.add_subplot(1,2,2)
    plt.imshow(depth_img, cmap='inferno')
    plt.show()


if __name__ == "__main__":

    # stereo('curule')
    stereo('octagon')
    # stereo('pendulum')