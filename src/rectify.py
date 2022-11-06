import cv2
import numpy as np
import matplotlib.pyplot as plt


def rectifyImages(img1, img2, F, src_pts, dst_pts):

    h, w = img2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, imgSize=(w, h), threshold=0)

    print("Homography Matrix for Left Image: H1 = ")
    print(H1)
    print()
    print("Homography Matrix for Right Image: H2 = ")
    print(H2)

    img1_rect = cv2.warpPerspective(img1, H1, (w, h))
    img2_rect = cv2.warpPerspective(img2, H2, (w, h))

    F_new = np.matmul(((np.linalg.inv(H2)).T), (np.matmul(F, np.linalg.inv(H1))))

    src_new_pts = (np.matmul(H1, ((np.hstack((src_pts, np.ones((len(src_pts), 1))))).T)))
    src_new_pts = np.divide(src_new_pts, src_new_pts[-1])
    src_new_pts = (src_new_pts[0:2]).T
    
    dst_new_pts = (np.matmul(H2, ((np.hstack((dst_pts, np.ones((len(dst_pts), 1))))).T)))
    dst_new_pts = np.divide(dst_new_pts, dst_new_pts[-1])
    dst_new_pts = (dst_new_pts[0:2]).T

    return img1_rect, img2_rect, F_new, src_new_pts, dst_new_pts


def drawlines(img1,img2,lines,pts1,pts2):

    r,c = img1.shape

    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    for r,pt1,pt2 in zip(lines,pts1,pts2):

        # color = tuple(np.random.randint(0, 255, 3).tolist())
        color = (255, 0, 0)
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, (round(pt1[0]), round(pt1[1])), 5, color, -1)
        img2 = cv2.circle(img2, (round(pt2[0]), round(pt2[1])), 5, color, -1)

    return img1, img2


def drawEpipolarLines(img1_rect, img2_rect, F, src_pts, dst_pts):

    lines1 = cv2.computeCorrespondEpilines(dst_pts.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img3, _ = drawlines(img1_rect, img2_rect, lines1, src_pts, dst_pts)

    lines2 = cv2.computeCorrespondEpilines(src_pts.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    img4, _ = drawlines(img2_rect, img1_rect, lines2, dst_pts, src_pts)

    img5 = np.concatenate((img3, img4), axis=1)
    # cv2.imwrite('img5.jpg', img5)

    plt.subplot(121),plt.imshow(img3)
    plt.subplot(122),plt.imshow(img4)
    plt.show()