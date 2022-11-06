import cv2
import numpy as np
import matplotlib.pyplot as plt


def features_match(img1, img2):

    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures=500)

    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm = 6, table_numer=6, key_size=12, multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append(m)

    src_pts_temp = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts_temp = np.float32([kp2[m.trainIdx].pt for m in good])

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img3)
    plt.show()
    # cv2.imwrite('img1.jpg', img3)

    return src_pts_temp, dst_pts_temp


def normalizePoints(pts):

    t_x = np.mean(pts, axis=0)[0]
    t_y = np.mean(pts, axis=0)[1]
    s = (2**0.5)/(((1/(len(pts)))*(np.sum(np.sum(np.square(np.subtract(pts, np.mean(pts, axis=0))), axis=1))))**0.5)

    T = np.matmul((np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]])),(np.array([[1, 0, (-t_x)], [0, 1, (-t_y)], [0, 0, 1]])))

    pts_norm = np.matmul(T, ((np.hstack([pts, np.ones((len(pts), 1))])).T))
    pts_norm = ((np.divide(pts_norm, pts_norm[-1]))[0:2]).T

    
    return pts_norm, T


def computeFundamentalMatrix(src_pts, dst_pts):

    src_norm, T_src = normalizePoints(src_pts)
    dst_norm, T_dst = normalizePoints(dst_pts)

    A = []
    for i in range(len(src_norm)):
        A.append([((src_norm[i][0])*(dst_norm[i][0])), ((src_norm[i][0])*(dst_norm[i][1])), (src_norm[i][0]), ((src_norm[i][1])*(dst_norm[i][0])), ((src_norm[i][1])*(dst_norm[i][1])), (src_norm[i][1]), (dst_norm[i][0]), (dst_norm[i][1]), (1)])
    A = np.asarray(A)
    
    _, __, VT_A = np.linalg.svd(A)

    F_norm = np.reshape(VT_A[-1], (3, 3))

    if (np.linalg.matrix_rank(F_norm) == 3):

        U, S, VT = np.linalg.svd(F_norm)
        S[-1] = 0
        F_norm = np.matmul(U * S, VT)

    F = np.matmul(np.matmul((T_dst.T), F_norm), T_src)
    F = np.divide(F, F[-1][-1])

    return F


def checkInliers(src_pts, dst_pts, F, threshold):

    values = np.diagonal(np.matmul((np.matmul((np.hstack([dst_pts, np.ones((len(dst_pts), 1))])), (F))), ((np.hstack([src_pts, np.ones((len(src_pts), 1))])).T)))

    num_inliers = np.count_nonzero(np.abs(values) < threshold)

    bool = values < threshold

    src_pts = src_pts[bool]
    dst_pts = dst_pts[bool]

    return num_inliers, src_pts, dst_pts




def RANSACforFundamentalMatrix(src_pts, dst_pts):

    iter = 0
    max_iter = 2000
    F_threshold = 0.001

    max_inliers = 0

    while (iter < max_iter):

        rand_ind = np.random.choice(src_pts.shape[0], size=20, replace=False)

        F_iter = computeFundamentalMatrix(src_pts[rand_ind], dst_pts[rand_ind])

        iter_inliers, src_pts_iter, dst_pts_iter = checkInliers(src_pts, dst_pts, F_iter, F_threshold)

        print("Iter: ", iter, " Iter_Inliers: ", iter_inliers, " Max_Inliers: ", max_inliers)

        if iter_inliers > max_inliers:
            F = F_iter
            src_pts = src_pts_iter
            dst_pts = dst_pts_iter
            max_inliers = iter_inliers

        iter += 1

    return F, src_pts, dst_pts


def computeEssentialMatrix(F, cam0, cam1):

    E = np.matmul((np.matmul(cam1.T, F)), cam0)

    if (np.linalg.matrix_rank(E) == 3):

        U, S, VT = np.linalg.svd(E)
        S[-1] = 0
        E = np.matmul(U * S, VT)

    return E


def decomposeEssentialMatrix(E, cam0, cam1, src_pts, dst_pts):

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, _, VT = np.linalg.svd(E)

    C1 = C3 = U[:, 2].reshape(3, 1)
    C2 = C4 = -U[:, 2].reshape(3, 1)

    R1 = R2 = np.matmul((np.matmul(U, W)), VT)
    R3 = R4 = np.matmul((np.matmul(U, W.T)), VT)

    if (np.linalg.det(R1) < 0):
        R1 = -R1
        R2 = -R2
        C1 = -C1
        C2 = -C2
    if (np.linalg.det(R3) < 0):
        R3 = -R3
        R4 = -R4
        C3 = -C3
        C4 = -C4

    print('R1 = ')
    print(R1)
    print()
    print('R2 = ')
    print(R2)
    print()
    print('R3 = ')
    print(R3)
    print()
    print('R4 = ')
    print(R4)
    print()
    print('C1 = ')
    print(C1)
    print()
    print('C2 = ')
    print(C2)
    print()
    print('C3 = ')
    print(C3)
    print()
    print('C4 = ')
    print(C4)
    print()