"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import helper as hlp
import math
import scipy.signal as sig

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # N = number of points
    N = len(pts1)
    ## normalise
    # get centroid
    cx1,cy1 = np.sum(pts1[:,0])/N, np.sum(pts1[:,1])/N
    # write the transformation matrix
    trans1 = np.array([[1/M, 0, -cx1/M], [0,1/M,-cy1/M], [0,0,1]])
    # trans1 = np.array([[1/M, 0, 0], [0,1/M,0], [0,0,1]])
    # convert [[x1, y1], [x2, y2]...] to [[x1, y1, 1], [x2, y2, 1]...]
    pts1c = np.hstack((pts1, np.ones((N, 1))))
    # transform the matrix T * [x,y,1]t = Xn, this can be extended to mutliple points by taking transform
    npts1 = (trans1 @ pts1c.T).T
    # get the normalised x and y coord array
    x,y = npts1[:, 0].T, npts1[:, 1].T

    # do the same for the other image pts
    cx2,cy2 = np.sum(pts2[:,0])/N, np.sum(pts2[:,1])/N
    trans2 = np.array([[1/M, 0, -cx2/M], [0,1/M,-cy2/M], [0,0,1]])
    # trans2 = np.array([[1/M, 0, 0], [0,1/M,0], [0,0,1]])
    pts2c = np.hstack((pts2, np.ones((N, 1))))
    npts2 = (trans2 @ pts2c.T).T
    xd, yd = npts2[:, 0].T, npts2[:, 1].T

    # Ax = 0, x = eleemnts of fundamental matrix
    A = np.zeros((9, N))
    A[0] = x*xd
    A[1] = x*yd
    A[2] = x
    A[3] = xd*y
    A[4] = y*yd
    A[5] = y
    A[6] = xd
    A[7] = yd
    A[8] = np.ones_like(xd)
    A = A.T

    # Solve for f using SVD
    f = np.linalg.svd(A)[2][-1].reshape(3, 3) ### recheck
    
    u, s, vt = np.linalg.svd(f)
    s = np.vstack((np.diag(s)[:2], np.array([0,0,0]))) # np isnt letting me directly change the value of 3rd row
    fp = u @ s @ vt
    fpref = hlp.refineF(fp, npts1[:, :2], npts2[:,:2])

    funnorm = trans2.T @ fpref @ trans1

    return funnorm


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    w=5
    def calc_delta(im1, im2, pt2, pt1):
        x1, y1 = pt1
        x2, y2 = pt2
        win1 = im1[y1-w:y1+w+1, x1-w:x1+w+1]
        win2 = im2[y2-w:y2+w+1, x2-w:x2+w+1]
        return np.sum(np.sum((win1 - win2)**2, axis=2)**2)
    # x't F x = 0, [x y 1] M = 0
    pts2 = np.zeros_like(pts1)
    for ind, pt in enumerate(pts1):
        M = F @ np.array([pt[0], pt[1], 1]).T.astype('float64')
        M = M.ravel() # make m linear
        m, c = -M[0]/M[1], -M[2]/M[1]
        # y = mx + c
        max_pt = (0,0)
        min_delta = float("inf")
        for x2 in range(w,im1.shape[1]-w):
            y = m * x2 + c
            yf = math.floor(y)
            yc = math.ceil(y)
            if yf > w and yf < im1.shape[0]-w:
                delta = calc_delta(im1, im2, (x2, yf), pt)
                if delta < min_delta:
                    min_delta = delta
                    max_pt = (x2,yf)
            # considering this is giving worse off results
            # if yc > w and yc < im1.shape[0]-w:
            #     delta = calc_delta(im1, im2, (x2, yc), pt)
            #     if delta < min_delta:
            #         min_delta = delta
            #         max_pt = (x2,yc)
        pts2[ind] = np.array(max_pt)
    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return K2.T @ F @ K1

"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    N = len(pts1)
    pts3d = np.zeros((N,3))
    for ind in range(N):
        x1,y1 = pts1[ind]
        x2, y2 = pts2[ind]
        A = np.array([
            y1*P1[2] - P1[1],
            P1[0] - x1*P1[2],
            y2*P2[2] - P2[1],
            P2[0] - x2*P2[2]
        ])
        # use svd
        soln = np.linalg.svd(A)[2][-1]
        pts3d[ind] = soln[:3]/soln[3] # un homogenise the coordinates and add it

    # calculating reprojection errors
    def trans3t2(cam, pts3d):
        proper_form = np.hstack((pts3d, np.ones((N,1)))).T
        pts_new = cam @ proper_form
        pts_new = pts_new/pts_new[2]
        return pts_new[:2].T
    pts1_new = trans3t2(P1, pts3d)
    err = 0
    for i in range(N):
        err += math.sqrt(np.sum((pts1_new[i] - pts1[i])**2))
    return pts3d, err


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    c1 = -np.linalg.inv(K1 @ R1) @ K1 @ t1
    c2 = -np.linalg.inv(K2 @ R2) @ K2 @ t2

    r1 = (c1-c2)/np.sum((c1-c2)**2)
    r2 = np.cross(R1[2,:] ,r1.T).T
    # r2 = np.array([0,0,0]).T
    r3 = np.cross(r2.T,r1.T).T
    # r3 = np.array([0,0,0]).T
    Rn = np.array([r1, r2, r3]).reshape(3,3).T
    t1n = -Rn @ c1
    t2n = -Rn @ c2
    M1 = K2 @ Rn @ np.linalg.inv(K1 @ R1)
    M2 = K2 @ Rn @ np.linalg.inv(K2 @ R2)
    return M1, M2, K2, K2, Rn, Rn, t1n, t2n

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    best_disparity_map = np.zeros_like(im1)
    min_dist = None

    for d in range(max_disp):
        shift_img = np.roll(im2, -d, axis=1)
        diff = (shift_img - im1)**2
        dist_map = sig.convolve2d(diff, np.ones((win_size,win_size)), mode="same", boundary="fill", fillvalue=0)
        if min_dist is None:
            min_dist = dist_map # best_disparity_map is already set to zeroes
        else:
            best_disparity_map = np.where(dist_map < min_dist, d, best_disparity_map)
            min_dist = np.minimum(dist_map, best_disparity_map)
    return best_disparity_map

"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    c1 = -np.linalg.inv(K1 @ R1) @ K1 @ t1
    c2 = -np.linalg.inv(K2 @ R2) @ K2 @ t2
    b = math.sqrt(np.sum((c1-c2)**2))
    f = K1[1,1]
    depth_map = np.where(dispM == 0, 0, b*f/dispM)
    return depth_map