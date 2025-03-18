import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

# 1. Load the two temple images and the points from data/some_corresp.npz
pt_corr = np.load('../data/some_corresp.npz')
im1 = io.imread('../data/im1.png')
im2 = io.imread('../data/im2.png')

# 2. Run eight_point to compute F
f = sub.eight_point(pt_corr['pts1'], pt_corr['pts2'], max(im1.shape))
# hlp.displayEpipolarF(im1, im2, f)

# 3. Load points in image 1 from data/temple_coords.npz

pts1 = np.load('../data/temple_coords.npz')['pts1']

# 4. Run epipolar_correspondences to get points in image 2

pts2 = sub.epipolar_correspondences(im1, im2, f, pts1)

# 5. Compute the camera projection matrix P1

instrinsics = np.load('../data/intrinsics.npz')
K1 = instrinsics['K1']
Extr1 = np.hstack((np.identity(3), np.zeros((3,1))))
P1 = K1 @ Extr1

# 6. Use camera2 to get 4 camera projection matrices P2
K2 = instrinsics['K2']
E = sub.essential_matrix(f, K1, K2)
Extr2s = hlp.camera2(E)

# 7. Run triangulate using the projection matrices

def check_ahead(pts3d, Extr):
    num = 0
    for pt in pts3d:
        pt = pt.T
        trans_pt = pt - Extr[:,3]
        trans_pt = np.linalg.inv(Extr[:,:3]) @ pt
        if trans_pt[2] > 0:
            num += 1
    return 0

num_ahead = []
pts_set = []
errs = []
for index in range(4):
    Extr2 = Extr2s[:,:,index]
    P2 = K2 @ Extr2
    pts3d, err = sub.triangulate(P1, pts1, P2, pts2)
    pts_set.append(pts3d)
    errs.append(err)
    num_ahead.append(check_ahead(pts3d, Extr2))

best_ind = num_ahead.index(max(num_ahead))
print(errs, num_ahead)
best_extr = Extr2s[:,:,best_ind]
np.savez('../data/extrinsics.npz', R1=np.identity(3), t1=np.zeros((3,1)), R2=best_extr[:,:3],t2=best_extr[:,3].reshape(3,1))

# 9. Scatter plot the correct 3D points

def set_scales(ax, scale, x_start, y_start, z_start):
    ax.set_xlim(x_start, x_start+scale)
    ax.set_ylim(y_start, y_start+scale)
    ax.set_zlim(z_start, z_start+scale)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
set_scales(ax, 1.5, -0.8, -0.6, 3.3)
ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])

ax.scatter(Extr1[:,3].T[0],Extr1[:,3].T[1],Extr1[:,3].T[2], marker='^')
ax.scatter(best_extr[:,3].T[0],best_extr[:,3].T[1],best_extr[:,3].T[2], marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz