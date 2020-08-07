import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import glob
from matplotlib.animation import FFMpegWriter
from scipy import signal as sig
from scipy.spatial.distance import cdist


metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)


def harris(img, patch_size, kappa):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    sobel_y = sobel_x.T 
    Ix = sig.convolve2d(img, sobel_x, mode='same')
    Iy = sig.convolve2d(img, sobel_y, mode='same')
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    IxIy = Ix * Iy
    # rectangular unifoem window
    patch_weights = np.ones((patch_size, patch_size))/patch_size**2
    sumIxx = sig.convolve2d(Ixx, patch_weights, mode='same')
    sumIyy = sig.convolve2d(Iyy, patch_weights, mode='same')
    sumIxIy = sig.convolve2d(IxIy, patch_weights, mode='same')
    detM = sumIxx*sumIyy - sumIxIy*sumIxIy
    traceM = sumIxx * sumIyy
    scores = detM - (kappa * traceM**2)
    scores[scores < 0] = 0
    return scores


def ind2sub(array_shape, ind):
    rows = ind // array_shape[1]
    cols = ind % array_shape[1]
    return (rows, cols)


def select_keypoints(scores, num, r):
    keypoints = np.ones((2, num))
    for i in range(num):
        [row, col] = ind2sub(scores.shape, np.argmax(scores))
        keypoints[:, i] = [row, col]
        scores[max(0, row-r):min(scores.shape[0],row+r+1), max(0, col-r):min(scores.shape[1], col+r+1)] = \
            np.zeros((min(scores.shape[0],row+r+1)-max(0, row-r), min(scores.shape[1], col+r+1)-max(0, col-r)))
    return keypoints


def describe_keypoints(img, keypoints, r):
    descriptors = np.zeros(((2*r+1)**2, keypoints.shape[1]))
    padded_img = np.pad(img, (r, r), 'constant', constant_values=0)
    for i in range(keypoints.shape[1]):
        [row, col] = [int(keypoints[:, i][0]) + r, int(keypoints[:, i][1]) + r]
        # print(row, col)
        descriptors[:, i] = padded_img[row-r:row+r+1, col-r:col+r+1].reshape(((2*r+1)**2, ))
    return descriptors


def match_descriptors(des2, des1, lam):
    distances = cdist(des1.T, des2.T)
    # print(distances)
    min_distances = np.min(distances, axis=-1)
    # print(min_distances)
    min_distances_ind = np.argmin(distances, axis=-1)
    # print(min_distances_ind)
    sorted_dists = np.sort(min_distances)
    sorted_dists = sorted_dists[sorted_dists!=0]
    min_non_zero_dist = sorted_dists[1]
    min_distances_ind[min_distances >= lam * min_non_zero_dist] = 0
    # print(sorted_dists)
    unique_matches = np.zeros(min_distances_ind.shape)
    d, w = np.unique(min_distances_ind, return_index=True)
    # print(d, w)
    unique_matches[w] = min_distances_ind[w]
    return unique_matches


def plot_matches(img, matches, keypoints2, keypoints):
    pass

# np.random.seed(0)
# des1 = np.random.randint(10, size=(4,10))
# des2 = np.random.randint(10, size=(4,10))
# print(des1)
# print(des2)
# lam = 2
# z = match_descriptors(des2, des1, lam)
# print(z)


corner_patch_size = 9
harris_kappa = 0.08
num_keypoints = 200
descriptor_radius = 9
nonmaximum_supression_radius = 8
match_lambda = 4

img = mpimg.imread('data/000000.png')
print(img.shape)

# Corner Response (Harris)
harris_score = harris(img, corner_patch_size, harris_kappa)
keypoints = select_keypoints(harris_score, num_keypoints, nonmaximum_supression_radius)
descriptors = describe_keypoints(img, keypoints, descriptor_radius)
# for i in range(16):
#     plt.axis('off')
#     plt.subplot(4, 4, i+1)
#     plt.imshow(descriptors[:, i].reshape((19, 19)), cmap=plt.get_cmap('gray'))
# plt.show()

img2 = mpimg.imread('data/000001.png')
harris_score2 = harris(img2, corner_patch_size, harris_kappa)
keypoints2 = select_keypoints(harris_score2, num_keypoints, nonmaximum_supression_radius)
descriptors2 = describe_keypoints(img2, keypoints2, descriptor_radius)
# for i in range(16):
#     plt.axis('off')
#     plt.subplot(4, 4, i+1)
#     plt.imshow(descriptors2[:, i].reshape((19, 19)), cmap=plt.get_cmap('gray'))
# plt.show()

matches = match_descriptors(descriptors2, descriptors, match_lambda)

plt.figure(figsize=(15,5))
plt.axis('off')
plt.imshow(img2, cmap=plt.get_cmap('gray'))
plt.scatter(keypoints2[1], keypoints2[0], marker='x', color='r')
plt.plot(
    [keypoints[1, np.where(matches != 0)].squeeze(), keypoints2[1, np.array([matches[np.where(matches != 0)]]).astype('int')].squeeze()],
    [keypoints[0, np.where(matches != 0)].squeeze(), keypoints2[0, np.array([matches[np.where(matches != 0)]]).astype('int')].squeeze()],
    linewidth=4, color='blue'
    )
plt.show()

all_image_paths = sorted(glob.glob('data/*'))
# print(all_image_paths)
fig = plt.figure(figsize=(15,5))
with writer.saving(fig, "2.mp4", 100):
    for i in range(len(all_image_paths)):
        img = mpimg.imread(all_image_paths[i])
        harris_score = harris(img, corner_patch_size, harris_kappa)
        keypoints = select_keypoints(harris_score, num_keypoints, nonmaximum_supression_radius)
        descriptors = describe_keypoints(img, keypoints, descriptor_radius)
        if i != 0:
            matches = match_descriptors(descriptors, prev_descriptors, match_lambda)
            # plt.figure(figsize=(15,5))
            plt.axis('off')
            plt.imshow(img, cmap=plt.get_cmap('gray'))
            plt.scatter(keypoints[1], keypoints[0], marker='x', color='r')
            plt.plot(
                [prev_keypoints[1, np.where(matches != 0)].squeeze(), keypoints[1, np.array([matches[np.where(matches != 0)]]).astype('int')].squeeze()],
                [prev_keypoints[0, np.where(matches != 0)].squeeze(), keypoints[0, np.array([matches[np.where(matches != 0)]]).astype('int')].squeeze()],
                linewidth=4, color='blue'
                )
            writer.grab_frame()
            plt.clf()
            # plt.show()
        prev_descriptors = descriptors
        prev_keypoints = keypoints