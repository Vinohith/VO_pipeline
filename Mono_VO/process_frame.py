import cv2
import numpy as np

class VO_frontend(object):
    def __init__(self):
        self.detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=15.0, 
                qualityLevel=0.001, useHarrisDetector=False)
        self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    def get_keypoints(self, image):
        kps = self.detector.detect(image)
        kps, des = self.descriptor.compute(image, kps)
        # return kps, des
        return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
    def get_matches(self, image1, image2):
        idx1 = []
        idx2 = []
        matches = self.matcher.match(image1.des, image2.des)
        matches = sorted(matches, key = lambda x:x.distance)
        for match in matches:
            idx1.append(match.queryIdx)
            idx2.append(match.trainIdx)
        return idx1, idx2, matches
    def get_pose(self, image1, idx1, image2, idx2, K):
        Rt = np.eye(4)
        E, mask = cv2.findEssentialMat(image1.kps[idx1], image2.kps[idx2], K, cv2.RANSAC)
        _, rot, trans, mask = cv2.recoverPose(E, image1.kps[idx1], image2.kps[idx2], K)
        Rt[:3, :3] = rot
        Rt[:3, 3] = trans.squeeze()
        return Rt
    def get_triangulation(self, image1, idx1, image2, idx2):
        pts_4d = cv2.triangulatePoints(image1.pose[:3, :], image2.pose[:3, :],
                                       image1.kps[idx1].T, image2.kps[idx2].T)
        return pts_4d.T
