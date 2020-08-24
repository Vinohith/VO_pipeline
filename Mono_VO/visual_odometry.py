import cv2
import numpy as np
import matplotlib.pyplot as plt


class VisualOdometry:
    def __init__(self, camera, annotations):
        self.prev_frame = None
        self.current_frame = None
        self.prev_keypoints = None 
        self.current_keypoints = None
        self.prev_des = None 
        self.current_des = None
        self.matches = None
        self.detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=15.0, 
                qualityLevel=0.001, useHarrisDetector=False)
        self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        self.focal = camera.focal
        self.projection_center = (camera.cx, camera.cy)
        self.frame_number = 0
        self.R = np.eye(3)
        self.t = np.zeros((3,1))
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        with open(annotations) as f:
            self.annotations = f.readlines()


    def process_image(self, current_frame, frame_id):
        # print("*********************")
        # print(self.R)
        # print(self.t)
        # print("*********************")
        self.current_frame = current_frame
        if self.frame_number == 0:
            self.current_keypoints, self.current_des = self.get_keypoints()
            # img2 = cv2.drawKeypoints(self.current_frame, self.current_keypoints, None)
            # cv2.imshow('keypoints', img2)
            # cv2.waitKey(0)
            self.frame_number = 1

        elif self.frame_number == 1:
            self.current_keypoints, self.current_des = self.get_keypoints()
            kp1, kp2 = self.find_matches()
            # img2 = cv2.drawMatches(self.prev_frame, self.prev_keypoints, self.current_frame, self.current_keypoints, self.matches[:100], None, flags=2)
            # cv2.imshow('keypoints', img2)
            # cv2.waitKey(0)
            E, mask = cv2.findEssentialMat(kp2, kp1, self.focal, self.projection_center, cv2.RANSAC)
            _, rot, trans, mask = cv2.recoverPose(E, kp2, kp1, focal = self.focal, pp = self.projection_center)
            scale = self.get_absolute_scale(frame_id)
            self.R = rot.dot(self.R)
            self.t = self.t + scale * self.R.dot(trans)
            # self.R = rot.dot(self.R)

        self.prev_frame = self.current_frame
        self.prev_keypoints = self.current_keypoints
        self.prev_des = self.current_des


    def get_absolute_scale(self, frame_id):
        ss = self.annotations[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))


    def get_keypoints(self):
        kps = self.detector.detect(self.current_frame)
        kps, des = self.descriptor.compute(self.current_frame, kps)
        return kps, des


    def find_matches(self):
        kp1 = []
        kp2 = []
        self.matches = self.matcher.match(self.prev_des, self.current_des)
        self.matches = sorted(self.matches, key = lambda x:x.distance)
        for match in self.matches:
            kp1.append([self.prev_keypoints[match.queryIdx].pt[0], self.prev_keypoints[match.queryIdx].pt[1]])
            kp2.append([self.current_keypoints[match.trainIdx].pt[0], self.current_keypoints[match.trainIdx].pt[1]])
        return np.array(kp1), np.array(kp2)