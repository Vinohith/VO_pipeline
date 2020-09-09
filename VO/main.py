import numpy as np
import cv2
from mapp import Frame, Map, Point
import glob
from process_frame import VO_frontend
from display import Display3D


def process_frame(image, pose=None, verts=None):
    frame = Frame(mapp, image, K)
    print()
    print("******** Frame %d *********" % frame.id)
    # print("Keypoiints : ")
    # print(frame.kps)
    # print(frame.normalized_kps)
    if frame.id == 0:
        return 0, 0, 0
    frame1 = mapp.frames[-1]
    frame2 = mapp.frames[-2]
    # idx1, idx2, Rt = frontend.get_matches(frame1, frame2)
    idx1, idx2, matches = frontend.get_matches(frame1, frame2)
    Rt, E = frontend.get_pose(frame1, idx1, frame2, idx2, K)

    # for i,idx in enumerate(idx2):
    #     if frame2.pts[idx] is not None and frame1.pts[idx1[i]] is None:
    #         frame2.pts[idx].add_observation(frame1, idx1[i])
    frame1.pose = np.dot(Rt, frame2.pose)
    
    pts_4d = frontend.get_triangulation(frame1, idx1, frame2, idx2)
    pts_3d = pts_4d[:, :4] / pts_4d[:, 3:]
    
    new_pts_count = 0
    for i, p in enumerate(pts_3d):
        bool_pts1 = False
        bool_pts2 = False
        pl1 = np.dot(frame1.pose, p)
        pl2 = np.dot(frame2.pose, p)
        # print(pl1, pl2)
        if pl1[2] < 0 or pl2[2] < 0:
            continue
        pp1 = np.dot(frame1.K, pl1[:3])
        pp2 = np.dot(frame2.K, pl2[:3])
        pp1 = (pp1[0:2] / pp1[2]) - frame1.kps[idx1[i]]
        pp2 = (pp2[0:2] / pp2[2]) - frame2.kps[idx2[i]]
        pp1 = np.sum(pp1**2)
        pp2 = np.sum(pp2**2)
        # print(pp1, pp2)
        if pp1 > 2 or pp2 > 2:
            continue
        try:
            color = img[int(round(frame1.kps[idx1[i],1])), int(round(frame1.kps[idx1[i],0]))]
        except IndexError:
            color = (255,0,0)
        pt = Point(mapp, p[0:3], color)
        if frame2.pts[idx2[i]] is None:
            pt.add_observation(frame2, idx2[i])
            bool_pts2 = True
        if frame1.pts[idx1[i]] is None:
            pt.add_observation(frame1, idx1[i])
            bool_pts1 = True
        if bool_pts1 and bool_pts2:
            new_pts_count += 1

    print("Adding:   %d new points" % (new_pts_count))
    if frame.id >= 4 and frame.id%5 == 0:
        # print("Optimizer")
        err = mapp.optimize(iterations=50)
        print("Optimize: %f units of error" % err)
    print("Map:      %d points, %d frames" % (len(mapp.points), len(mapp.frames)))
    return frame1.pose[:3, 3]


mapp = Map()
frontend = VO_frontend()
img_paths = sorted(glob.glob("/home/vinohith/Downloads/kitti00/kitti/00/image_0/*"))
W = int(1241.0)
H = int(376.0)
K = np.array([[718.8560,0,607.1928],[0,718.8560,185.2157],[0,0,1]])
disp3d = Display3D()

for path in img_paths:
    img = cv2.imread(path, 0)
    x, y, z = process_frame(img)
    if disp3d is not None:
        disp3d.paint(mapp)