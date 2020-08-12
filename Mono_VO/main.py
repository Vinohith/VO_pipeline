from visual_odometry import VisualOdometry
from camera import PinholeCamera
import cv2
import glob
import numpy as np

camera = PinholeCamera(1241.0, 376.0, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(camera, "00.txt")


img_paths = sorted(glob.glob("data/*"))
traj = np.zeros((600,600,3), dtype=np.uint8)
frame_id = 0
for path in img_paths:
    img = cv2.imread(path, 0)
    vo.process_image(img, frame_id)
    t = vo.t
    if frame_id > 1:
        x, y, z = t[0], t[1], t[2]
    else:
        x, y, z = 0, 0, 0
    draw_x, draw_y = int(x)+290, int(z)+90
    true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90
    
    cv2.circle(traj, (true_x,true_y), 1, (255,0,0), 2)
    cv2.circle(traj, (draw_x, draw_y), 1, (0,0,255), 2)
    cv2.imshow('Road facing camera', img)
    cv2.imshow('Trajectory', traj)
    cv2.waitKey(1)
    frame_id += 1
cv2.imwrite('map2.png', traj)