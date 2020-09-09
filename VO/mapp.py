import numpy as np
from process_frame import VO_frontend
from g2o_optimizer import optimize

def add_ones(x):
    if len(x.shape) == 1:
        return np.concatenate([x,np.array([1.0])], axis=0)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Point(object):
    def __init__(self, mapp, loc, color, tid=None):
        self.pt = np.array(loc)
        self.frames = []
        self.idxs = []
        self.color = np.copy(color)
        self.id = mapp.add_point(self)
    def add_observation(self, frame, idx):
        assert frame.pts[idx] is None
        assert frame not in self.frames
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
    def homogeneous(self):
        return add_ones(self.pt)

class Frame(object):
    def __init__(self, mapp, image, K, pose=np.eye(4), tid=None):
        self.image = image
        self.K = K
        self.pose = pose
        self.frontend = VO_frontend()
        self.kps, self.des = self.frontend.get_keypoints(image)
        self.pts = [None]*len(self.kps)
        self.id = mapp.add_frame(self)
        self.SE3 = None
    @property
    def normalized_kps(self):
        if not hasattr(self, '_normalized_kps'):
            self._normalized_kps = np.dot(np.linalg.inv(self.K), add_ones(self.kps).T).T[:, :2]
        return self._normalized_kps

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.max_frame = 0
        self.max_point = 0
    def add_point(self, point):
        ret = self.max_point
        self.max_point = self.max_point + 1
        self.points.append(point)
        return ret
    def add_frame(self, frame):
        ret = self.max_frame
        self.max_frame = self.max_frame + 1
        self.frames.append(frame)
        return ret
    def optimize(self, local_window=20, fix_points=False, verbose=False, iterations=50):
        error = optimize(self.frames, self.points, local_window, fix_points, verbose, iterations)
        # print(error)
        return error
        # culled_pt_count = 0
        # for p in self.points:
        #     old_point = len(p.frames)
    def test(self):
        for f in self.frames[-1:]:
            print(f)
        # print(self.frames[:])