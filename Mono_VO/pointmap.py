import numpy as np
from process_frame import VO_frontend
from g2o_optimizer import optimize


class Point(object):
    def __init__(self, mapp, loc, color=(255,0,0), tid=None):
        self.pt = np.array(loc)
        self.frames = []
        self.idxs = []
        self.color = np.copy(color)
        # self.id = tid if tid is not None else mapp.add_point(self)
        self.id = mapp.add_point(self)
    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)


class Frame(object):
    def __init__(self, mapp, image, K, pose=np.eye(4), tid=None):
        self.image = image
        self.K = K
        self.pose = pose
        self.frontend = VO_frontend()
        self.kps, self.des = self.frontend.get_keypoints(image)
        self.pts = [None]*len(self.kps)
        self.id = mapp.add_frame(self)


class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.max_frame = 0
        self.max_point = 0
    def sequence(self):
        ret = {}
        ret['points'] = [{'id':p.id, 'pt':p.pt, 'color':pt.color} for p in self.points]
        ret['frames'] = []
        for f in self.frames:
            ret['frames'].append({
                'if':f.id, 'K':f.K, 'pose':f.pose, 'h':f.h, 'w':f.w,
                'kpts':f.kpts, 'des':f.des,
                'pts':[p.id if p is not None else -1 for p in f.pts]
            })
        ret['max_frame'] = self.max_frame
        ret['max_point'] = self.max_point
        return ret
    def desequence(self, ret):
        self.max_frame = ret['max_frame']
        self.max_point = ret['max_point']
        self.points = []
        self.frames = []
        pids = {}
        for p in ret['points']:
            pp = Point(self, p['pt'], p['color'], p['id'])
            self.points.append(pp)
            pids[p['id']] = pp
        for f in ret['frames']:
            ff = Frame(self, None, f['K'], f['pose'], f['id'])
            ff.w, ff.h = f['w'], f['h']
            ff.kpts = np.array(f['kpts'])
            ff.des = np.array(f['des'])
            for i, p in enumerate(f['pts']):
                if p != -1:
                    ff.pts[i] = pids[p]
            self.frames.append(ff)
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
        return error
        # culled_pt_count = 0
        # for p in self.points:
        #     old_point = len(p.frames)
    def test(self):
        for f in self.frames[-1:]:
            print(f)
        # print(self.frames[:])