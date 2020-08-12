class PinholeCamera:
    def __init__(self, width, height, focal, 
                 cx, cy, k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0, k3 = 0.0):
        self.width = width
        self.height = height
        self.focal = focal
        self.cx = cx 
        self.cy = cy
        self.d = [k1, k2, p1, p2, k3]   