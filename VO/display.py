from multiprocessing import Process, Queue
import pangolin
import OpenGL.GL as gl
import numpy as np

class Display3D(object):
  def __init__(self):
    self.state = None
    self.q = Queue()
    self.vp = Process(target=self.viewer_thread, args=(self.q,))
    self.vp.daemon = True
    self.vp.start()

  def viewer_thread(self, q):
    self.viewer_init(1024, 768)
    while 1:
      self.viewer_refresh(q)

  def viewer_init(self, w, h):
    # pangolin.CreateWindowAndBind('Map Viewer', w, h)
    # gl.glEnable(gl.GL_DEPTH_TEST)

    # self.scam = pangolin.OpenGlRenderState(
    #   pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
    #   pangolin.ModelViewLookAt(0, -10, -8,
    #                            0, 0, 0,
    #                            0, -1, 0))
    # self.handler = pangolin.Handler3D(self.scam)

    # # Create Interactive View in window
    # self.dcam = pangolin.CreateDisplay()
    # self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
    # self.dcam.SetHandler(self.handler)
    # # hack to avoid small Pangolin, no idea why it's *2
    # self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
    # self.dcam.Activate()
    pass


  def viewer_refresh(self, q):
    while not q.empty():
      self.state = q.get()
      np.save("traj_full2", self.state)

    # gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    # gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    # self.dcam.Activate(self.scam)

    # # print("self.state : ", len(self.state[0]), len(self.state[1]), len(self.state[2]))
    # if self.state is not None:
    #   if self.state[0].shape[0] >= 2:
    #     # draw poses
    #     gl.glColor3f(0.0, 1.0, 0.0)
    #     pangolin.DrawCameras(self.state[0][:-1])

    #   if self.state[0].shape[0] >= 1:
    #     # draw current pose as yellow
    #     gl.glColor3f(1.0, 1.0, 0.0)
    #     pangolin.DrawCameras(self.state[0][-1:])

    #   if self.state[1].shape[0] != 0:
    #     # draw keypoints
    #     gl.glPointSize(1)
    #     gl.glColor3f(1.0, 0.0, 0.0)
    #     pangolin.DrawPoints(self.state[1], self.state[2])

    # pangolin.FinishFrame()

  def paint(self, mapp):
    # print("Painting point")
    if self.q is None:
      return

    poses, pts, colors = [], [], []
    for f in mapp.frames:
      # invert pose for display only
      poses.append(np.linalg.inv(f.pose))
    for p in mapp.points:
      pts.append(p.pt)
      colors.append(p.color)
    # print(np.array(poses), np.array(pts).shape, np.expand_dims(np.array(colors)/256.0, axis=-1).shape)
    self.q.put((np.array(poses), np.array(pts), np.expand_dims(np.array(colors)/256.0, axis=-1)))