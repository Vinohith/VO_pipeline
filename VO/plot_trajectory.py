import numpy as np
import OpenGL.GL as gl
import pangolin


state = np.load("traj_full.npy", allow_pickle=True)

w = 1024
h = 768

pangolin.CreateWindowAndBind('Map Viewer', w, h)
gl.glEnable(gl.GL_DEPTH_TEST)

scam = pangolin.OpenGlRenderState(
    pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
    pangolin.ModelViewLookAt(0, -10, -8,
                            0, 0, 0,
                            0, -1, 0))
handler = pangolin.Handler3D(scam)

# Create Interactive View in window
dcam = pangolin.CreateDisplay()
dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
dcam.SetHandler(handler)
# hack to avoid small Pangolin, no idea why it's *2
dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
dcam.Activate()

while not pangolin.ShouldQuit():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    dcam.Activate(scam)
    if state is not None:
        if state[0].shape[0] >= 2:
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawCameras(state[0][:-1])
        if state[0].shape[0] >= 1:
            gl.glColor3f(1.0, 1.0, 0.0)
            pangolin.DrawCameras(state[0][-1:])
        # if state[1].shape[0] != 0:
        #     gl.glPointSize(1)
        #     gl.glColor3f(1.0, 0.0, 0.0)
        #     pangolin.DrawPoints(state[1], state[2])
    pangolin.FinishFrame()