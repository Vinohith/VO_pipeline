import numpy as np
import OpenGL.GL as gl
import pangolin
from scipy.spatial.transform import Rotation as R

poses = []
traj_file = np.loadtxt("trajectory.txt")
for traj in traj_file:
	Twr = np.eye(4)
	Twr[:3, :3] = R.from_quat(traj[-4:]).as_matrix()
	Twr[:3, 3] = traj[1:4].T
	poses.append(Twr)
poses = np.array(poses)
print(len(poses))
# print(poses)

# for pose in poses:
# 	print("******** : ")
# 	print(np.linalg.inv(pose))
# 	print(pose)
# 	Ow = pose[:3, 3]
# 	Xw = 0.1*pose[:3, :3].dot(np.array([1, 0, 0]).T) + Ow
# 	Yw = 0.1*pose[:3, :3].dot(np.array([0, 1, 0]).T) + Ow
# 	Zw = 0.1*pose[:3, :3].dot(np.array([0, 0, 1]).T) + Ow
# 	print(Ow, Xw, Yw, Zw)

pangolin.CreateWindowAndBind("Trajectory Viewer", 1024, 768)
gl.glEnable(gl.GL_DEPTH_TEST)
gl.glEnable(gl.GL_BLEND)
gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

scam = pangolin.OpenGlRenderState(
				pangolin.ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
				pangolin.ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0))
handler = pangolin.Handler3D(scam)

dcam = pangolin.CreateDisplay()
dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0 / 768.0)
dcam.SetHandler(handler)

while not pangolin.ShouldQuit():
	gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
	gl.glClearColor(1.0, 1.0, 1.0, 1.0)
	dcam.Activate(scam)
	gl.glLineWidth(2)
	Ow = np.array([0.0, 0.0, 0.0]).T
	for pose in poses:
		Ow = pose[:3, 3]
		Xw = pose[:3, :3].dot(np.array([0.1, 0, 0]).T) + Ow
		Yw = pose[:3, :3].dot(np.array([0, 0.1, 0]).T) + Ow
		Zw = pose[:3, :3].dot(np.array([0, 0, 0.1]).T) + Ow
		gl.glBegin(gl.GL_LINES)
		gl.glColor3f(1.0, 0.0, 0.0)
		gl.glVertex3d(Ow[0], Ow[1], Ow[2])
		gl.glVertex3d(Xw[0], Xw[1], Xw[2])
		gl.glColor3f(0.0, 1.0, 0.0)
		gl.glVertex3d(Ow[0], Ow[1], Ow[2])
		gl.glVertex3d(Yw[0], Yw[1], Yw[2])
		gl.glColor3f(0.0, 0.0, 1.0)
		gl.glVertex3d(Ow[0], Ow[1], Ow[2])
		gl.glVertex3d(Zw[0], Zw[1], Zw[2])
		gl.glEnd()
	pangolin.FinishFrame()