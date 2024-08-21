from multiprocessing import Process, Queue
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import OpenGL.GL as gl
import OpenGL.GLU as glu


class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = None

    def create_viewer(self):
        self.q = Queue()
        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()

    def viewer_thread(self, q):
        self.viewer_init(1280, 720)
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pygame.init()
        pygame.display.set_mode((w, h), DOUBLEBUF | OPENGL)
        glu.gluPerspective(45, (w / h), 0.1, 10000)
        gl.glEnable(gl.GL_DEPTH_TEST)
        self.set_camera()

    def set_camera(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        glu.gluLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.set_camera()

        gl.glLineWidth(1)
        gl.glColor3f(0.0, 1.0, 0.0)
        self.draw_cameras(self.state[0])

        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        self.draw_points(self.state[1])

        pygame.display.flip()

    def draw_cameras(self, poses):
        for pose in poses:
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3fv(pose[0:3, 3]) 
            gl.glVertex3fv(pose[0:3, 3] + pose[0:3, 2]) 
            gl.glEnd()

    def draw_points(self, points):
        gl.glBegin(gl.GL_POINTS)
        for point in points:
            gl.glVertex3fv(point)
        gl.glEnd()

    def display(self):
        if self.q is None:
            return
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
        self.q.put((np.array(poses), np.array(pts)))


class Point(object):
    def __init__(self, mapp, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)
