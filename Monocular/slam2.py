import cv2
from view import Display
from extractor import Frame, denormalize, match_frames, add_ones
import numpy as np
from pointmap import Map, Point

W, H = 1920 // 2, 1080 // 2
F = 270

K = np.array(([F, 0, W // 2], [0, F, H // 2], [0, 0, 1]))
Kinv = np.linalg.inv(K)

display = Display(W, H)
mapp = Map()
mapp.create_viewer()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    if len(mapp.frames) < 2:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    matches, idx1, idx2 = match_frames(f1, f2)
    Rt = extractPose(matches, f1, f2)

    f1.pose = np.dot(Rt, f2.pose)

    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)

def triangulate(pose1, pose2, pts1, pts2):
    """Triangulate points using two camera poses and corresponding image points."""
    pts4d = cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T
    return pts4d

if __name__ == '__main__':
    # Example usage
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame)
        display.paint(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

