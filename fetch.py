import os 
cw = os.getcwd()
import cv2
import numpy as np
path = cw + "/dashpic.png"
image_1 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
image_2 = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(nfeatures=1000)

keypoints1, descriptors1 = orb.detectAndCompute(image_1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image_2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1,descriptors2)

points1 = np.array([keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32)
points2 = np.array([keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

points2, status, err = cv2.calcOpticalFlowPyrLK(image_1, image_2, points1, None, **lk_params)

good_new = points2[status.ravel() == 1]
good_old = points1[status.ravel() == 1]


output_img = cv2.cvtColor(image_2, cv2.COLOR_GRAY2BGR)
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    cv2.line(output_img, (int(a), int(b)), (int(c), int(d)), color=(0, 255, 0), thickness=2)
    cv2.circle(output_img, (int(a), int(b)), radius=5, color=(0, 0, 255), thickness=-1)

cv2.imshow('Optical Flow', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()




